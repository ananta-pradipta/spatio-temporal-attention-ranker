"""StockMixer baseline trainer using the v2 protocol (matches RAG-STAR).

Same panel, masks, fold definitions, embargo, seeds, loss, and metrics
as ``src.v2.training.train_dow_epistar``. Wraps the vendored StockMixer
architecture (SJTU-DMTai, AAAI 2024, Fan and Shen).

Run:
    python -m src.baselines.train_stockmixer_v2 --fold 1 --seed 42

Output: results/baselines_244/stockmixer_v2/fold{F}_seed{S}.json (+ npz).
"""
from __future__ import annotations

import argparse
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from src.baselines.stockmixer_adapter import StockMixerAdapter
from src.baselines.v2_runner import (
    V2BaselineConfig,
    build_age_features,
    build_masks,
    build_panel,
    cs_mse_loss,
    evaluate_predictions,
    fold_split,
    save_result,
    set_seeds,
    standardize_features,
    warmup_cosine_lr,
)


@dataclass
class StockMixerV2Config(V2BaselineConfig):
    output_dir: str = "results/baselines_244/stockmixer_v2"
    # StockMixer's stride-2 conv requires even W; use 16 (their default).
    temporal_window: int = 16
    # NoGraphMixer hidden dim. Original paper: m in {8, 12, 16, 20, 24, 25};
    # we use 12 as a tuned choice for our ~244-ticker universe (small lift
    # observed vs m=8; larger m didn't help on val).
    market_hidden: int = 12


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, choices=[1, 2, 3, 4, 5], required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--panel_kind", type=str, default="biotech",
                   choices=["biotech", "lattice_native"])
    p.add_argument("--two_regime_val", action="store_true")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--panel_end", type=str, default=None)
    args = p.parse_args()

    cfg = StockMixerV2Config(fold=args.fold, seed=args.seed)
    if args.smoke:
        cfg.epochs = 2
    cfg.panel_kind = args.panel_kind
    cfg.two_regime_val = args.two_regime_val
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.panel_end:
        cfg.panel_end = args.panel_end
    elif args.panel_kind == "lattice_native":
        cfg.panel_end = "2025-12-31"

    set_seeds(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[StockMixer-v2] fold={cfg.fold} seed={cfg.seed} device={device}")

    x_raw, y, tickers, dates = build_panel(cfg)
    T, N, Fdim = x_raw.shape
    print(f"[StockMixer-v2] panel: T={T} N={N} F={Fdim}")
    if N < 50:
        raise RuntimeError("Panel too small")

    mm = build_masks(cfg, dates, tickers)
    tradable = mm["tradable_mask"]
    loss_mask = mm["loss_mask"]
    hist20 = mm["history_valid_20d"]
    hist60 = mm["history_valid_60d"]

    train_idx, val_idx, test_idx = fold_split(cfg, dates)
    print(f"[StockMixer-v2] fold {cfg.fold}: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    x = standardize_features(x_raw, tradable, train_idx)
    age_feat = build_age_features(tradable, hist20, hist60)
    age_days = age_feat[..., 0].astype(np.int64)

    x_t = torch.from_numpy(x).to(device)
    y_t = torch.from_numpy(y).to(device)

    W = cfg.temporal_window
    model = StockMixerAdapter(
        stocks_pad=N, time_steps=W, channels=Fdim, market=cfg.market_hidden,
    ).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    total_steps = cfg.epochs * max(1, len(train_idx))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optim, lr_lambda=lambda s: warmup_cosine_lr(s, cfg.warmup_steps, total_steps)
    )

    def run_split(idx: np.ndarray, train_: bool) -> tuple[float, np.ndarray, np.ndarray]:
        model.train(train_)
        losses = []
        y_hat_all = np.zeros((T, N), dtype=np.float32)
        emask = np.zeros((T, N), dtype=bool)
        for t in idx:
            t = int(t)
            if t < W - 1:
                continue
            m_np = tradable[t]
            if m_np.sum() < 3:
                continue
            m = torch.from_numpy(m_np).to(device)
            x_win = x_t[t - W + 1: t + 1].permute(1, 0, 2)  # [N, W, F]
            y_hat = model(x_win, m)                         # [N]
            l = cs_mse_loss(y_hat, y_t[t], torch.from_numpy(loss_mask[t]).to(device))
            if train_:
                optim.zero_grad()
                l.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optim.step()
                scheduler.step()
            losses.append(float(l.item()))
            y_hat_all[t] = y_hat.detach().cpu().numpy()
            emask[t] = loss_mask[t]
        return (float(np.mean(losses)) if losses else float("nan"), y_hat_all, emask)

    history: list = []
    best_val_ic = -1e9
    best_state = None
    patience = 0
    for epoch in range(cfg.epochs):
        t0 = time.time()
        np.random.seed(cfg.seed + epoch)
        perm = np.random.permutation(train_idx)
        train_loss, _, _ = run_split(perm, train_=True)
        val_loss, val_yhat, val_mask = run_split(val_idx, train_=False)
        val_metrics = evaluate_predictions(val_yhat, y, val_mask, age_days)
        dt = time.time() - t0
        improved = val_metrics["ic"] > best_val_ic + 1e-5
        print(f"[StockMixer-v2] epoch {epoch}: train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} val_ic={val_metrics['ic']:+.4f} "
              f"({dt:.1f}s)" + ("  *best*" if improved else ""))
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_ic": val_metrics["ic"],
            "val_rank_ic": val_metrics["rank_ic"],
            "time_sec": round(dt, 2),
        })
        if improved:
            best_val_ic = val_metrics["ic"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= cfg.early_stop_patience:
                print(f"[StockMixer-v2] early stop epoch {epoch} best_val_ic={best_val_ic:+.4f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    _, test_yhat, test_mask = run_split(test_idx, train_=False)
    test_metrics = evaluate_predictions(test_yhat, y, test_mask, age_days)
    val_metrics_final = evaluate_predictions(val_yhat, y, val_mask, age_days)

    print(f"[StockMixer-v2] TEST ic={test_metrics['ic']:+.4f} rank_ic={test_metrics['rank_ic']:+.4f}")

    out_path = save_result(
        out_dir=Path(cfg.output_dir),
        fold=cfg.fold, seed=cfg.seed,
        model_name="StockMixer (v2 protocol)",
        test_metrics=test_metrics,
        val_metrics=val_metrics_final,
        test_y_hat=test_yhat,
        test_eval_mask=test_mask,
        history=history,
        config=asdict(cfg),
        n_panel=(T, N, Fdim),
        n_train=len(train_idx), n_val=len(val_idx), n_test=len(test_idx),
        y_true=y, tickers=tickers, dates=dates,
        age_days=age_days, tradable_mask=tradable,
    )
    print(f"[StockMixer-v2] wrote {out_path}")


if __name__ == "__main__":
    main()
