"""MASTER baseline trainer using the v2 protocol (matches RAG-STAR).

Same panel, masks, fold definitions, embargo, seeds, loss, and metrics
as ``src.v2.training.train_dow_epistar``. The only difference from
RAG-STAR is the model: this script wraps the vendored MASTER
architecture (SJTU-DMTai, AAAI 2024, Li et al.) instead of the
RAG-STAR architecture.

Run:
    python -m src.baselines.train_master_v2 --fold 1 --seed 42

Output: results/baselines_244/master_v2/fold{F}_seed{S}.json (+ npz).
"""
from __future__ import annotations

import argparse
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.baselines.master_adapter import MASTERAdapter
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
class MASTERV2Config(V2BaselineConfig):
    output_dir: str = "results/baselines_244/master_v2"
    # MASTER-specific.
    d_model: int = 128
    t_nhead: int = 4
    s_nhead: int = 4
    T_dropout_rate: float = 0.1
    S_dropout_rate: float = 0.1
    beta: float = 5.0


def build_market_signature(x_raw: np.ndarray, mask: np.ndarray, train_idx: np.ndarray) -> np.ndarray:
    """7-dim daily market regime signature (causal cross-sectional summary).

    [mean_return, dispersion, mean_vol, vol_dispersion, top_vol_decile,
     pct_negative, active_count_ratio]
    All standardised on the training fold.
    """
    log_ret = x_raw[..., 0]
    rv20 = x_raw[..., 5]
    t_total, n = log_ret.shape
    sig = np.zeros((t_total, 7), dtype=np.float32)
    for t in range(t_total):
        m = mask[t]
        if m.sum() < 5:
            continue
        r = log_ret[t, m]
        v = rv20[t, m]
        sig[t, 0] = float(np.mean(r))
        sig[t, 1] = float(np.std(r))
        sig[t, 2] = float(np.mean(v))
        sig[t, 3] = float(np.std(v))
        sig[t, 4] = float(np.quantile(v, 0.9))
        sig[t, 5] = float((r < 0).mean())
        sig[t, 6] = float(m.sum()) / float(n)
    sig_train = sig[train_idx]
    finite = np.isfinite(sig_train).all(axis=1)
    mu = sig_train[finite].mean(axis=0)
    sd = sig_train[finite].std(axis=0).clip(min=1e-6)
    z = (sig - mu) / sd
    return np.where(np.isfinite(z), z, 0.0).astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, choices=[1, 2, 3], required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    cfg = MASTERV2Config(fold=args.fold, seed=args.seed)
    if args.smoke:
        cfg.epochs = 2

    set_seeds(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[MASTER-v2] fold={cfg.fold} seed={cfg.seed} device={device}")

    x_raw, y, tickers, dates = build_panel(cfg)
    T, N, Fdim = x_raw.shape
    print(f"[MASTER-v2] panel: T={T} N={N} F={Fdim}")
    if N < 50:
        raise RuntimeError("Panel too small")

    mm = build_masks(cfg, dates, tickers)
    tradable = mm["tradable_mask"]
    loss_mask = mm["loss_mask"]
    hist20 = mm["history_valid_20d"]
    hist60 = mm["history_valid_60d"]

    train_idx, val_idx, test_idx = fold_split(cfg, dates)
    print(f"[MASTER-v2] fold {cfg.fold}: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    x = standardize_features(x_raw, tradable, train_idx)

    age_feat = build_age_features(tradable, hist20, hist60)
    age_days = age_feat[..., 0].astype(np.int64)

    # 7-d daily market regime signature for MASTER's gate.
    sigs_z = build_market_signature(x_raw, tradable, train_idx)
    d_gate = sigs_z.shape[1]
    print(f"[MASTER-v2] market regime gate: {d_gate} dims")

    x_t = torch.from_numpy(x).to(device)
    y_t = torch.from_numpy(y).to(device)
    mask_t = torch.from_numpy(tradable).to(device)
    sigs_z_t = torch.from_numpy(sigs_z).to(device)

    model = MASTERAdapter(
        d_feat=Fdim, d_model=cfg.d_model,
        t_nhead=cfg.t_nhead, s_nhead=cfg.s_nhead,
        gate_input_start_index=Fdim,
        gate_input_end_index=Fdim + d_gate,
        T_dropout_rate=cfg.T_dropout_rate,
        S_dropout_rate=cfg.S_dropout_rate,
        beta=cfg.beta,
    ).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    total_steps = cfg.epochs * max(1, len(train_idx))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optim, lr_lambda=lambda s: warmup_cosine_lr(s, cfg.warmup_steps, total_steps)
    )

    W = cfg.temporal_window

    def run_split(idx: np.ndarray, train_: bool) -> tuple[float, np.ndarray, np.ndarray]:
        model.train(train_)
        losses = []
        y_hat_all = np.zeros((T, N), dtype=np.float32)
        emask = np.zeros((T, N), dtype=bool)
        for t in idx:
            t = int(t)
            if t < W - 1:
                continue
            m = mask_t[t]
            if m.sum() < 3:
                continue
            active_idx = m.nonzero(as_tuple=True)[0]
            stock_feat = x_t[t - W + 1: t + 1, active_idx, :].transpose(0, 1)  # [A, W, F]
            A = active_idx.shape[0]
            market_b = sigs_z_t[t - W + 1: t + 1].unsqueeze(0).expand(A, W, d_gate)
            inp = torch.cat([stock_feat, market_b], dim=-1)
            y_hat = model(inp)
            y_full = torch.zeros(N, device=device, dtype=y_hat.dtype)
            y_full[active_idx] = y_hat
            l = cs_mse_loss(y_full, y_t[t], torch.from_numpy(loss_mask[t]).to(device))
            if train_:
                optim.zero_grad()
                l.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optim.step()
                scheduler.step()
            losses.append(float(l.item()))
            y_hat_all[t] = y_full.detach().cpu().numpy()
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
        print(f"[MASTER-v2] epoch {epoch}: train_loss={train_loss:.4f} "
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
                print(f"[MASTER-v2] early stop epoch {epoch} best_val_ic={best_val_ic:+.4f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    _, test_yhat, test_mask = run_split(test_idx, train_=False)
    test_metrics = evaluate_predictions(test_yhat, y, test_mask, age_days)
    val_metrics_final = evaluate_predictions(*run_split(val_idx, train_=False)[1:], age_days=age_days) if False else evaluate_predictions(val_yhat, y, val_mask, age_days)

    print(f"[MASTER-v2] TEST ic={test_metrics['ic']:+.4f} rank_ic={test_metrics['rank_ic']:+.4f}")

    out_path = save_result(
        out_dir=Path(cfg.output_dir),
        fold=cfg.fold, seed=cfg.seed,
        model_name="MASTER (v2 protocol)",
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
    print(f"[MASTER-v2] wrote {out_path}")


if __name__ == "__main__":
    main()
