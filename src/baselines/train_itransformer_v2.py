"""iTransformer baseline trainer using the v2 protocol (matches RAG-STAR).

Same panel, masks, fold definitions, embargo, seeds, loss, and metrics
as ``src.v2.training.train_dow_epistar``. The only difference from
RAG-STAR is the model: this script wraps the vendored iTransformer
architecture (Liu, Hu, Liu, Zhou, Li, Long, ICLR 2024).

Reference code source vendored at ``src/baselines/vendored/itransformer/``
is adapted from https://github.com/thuml/iTransformer (MIT).

iTransformer replaces the failed PatchTST attempt as the
"Pure time-series transformer" baseline family in the paper. Its
inverted formulation -- variates as tokens, attention across variates
-- aligns with cross-sectional ranking because every ticker can attend
to every other ticker on the active day. PatchTST's channel-independent
design encoded each ticker in isolation, which is structurally
incompatible with cross-sectional MSE.

Hyperparameters: AdamW + warmup-cosine schedule, fp16 autocast, and
gradient clipping conventions of every other v2 baseline so that
fairness only requires controlling the architecture. Project-side
defaults from the task brief: d_model=128, n_heads=4, d_ff=256,
e_layers=2.

Loss: pure cross-sectional MSE on z-scored 5d forward log returns
(``cs_mse_loss``). iTransformer has no auxiliary loss, so this
matches MASTER / StockMixer / DySTAGE / PatchTST exactly.

Run:
    python -m src.baselines.train_itransformer_v2 --fold 1 --seed 42

Output: results/baselines_244/itransformer_v2/fold{F}_seed{S}.json (+ npz).
"""
from __future__ import annotations

import argparse
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from src.baselines.itransformer_adapter import (
    ITransformerAdapter,
    ITransformerHyperparams,
)
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
class ITransformerV2Config(V2BaselineConfig):
    """Top-level config bundling the v2 protocol + iTransformer-specific knobs.

    iTransformer hyperparameter rationale:
      - d_model=128, n_heads=4, d_ff=256, e_layers=2 follow the task
        brief's project-side adaptation: smaller than the paper's long-
        horizon forecasting configurations because our variate
        dimension is the per-day active ticker count (~150-200), not
        thousands of multivariate series.
      - dropout=0.1 matches the paper's default.
      - use_norm=False because the v2 protocol already standardises
        features with training-fold statistics; the paper's non-stationary
        normalisation would re-scale per (day, lookback) and strip the
        cross-sectional level information that the ranking head needs
        (mirroring why we set RevIN=False for PatchTST).
      - pred_len=1 because we produce a single ranking score per ticker.
    """
    output_dir: str = "results/baselines_244/itransformer_v2"
    # iTransformer architecture.
    d_model: int = 128
    n_heads: int = 4
    d_ff: int = 256
    e_layers: int = 2
    dropout: float = 0.1
    activation: str = "gelu"
    use_norm: bool = False


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, choices=[1, 2, 3], required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true",
                   help="Limit to 2 epochs and abbreviated output.")
    p.add_argument("--max_epochs", type=int, default=None,
                   help="Override config.epochs (e.g. 1 for a smoke check).")
    args = p.parse_args()

    cfg = ITransformerV2Config(fold=args.fold, seed=args.seed)
    if args.smoke:
        cfg.epochs = 2
    if args.max_epochs is not None:
        cfg.epochs = int(args.max_epochs)

    set_seeds(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[iTransformer-v2] fold={cfg.fold} seed={cfg.seed} device={device}")

    x_raw, y, tickers, dates = build_panel(cfg)
    T, N, Fdim = x_raw.shape
    print(f"[iTransformer-v2] panel: T={T} N={N} F={Fdim}")
    if N < 50:
        raise RuntimeError("Panel too small")

    mm = build_masks(cfg, dates, tickers)
    tradable = mm["tradable_mask"]
    loss_mask = mm["loss_mask"]
    hist20 = mm["history_valid_20d"]
    hist60 = mm["history_valid_60d"]

    train_idx, val_idx, test_idx = fold_split(cfg, dates)
    print(f"[iTransformer-v2] fold {cfg.fold}: "
          f"train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    x = standardize_features(x_raw, tradable, train_idx)
    age_feat = build_age_features(tradable, hist20, hist60)
    age_days = age_feat[..., 0].astype(np.int64)

    x_t = torch.from_numpy(x).to(device)
    y_t = torch.from_numpy(y).to(device)

    W = cfg.temporal_window

    hp = ITransformerHyperparams(
        d_feat=Fdim,
        context_window=W,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        d_ff=cfg.d_ff,
        e_layers=cfg.e_layers,
        dropout=cfg.dropout,
        activation=cfg.activation,
        use_norm=cfg.use_norm,
        pred_len=1,
    )
    model = ITransformerAdapter(hp).to(device)
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    total_steps = cfg.epochs * max(1, len(train_idx))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optim, lr_lambda=lambda s: warmup_cosine_lr(s, cfg.warmup_steps, total_steps)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

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
            active_idx = np.flatnonzero(m_np)
            active_t = torch.from_numpy(active_idx).to(device)
            # Whole active panel feeds through iTransformer in one
            # forward pass so cross-ticker attention can run; this is
            # the load-bearing structural reason iTransformer fits the
            # cross-sectional task.
            x_win = x_t[t - W + 1: t + 1, active_t, :].transpose(0, 1)  # (A, W, F)
            y_target_full = y_t[t]                                       # (N,)
            lmask_t = torch.from_numpy(loss_mask[t]).to(device)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                y_hat_active = model(x_win)                              # (A,)
                y_full = torch.zeros(N, device=device, dtype=y_hat_active.dtype)
                y_full[active_t] = y_hat_active
                cs_loss = cs_mse_loss(y_full, y_target_full, lmask_t)

            if train_:
                optim.zero_grad()
                scaler.scale(cs_loss).backward()
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optim)
                scaler.update()
                scheduler.step()
            losses.append(float(cs_loss.item()))
            y_hat_all[t] = y_full.detach().float().cpu().numpy()
            emask[t] = loss_mask[t]
        return (float(np.mean(losses)) if losses else float("nan"),
                y_hat_all, emask)

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
        print(f"[iTransformer-v2] epoch {epoch}: train_loss={train_loss:.4f} "
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
                print(f"[iTransformer-v2] early stop epoch {epoch} "
                      f"best_val_ic={best_val_ic:+.4f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    _, test_yhat, test_mask = run_split(test_idx, train_=False)
    test_metrics = evaluate_predictions(test_yhat, y, test_mask, age_days)
    val_metrics_final = evaluate_predictions(val_yhat, y, val_mask, age_days)

    print(f"[iTransformer-v2] TEST ic={test_metrics['ic']:+.4f} "
          f"rank_ic={test_metrics['rank_ic']:+.4f} "
          f"ndcg10={test_metrics['ndcg10']:.4f} "
          f"ndcg50={test_metrics['ndcg50']:.4f}")

    out_path = save_result(
        out_dir=Path(cfg.output_dir),
        fold=cfg.fold, seed=cfg.seed,
        model_name="iTransformer (v2 protocol)",
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
    print(f"[iTransformer-v2] wrote {out_path}")


if __name__ == "__main__":
    main()
