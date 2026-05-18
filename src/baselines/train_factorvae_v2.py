"""FactorVAE baseline trainer using the v2 protocol (matches RAG-STAR).

Same panel, masks, fold definitions, embargo, seeds, and metrics as
``src.v2.training.train_dow_epistar``. The only difference from
RAG-STAR is the model: this script wraps the vendored FactorVAE
architecture (Duan, Wang, Zhang, Li, AAAI 2022).

Reference code source vendored at ``src/baselines/vendored/factorvae/``
is adapted from https://github.com/x7jeon8gi/FactorVAE (unofficial
PyTorch implementation; the original authors did not release code).

Hyperparameters: we keep the AdamW + warmup-cosine schedule, fp16
autocast, and gradient clipping conventions of every other v2 baseline
so that fairness only requires controlling the architecture. The
FactorVAE-specific knobs default to the AAAI 2022 paper's CSI 300
configuration: K = 8 factors, GRU hidden 64, soft-portfolio set
M = 64 (we match the universe size more closely than the paper's 128
because our active count per day is ~150-200 active biotechs).

Loss: cross-sectional MSE on z-scored 5d forward log returns (the v2
protocol's ``cs_mse_loss``) + lambda * FactorVAE ELBO. Lambda defaults
to 0.1 which keeps the score-head dominant for ranking while still
training the factor encoder. Set ``--vae_weight 0`` to ablate the
ELBO branch and run as a pure score head (debug only).

Run:
    python -m src.baselines.train_factorvae_v2 --fold 1 --seed 42

Output: results/baselines_244/factorvae_v2/fold{F}_seed{S}.json (+ npz).
"""
from __future__ import annotations

import argparse
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from src.baselines.factorvae_adapter import FactorVAEAdapter, FactorVAEHyperparams
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
class FactorVAEV2Config(V2BaselineConfig):
    """Top-level config bundling the v2 protocol + FactorVAE-specific knobs.

    FactorVAE hyperparameter rationale:
      - num_factors K = 8 follows the AAAI 2022 paper's CSI 300 config.
      - hidden_size H = 64 matches the strongest commonly-cited unofficial
        PyTorch reimplementation (x7jeon8gi/FactorVAE) which performs best
        on a similarly-sized universe (~300 stocks); paper-stated H is 20
        but with our 22 raw features we found H=20 underfits.
      - num_portfolio M = 64 is the soft portfolio set size in
        FactorEncoder. Constraint: M must be <= the active ticker count
        on each day; with biotech 244-universe active counts of ~150-200
        per day, M = 64 is comfortably below.
      - vae_weight = 0.1 down-weights FactorVAE's ELBO relative to the
        cross-sectional MSE so the score head stays dominant for the
        ranking-style metrics (IC, NDCG@k); the ELBO acts as a
        regulariser on the factor head.
    """
    output_dir: str = "results/baselines_244/factorvae_v2"
    # FactorVAE architecture.
    d_model: int = 64               # GRU hidden + alpha/beta head width
    num_factors: int = 8            # K
    num_portfolio: int = 64         # M (must be <= min daily active count)
    num_layers: int = 1             # GRU depth
    # Loss mix.
    vae_weight: float = 0.1


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, choices=[1, 2, 3, 4, 5], required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true",
                   help="Limit to 2 epochs and abbreviated output.")
    p.add_argument("--max_epochs", type=int, default=None,
                   help="Override config.epochs (e.g. 1 for a smoke check).")
    p.add_argument("--vae_weight", type=float, default=None,
                   help="Override config.vae_weight (debug only).")
    p.add_argument("--panel_kind", type=str, default="biotech",
                   choices=["biotech", "lattice_native"])
    p.add_argument("--two_regime_val", action="store_true")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--panel_end", type=str, default=None,
                   help="Override panel end date (lattice_native default: 2025-12-31).")
    args = p.parse_args()

    cfg = FactorVAEV2Config(fold=args.fold, seed=args.seed)
    if args.smoke:
        cfg.epochs = 2
    if args.max_epochs is not None:
        cfg.epochs = int(args.max_epochs)
    if args.vae_weight is not None:
        cfg.vae_weight = float(args.vae_weight)
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
    print(f"[FactorVAE-v2] fold={cfg.fold} seed={cfg.seed} device={device}")

    x_raw, y, tickers, dates = build_panel(cfg)
    T, N, Fdim = x_raw.shape
    print(f"[FactorVAE-v2] panel: T={T} N={N} F={Fdim}")
    if N < 50:
        raise RuntimeError("Panel too small")

    mm = build_masks(cfg, dates, tickers)
    tradable = mm["tradable_mask"]
    loss_mask = mm["loss_mask"]
    hist20 = mm["history_valid_20d"]
    hist60 = mm["history_valid_60d"]

    train_idx, val_idx, test_idx = fold_split(cfg, dates)
    print(f"[FactorVAE-v2] fold {cfg.fold}: "
          f"train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    x = standardize_features(x_raw, tradable, train_idx)
    age_feat = build_age_features(tradable, hist20, hist60)
    age_days = age_feat[..., 0].astype(np.int64)

    x_t = torch.from_numpy(x).to(device)
    y_t = torch.from_numpy(y).to(device)

    W = cfg.temporal_window

    # Guard num_portfolio: must be <= min active count among training days.
    train_active = [int(tradable[t].sum()) for t in train_idx if t >= W - 1]
    if train_active:
        min_active = min(train_active)
        if cfg.num_portfolio > min_active:
            new_M = max(8, min_active - 1)
            print(f"[FactorVAE-v2] num_portfolio={cfg.num_portfolio} > min_active="
                  f"{min_active}; clamping to {new_M}")
            cfg.num_portfolio = new_M

    hp = FactorVAEHyperparams(
        d_feat=Fdim,
        hidden_size=cfg.d_model,
        num_factors=cfg.num_factors,
        num_portfolio=cfg.num_portfolio,
        num_layers=cfg.num_layers,
    )
    model = FactorVAEAdapter(hp).to(device)
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
            if m_np.sum() < max(8, cfg.num_portfolio + 1):
                # FactorVAE encoder requires >= num_portfolio active stocks.
                continue
            active_idx = np.flatnonzero(m_np)
            active_t = torch.from_numpy(active_idx).to(device)
            x_win = x_t[t - W + 1: t + 1, active_t, :].transpose(0, 1)  # (A, W, F)
            y_target_full = y_t[t]                                      # (N,)
            lmask_t = torch.from_numpy(loss_mask[t]).to(device)

            # FactorVAE's encoder needs returns aligned with active stocks; we
            # use the cross-sectionally z-scored target on active tickers only.
            y_active = y_target_full[active_t]
            if y_active.numel() >= 2:
                mu = y_active.mean()
                sd = y_active.std().clamp(min=1e-6)
                y_active_z = (y_active - mu) / sd
            else:
                y_active_z = y_active

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                if train_ and cfg.vae_weight > 0:
                    vae_loss, y_hat_active = model.training_loss(x_win, y_active_z)
                else:
                    y_hat_active = model(x_win)
                    vae_loss = torch.zeros((), device=device)

                y_full = torch.zeros(N, device=device, dtype=y_hat_active.dtype)
                y_full[active_t] = y_hat_active
                cs_loss = cs_mse_loss(y_full, y_target_full, lmask_t)
                if train_:
                    total_loss = cs_loss + cfg.vae_weight * vae_loss
                else:
                    total_loss = cs_loss

            if train_:
                optim.zero_grad()
                scaler.scale(total_loss).backward()
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optim)
                scaler.update()
                scheduler.step()
            losses.append(float(total_loss.item()))
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
        print(f"[FactorVAE-v2] epoch {epoch}: train_loss={train_loss:.4f} "
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
                print(f"[FactorVAE-v2] early stop epoch {epoch} "
                      f"best_val_ic={best_val_ic:+.4f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    _, test_yhat, test_mask = run_split(test_idx, train_=False)
    test_metrics = evaluate_predictions(test_yhat, y, test_mask, age_days)
    val_metrics_final = evaluate_predictions(val_yhat, y, val_mask, age_days)

    print(f"[FactorVAE-v2] TEST ic={test_metrics['ic']:+.4f} "
          f"rank_ic={test_metrics['rank_ic']:+.4f} "
          f"ndcg10={test_metrics['ndcg10']:.4f} "
          f"ndcg50={test_metrics['ndcg50']:.4f}")

    out_path = save_result(
        out_dir=Path(cfg.output_dir),
        fold=cfg.fold, seed=cfg.seed,
        model_name="FactorVAE (v2 protocol)",
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
    print(f"[FactorVAE-v2] wrote {out_path}")


if __name__ == "__main__":
    main()
