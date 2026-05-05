"""Training loop for the v2 STAR baseline.

Reuses the v2 panel loader, fold definitions, and evaluation utilities.
Loss is `cs_robust_loss` (Huber + inverse-volatility per-ticker
weighting), the v1 "iter 10" recipe.

Usage:
    python -m src.baselines.star_v2.train --config configs/baseline_star.yaml \\
        --fold 1 --seed 42
"""
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from src.baselines.star_v2.model import STARBaseline, STARBaselineConfig
from src.mtgn.model.utils.patch_construction import (
    build_patches,
    precompute_top_neighbors,
)
from src.mtgn.training.graph_builder import GraphConfig, build_correlation_edges
from src.mtgn.training.losses import cs_robust_loss
from src.mtgn.training.panel_enriched import (
    EnrichedPanelConfig,
    build_enriched_panel,
    panel_to_tensors,
)
from src.v2.model.star_backbone import STARBackboneConfig
from src.v2.training.folds import fold_indices


@dataclass
class TrainConfig:
    """Top-level hyperparameters."""

    fold: int = 1
    seed: int = 42
    panel_start: str = "2015-01-09"
    panel_end: str = "2022-12-31"
    horizon_days: int = 5
    universe_csv: str = "data/raw/biotech_universe_v1.csv"
    epochs: int = 10
    learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-5
    warmup_steps: int = 500
    grad_clip: float = 1.0
    early_stop_patience: int = 3
    huber_delta: float = 1.0
    use_vol_weight: bool = True
    vol_feature_idx: int = 6
    output_dir: str = "results/baselines_244/star"


def set_seeds(seed: int) -> None:
    """Set seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def per_day_ic(
    y_hat: np.ndarray, y: np.ndarray, mask: np.ndarray, rank: bool = False
) -> tuple[float, np.ndarray]:
    """Daily-then-mean Information Coefficient (IC)."""
    t_total = y_hat.shape[0]
    ics = np.full(t_total, np.nan, dtype=np.float64)
    for t in range(t_total):
        m = mask[t]
        if m.sum() < 5:
            continue
        a = y_hat[t, m]
        b = y[t, m]
        if rank:
            a = pd.Series(a).rank().to_numpy()
            b = pd.Series(b).rank().to_numpy()
        sa, sb = a.std(), b.std()
        if sa < 1e-9 or sb < 1e-9:
            continue
        ics[t] = float(np.corrcoef(a, b)[0, 1])
    return (np.nanmean(ics) if not np.all(np.isnan(ics)) else 0.0, ics)


def ndcg_at_k(
    y_hat: np.ndarray, y: np.ndarray, mask: np.ndarray, k: int
) -> float:
    """Daily-mean Normalized Discounted Cumulative Gain at k."""
    out = []
    for t in range(y_hat.shape[0]):
        m = mask[t]
        if m.sum() < k + 1:
            continue
        scores = y_hat[t, m]
        rels = y[t, m]
        rels_pos = rels - rels.min() + 1e-9
        order = np.argsort(-scores)[:k]
        gains = rels_pos[order]
        discounts = 1.0 / np.log2(np.arange(2, k + 2))
        dcg = float((gains * discounts).sum())
        ideal = np.sort(rels_pos)[::-1][:k]
        idcg = float((ideal * discounts).sum())
        if idcg < 1e-9:
            continue
        out.append(dcg / idcg)
    return float(np.mean(out)) if out else 0.0


def standardize_features(
    x: np.ndarray, mask: np.ndarray, train_idx: np.ndarray
) -> np.ndarray:
    """Per-feature train-fold z-score standardization."""
    flat_train_mask = mask[train_idx]
    x_train = x[train_idx]
    out = np.zeros_like(x)
    for f in range(x.shape[2]):
        vals = x_train[..., f][flat_train_mask]
        if vals.size < 2:
            mu, sd = 0.0, 1.0
        else:
            mu = float(np.mean(vals))
            sd = float(np.std(vals))
            if sd < 1e-6:
                sd = 1.0
        out[..., f] = (x[..., f] - mu) / sd
    out = out * mask[..., None]
    return out


def warmup_cosine_lr(step: int, warmup: int, total: int) -> float:
    """Linear warmup then cosine decay to 0.1x."""
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))


def main(cfg_path: str, fold: int, seed: int, smoke: bool = False) -> None:
    """Train STAR baseline on one (fold, seed) pair."""
    with open(cfg_path) as f:
        raw = yaml.safe_load(f)

    train_cfg = TrainConfig(**{**raw.get("train", {}), "fold": fold, "seed": seed})
    backbone_cfg = STARBackboneConfig(**raw.get("backbone", {}))
    model_cfg = STARBaselineConfig(
        backbone=backbone_cfg,
        head_hidden_dim=raw.get("model", {}).get("head_hidden_dim", 64),
        head_dropout=raw.get("model", {}).get("head_dropout", 0.1),
    )
    if smoke:
        train_cfg.epochs = 2

    set_seeds(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[STAR-baseline] fold={fold} seed={seed} device={device}")

    panel_cfg = EnrichedPanelConfig(
        start_date=pd.Timestamp(train_cfg.panel_start),
        end_date=pd.Timestamp(train_cfg.panel_end),
        horizon_days=train_cfg.horizon_days,
        universe_csv=Path(train_cfg.universe_csv),
    )
    panel, tickers, dates = build_enriched_panel(panel_cfg)
    tens = panel_to_tensors(panel, tickers, dates)
    x_raw = tens["x"]
    y = tens["y"]
    mask = tens["mask"]
    print(f"[STAR-baseline] panel: T={x_raw.shape[0]} N={x_raw.shape[1]} F={x_raw.shape[2]}")
    if x_raw.shape[1] < 50:
        raise RuntimeError(f"Panel has only {x_raw.shape[1]} active tickers; aborting.")

    train_idx, val_idx, test_idx = fold_indices(fold, dates)
    print(f"[STAR-baseline] fold {fold}: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    x = standardize_features(x_raw, mask, train_idx).astype(np.float32)

    graph_cfg = GraphConfig(correlation_window_days=60, correlation_threshold=0.3)
    edge_index, edge_weight = build_correlation_edges(x_raw[train_idx], graph_cfg)
    top_neighbors = precompute_top_neighbors(
        edge_index, edge_weight, num_nodes=x.shape[1], N=backbone_cfg.num_neighbors
    )
    top_neighbors_t = torch.from_numpy(top_neighbors).to(device)
    print(f"[STAR-baseline] mechanistic graph: |E|={edge_index.shape[1]}")

    model = STARBaseline(model_cfg).to(device)
    optim = AdamW(model.parameters(), lr=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay)
    total_steps = train_cfg.epochs * max(1, len(train_idx))
    scheduler = LambdaLR(optim, lr_lambda=lambda s: warmup_cosine_lr(s, train_cfg.warmup_steps, total_steps))
    use_amp = device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    w = backbone_cfg.temporal_window
    best_val_ic = -1e9
    best_state = None
    patience = 0
    history: list[dict] = []

    def forward_one_day(t_idx: int) -> dict:
        if t_idx < w:
            return {}
        x_window = torch.from_numpy(x[t_idx - w + 1 : t_idx + 1]).to(device)
        mask_window = torch.from_numpy(mask[t_idx - w + 1 : t_idx + 1]).to(device)
        active_mask_t = torch.from_numpy(mask[t_idx]).to(device)
        if active_mask_t.sum() < 5:
            return {}
        active_idx = active_mask_t.nonzero(as_tuple=False).squeeze(-1)
        patches, patch_mask = build_patches(
            x_window=x_window, mask_window=mask_window,
            top_neighbors=top_neighbors_t, active_idx=active_idx,
        )
        with autocast(enabled=use_amp, dtype=torch.float16):
            out = model.forward_day(patches, patch_mask, active_mask_t)
        out["y_hat"] = out["y_hat"].float()
        out["active_mask"] = active_mask_t
        out["t_idx"] = t_idx
        return out

    @torch.no_grad()
    def evaluate(idx: np.ndarray) -> dict:
        model.eval()
        T = x.shape[0]
        N = x.shape[1]
        y_hat_all = np.zeros((T, N), dtype=np.float32)
        eval_mask = np.zeros((T, N), dtype=bool)
        for t_idx in idx:
            t_idx = int(t_idx)
            out = forward_one_day(t_idx)
            if not out:
                continue
            y_hat_all[t_idx] = out["y_hat"].detach().cpu().numpy()
            eval_mask[t_idx] = mask[t_idx]
        ic, ic_per_day = per_day_ic(y_hat_all, y, eval_mask, rank=False)
        rank_ic, _ = per_day_ic(y_hat_all, y, eval_mask, rank=True)
        ndcg10 = ndcg_at_k(y_hat_all, y, eval_mask, 10)
        ndcg50 = ndcg_at_k(y_hat_all, y, eval_mask, 50)
        return {
            "ic": ic, "rank_ic": rank_ic, "ndcg10": ndcg10, "ndcg50": ndcg50,
            "n_days_scored": int(np.sum(~np.isnan(ic_per_day))),
            "y_hat_all": y_hat_all, "eval_mask": eval_mask,
        }

    step = 0
    smoke_step_cap = 80
    for epoch in range(train_cfg.epochs):
        model.train()
        np.random.seed(seed + epoch)
        perm = np.random.permutation(train_idx)
        epoch_losses: list[float] = []
        for t_idx in perm:
            t_idx = int(t_idx)
            if t_idx < w:
                continue
            out = forward_one_day(t_idx)
            if not out:
                continue
            y_true_t = torch.from_numpy(y[t_idx]).to(device)
            mask_t = out["active_mask"]
            vol_t = None
            if train_cfg.use_vol_weight:
                vol_t = torch.from_numpy(np.abs(x_raw[t_idx, :, train_cfg.vol_feature_idx])).float().to(device)
            loss = cs_robust_loss(
                out["y_hat"], y_true_t, mask_t,
                delta=train_cfg.huber_delta, vol=vol_t,
            )
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            scaler.step(optim)
            scaler.update()
            scheduler.step()
            epoch_losses.append(float(loss.item()))
            step += 1
            if smoke and step >= smoke_step_cap:
                break

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        val_metrics = evaluate(val_idx)
        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "val_ic": val_metrics["ic"], "val_rank_ic": val_metrics["rank_ic"],
        })
        print(f"[STAR-baseline] epoch {epoch}: train_loss={train_loss:.4f} "
              f"val_ic={val_metrics['ic']:.4f} val_rank_ic={val_metrics['rank_ic']:.4f}")

        if val_metrics["ic"] > best_val_ic + 1e-5:
            best_val_ic = val_metrics["ic"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= train_cfg.early_stop_patience:
                print(f"[STAR-baseline] early stop at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate(test_idx)
    val_metrics_final = evaluate(val_idx)

    out_dir = Path(train_cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"fold{fold}_seed{seed}.json"
    pred_path = out_dir / f"fold{fold}_seed{seed}_predictions.npz"

    np.savez_compressed(
        pred_path,
        y_hat=test_metrics["y_hat_all"], y_true=y, mask=test_metrics["eval_mask"],
        test_idx=np.asarray(test_idx, dtype=np.int64),
        tickers=np.asarray(tickers, dtype=str),
        dates=np.asarray([str(d) for d in dates], dtype=str),
    )
    if best_state is not None:
        torch.save(best_state, out_dir / f"fold{fold}_seed{seed}_ckpt.pt")

    payload = {
        "fold": fold, "seed": seed, "model": "STAR",
        "panel_start": train_cfg.panel_start, "panel_end": train_cfg.panel_end,
        "n_tickers": int(x.shape[1]), "n_dates": int(x.shape[0]),
        "n_train": int(len(train_idx)), "n_val": int(len(val_idx)), "n_test": int(len(test_idx)),
        "ic": test_metrics["ic"], "rank_ic": test_metrics["rank_ic"],
        "ndcg10": test_metrics["ndcg10"], "ndcg50": test_metrics["ndcg50"],
        "val_ic": val_metrics_final["ic"], "val_rank_ic": val_metrics_final["rank_ic"],
        "best_val_ic": best_val_ic, "history": history,
        "config": {"train": asdict(train_cfg), "backbone": asdict(backbone_cfg),
                   "model": {"head_hidden_dim": model_cfg.head_hidden_dim,
                             "head_dropout": model_cfg.head_dropout}},
    }
    # Drop the heavy arrays before json dump
    test_metrics.pop("y_hat_all", None); test_metrics.pop("eval_mask", None)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"[STAR-baseline] wrote {out_path}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/baseline_star.yaml")
    p.add_argument("--fold", type=int, choices=[1, 2, 3], required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.config, args.fold, args.seed, smoke=args.smoke)
