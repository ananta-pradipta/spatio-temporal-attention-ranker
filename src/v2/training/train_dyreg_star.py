"""Training loop for DyReg-STAR (Stage 1).

Single-fold, single-seed driver. The script:
    1. Builds the 22-feature panel via the v1 enriched-panel loader.
    2. Builds the static mechanistic graph from training-window
       correlations only (used optionally as a mixture component).
    3. Precomputes [T, N, K] dynamic neighbor lists from rolling 60-day
       correlations using only data available up to each day t.
    4. Trains DyReg-STAR with cross-sectional Mean Squared Error (MSE)
       on z-scored 5-day forward log returns.
    5. Evaluates Information Coefficient (IC), Rank-IC, and Normalized
       Discounted Cumulative Gain (NDCG) at 10 and 50 on validation
       and test.
    6. Writes a JSON results file plus a graph-diagnostics CSV.

Usage:
    python -m src.v2.training.train_dyreg_star --config configs/dyreg_star.yaml \\
        --fold 1 --seed 42

Reproducibility: all seeds, hyperparameters, and run-time choices are
echoed to the JSON output.
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

from src.mtgn.training.graph_builder import GraphConfig, build_correlation_edges
from src.mtgn.training.panel_enriched import (
    EnrichedPanelConfig,
    build_enriched_panel,
    panel_to_tensors,
)
from src.mtgn.model.utils.patch_construction import build_patches
from src.v2.graph.dynamic_edges import (
    DynamicGraphConfig,
    build_dynamic_neighbors,
    static_score_from_correlation_edges,
)
from src.v2.model.dyreg_star import DyRegSTAR, DyRegSTARConfig
from src.v2.model.star_backbone import STARBackboneConfig
from src.v2.training.folds import fold_indices


@dataclass
class TrainConfig:
    """Top-level training hyperparameters."""

    fold: int = 1
    seed: int = 42
    panel_start: str = "2015-01-09"
    panel_end: str = "2022-12-31"
    horizon_days: int = 5
    universe_csv: str = "data/raw/biotech_universe_v1.csv"
    epochs: int = 15
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_steps: int = 500
    grad_clip: float = 1.0
    early_stop_patience: int = 5
    output_dir: str = "results/dyreg_star"


def set_seeds(seed: int) -> None:
    """Set seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cs_mse_loss(
    y_hat: torch.Tensor, y_true: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Cross-sectional MSE on per-day z-scored true returns."""
    m = mask.bool()
    yh = y_hat[m]
    yt = y_true[m]
    if yt.numel() < 2:
        return torch.zeros((), device=y_hat.device, dtype=y_hat.dtype)
    mu = yt.mean()
    sd = yt.std().clamp(min=1e-6)
    yt_zs = (yt - mu) / sd
    return ((yh - yt_zs) ** 2).mean()


def per_day_ic(
    y_hat: np.ndarray, y_true: np.ndarray, mask: np.ndarray, rank: bool = False
) -> tuple[float, np.ndarray]:
    """Daily-then-mean IC. Returns (mean_ic, per_day_ic_array)."""
    t_total = y_hat.shape[0]
    ics = np.full(t_total, np.nan, dtype=np.float64)
    for t in range(t_total):
        m = mask[t]
        if m.sum() < 5:
            continue
        a = y_hat[t, m]
        b = y_true[t, m]
        if rank:
            a = pd.Series(a).rank().to_numpy()
            b = pd.Series(b).rank().to_numpy()
        sa = a.std()
        sb = b.std()
        if sa < 1e-9 or sb < 1e-9:
            continue
        ics[t] = float(np.corrcoef(a, b)[0, 1])
    valid = ~np.isnan(ics)
    if valid.sum() == 0:
        return 0.0, ics
    return float(np.nanmean(ics[valid])), ics


def ndcg_at_k(
    y_hat: np.ndarray, y_true: np.ndarray, mask: np.ndarray, k: int
) -> float:
    """Daily-mean NDCG at k computed on the active cross-section."""
    out = []
    for t in range(y_hat.shape[0]):
        m = mask[t]
        if m.sum() < k + 1:
            continue
        scores = y_hat[t, m]
        rels = y_true[t, m]
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
    """Per-feature z-score standardization using train-fold statistics."""
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
    """Train DyReg-STAR on one (fold, seed) pair and write a JSON report."""
    with open(cfg_path) as f:
        raw = yaml.safe_load(f)

    train_cfg = TrainConfig(**{**raw.get("train", {}), "fold": fold, "seed": seed})
    backbone_cfg = STARBackboneConfig(**raw.get("backbone", {}))
    graph_cfg = DynamicGraphConfig(**raw.get("dynamic_graph", {}))
    model_cfg = DyRegSTARConfig(
        backbone=backbone_cfg,
        head_hidden_dim=raw.get("model", {}).get("head_hidden_dim", 64),
        head_dropout=raw.get("model", {}).get("head_dropout", 0.1),
    )
    if smoke:
        train_cfg.epochs = 2

    set_seeds(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DyReg-STAR] fold={fold} seed={seed} device={device}")

    panel_cfg = EnrichedPanelConfig(
        start_date=pd.Timestamp(train_cfg.panel_start),
        end_date=pd.Timestamp(train_cfg.panel_end),
        horizon_days=train_cfg.horizon_days,
        universe_csv=Path(train_cfg.universe_csv),
    )
    panel, tickers, dates = build_enriched_panel(panel_cfg)
    tensors = panel_to_tensors(panel, tickers, dates)
    x_raw = tensors["x"]
    y = tensors["y"]
    mask = tensors["mask"]
    print(f"[DyReg-STAR] panel: T={x_raw.shape[0]} N={x_raw.shape[1]} F={x_raw.shape[2]}")
    if x_raw.shape[1] < 50:
        raise RuntimeError(f"Panel has only {x_raw.shape[1]} active tickers; aborting.")

    train_idx, val_idx, test_idx = fold_indices(fold, dates)
    print(f"[DyReg-STAR] fold {fold}: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    x = standardize_features(x_raw, mask, train_idx).astype(np.float32)

    # Optional static graph for the mixture component.
    static_score = None
    if graph_cfg.static_mix_weight > 0.0:
        static_cfg = GraphConfig(correlation_window_days=60, correlation_threshold=0.3)
        edge_index, edge_weight = build_correlation_edges(x_raw[train_idx], static_cfg)
        static_score = static_score_from_correlation_edges(
            edge_index, edge_weight, num_nodes=x.shape[1]
        )
        print(f"[DyReg-STAR] static graph: |E|={edge_index.shape[1]} mix_w={graph_cfg.static_mix_weight}")

    # Precompute the [T, N, K] dynamic neighbor matrix.
    print(f"[DyReg-STAR] building dynamic neighbors: window={graph_cfg.window_days} K={graph_cfg.top_k}")
    dyn_neighbors = build_dynamic_neighbors(
        returns=x_raw[..., 0],
        mask=mask,
        cfg=graph_cfg,
        static_score=static_score,
    )
    coverage = (dyn_neighbors >= 0).any(axis=-1).sum() / float(mask.sum())
    print(f"[DyReg-STAR] dynamic-neighbor coverage on active cells: {coverage:.3f}")
    dyn_neighbors_t = torch.from_numpy(dyn_neighbors).to(device)

    model = DyRegSTAR(model_cfg).to(device)
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
        active_idx = torch.nonzero(active_mask_t, as_tuple=False).squeeze(-1)
        top_neighbors_day = dyn_neighbors_t[t_idx]  # [N, K] day-specific
        patches, patch_mask = build_patches(
            x_window=x_window, mask_window=mask_window,
            top_neighbors=top_neighbors_day, active_idx=active_idx,
        )
        with autocast(enabled=use_amp, dtype=torch.float16):
            out = model.forward_day(patches, patch_mask, active_mask_t)
        out["y_hat"] = out["y_hat"].float()
        out["active_mask"] = active_mask_t
        out["t_idx"] = t_idx
        return out

    def evaluate(idx: np.ndarray, label: str) -> dict:
        model.eval()
        T = x.shape[0]
        N = x.shape[1]
        y_hat_all = np.zeros((T, N), dtype=np.float32)
        eval_mask = np.zeros((T, N), dtype=bool)
        with torch.no_grad():
            for t_idx in idx:
                out = forward_one_day(int(t_idx))
                if not out:
                    continue
                y_hat_all[t_idx] = out["y_hat"].detach().cpu().numpy()
                eval_mask[t_idx] = mask[t_idx]
        ic, ic_per_day = per_day_ic(y_hat_all, y, eval_mask, rank=False)
        rank_ic, _ = per_day_ic(y_hat_all, y, eval_mask, rank=True)
        ndcg10 = ndcg_at_k(y_hat_all, y, eval_mask, 10)
        ndcg50 = ndcg_at_k(y_hat_all, y, eval_mask, 50)
        return {
            "label": label, "ic": ic, "rank_ic": rank_ic,
            "ndcg10": ndcg10, "ndcg50": ndcg50,
            "n_days_scored": int(np.sum(~np.isnan(ic_per_day))),
        }

    step = 0
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
            loss = cs_mse_loss(out["y_hat"], y_true_t, mask_t)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            scaler.step(optim)
            scaler.update()
            scheduler.step()
            epoch_losses.append(float(loss.item()))
            step += 1
            if smoke and step >= 80:
                break

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        val_metrics = evaluate(val_idx, "val")
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_ic": val_metrics["ic"],
            "val_rank_ic": val_metrics["rank_ic"],
        })
        print(f"[DyReg-STAR] epoch {epoch}: train_loss={train_loss:.4f} "
              f"val_ic={val_metrics['ic']:.4f} val_rank_ic={val_metrics['rank_ic']:.4f}")
        if val_metrics["ic"] > best_val_ic + 1e-5:
            best_val_ic = val_metrics["ic"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= train_cfg.early_stop_patience:
                print(f"[DyReg-STAR] early stop at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate(test_idx, "test")
    val_metrics_final = evaluate(val_idx, "val")

    out_dir = Path(train_cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save predictions and checkpoint for downstream diagnostics.
    model.eval()
    pred_path = out_dir / f"fold{fold}_seed{seed}_predictions.npz"
    y_hat_test = np.zeros((x.shape[0], x.shape[1]), dtype=np.float32)
    pred_mask = np.zeros((x.shape[0], x.shape[1]), dtype=bool)
    with torch.no_grad():
        for t_idx in test_idx:
            out_d = forward_one_day(int(t_idx))
            if not out_d:
                continue
            y_hat_test[t_idx] = out_d["y_hat"].detach().cpu().numpy()
            pred_mask[t_idx] = mask[t_idx]
    np.savez_compressed(
        pred_path,
        y_hat=y_hat_test, y_true=y, mask=pred_mask,
        test_idx=np.asarray(test_idx, dtype=np.int64),
        tickers=np.asarray(tickers, dtype=str),
        dates=np.asarray([str(d) for d in dates], dtype=str),
    )
    print(f"[DyReg-STAR] wrote {pred_path}")
    if best_state is not None:
        torch.save(best_state, out_dir / f"fold{fold}_seed{seed}_ckpt.pt")
        print(f"[DyReg-STAR] wrote {out_dir / f'fold{fold}_seed{seed}_ckpt.pt'}")

    out_path = out_dir / f"fold{fold}_seed{seed}.json"

    # Compute graph diagnostics: average turnover across consecutive test days,
    # average overlap with the static graph, and graph density.
    def neighbor_set(t: int, i: int) -> set[int]:
        return {int(n) for n in dyn_neighbors[t, i] if n >= 0}

    if static_score is not None:
        # Build static top-K from the static score for overlap stats.
        static_top = np.full((x.shape[1], graph_cfg.top_k), -1, dtype=np.int64)
        s_static = static_score.copy()
        np.fill_diagonal(s_static, -np.inf)
        for i in range(x.shape[1]):
            order = np.argsort(-s_static[i])[: graph_cfg.top_k]
            static_top[i] = order
        avg_overlap = []
        for t in test_idx:
            for i in range(x.shape[1]):
                if not mask[t, i]:
                    continue
                dyn_set = neighbor_set(t, i)
                static_set = {int(n) for n in static_top[i]}
                if dyn_set:
                    avg_overlap.append(len(dyn_set & static_set) / float(graph_cfg.top_k))
        overlap_with_static = float(np.mean(avg_overlap)) if avg_overlap else 0.0
    else:
        overlap_with_static = float("nan")

    turnover = []
    for t in test_idx[1:]:
        for i in range(x.shape[1]):
            if not mask[t, i] or not mask[t - 1, i]:
                continue
            a = neighbor_set(t, i)
            b = neighbor_set(int(t) - 1, i)
            if a or b:
                turnover.append(len(a.symmetric_difference(b)) / float(2 * graph_cfg.top_k))
    avg_turnover = float(np.mean(turnover)) if turnover else 0.0

    payload = {
        "fold": fold,
        "seed": seed,
        "model": "DyReg-STAR",
        "panel_start": train_cfg.panel_start,
        "panel_end": train_cfg.panel_end,
        "n_tickers": int(x.shape[1]),
        "n_dates": int(x.shape[0]),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "ic": test_metrics["ic"],
        "rank_ic": test_metrics["rank_ic"],
        "ndcg10": test_metrics["ndcg10"],
        "ndcg50": test_metrics["ndcg50"],
        "val_ic": val_metrics_final["ic"],
        "val_rank_ic": val_metrics_final["rank_ic"],
        "best_val_ic": best_val_ic,
        "graph_avg_turnover_test": avg_turnover,
        "graph_avg_overlap_with_static_test": overlap_with_static,
        "graph_neighbor_coverage_active_cells": float(coverage),
        "history": history,
        "config": {
            "train": asdict(train_cfg),
            "backbone": asdict(backbone_cfg),
            "dynamic_graph": asdict(graph_cfg),
            "model": {
                "head_hidden_dim": model_cfg.head_hidden_dim,
                "head_dropout": model_cfg.head_dropout,
            },
        },
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"[DyReg-STAR] wrote {out_path}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/dyreg_star.yaml")
    p.add_argument("--fold", type=int, choices=[1, 2, 3], required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.config, args.fold, args.seed, smoke=args.smoke)
