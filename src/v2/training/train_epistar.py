"""Training loop for epiSTAR.

Single-fold, single-seed driver. The script:
    1. Builds the 22-feature panel via the v1 enriched-panel loader.
    2. Builds episode keys (8-dim risk + 6-dim cross-sectional diagnostics).
    3. Builds episode values (cached STAR-day summary statistics; for the
       Stage 1 implementation we use the raw episode key plus a small set
       of returns-based summaries, not a frozen STAR pass).
    4. Builds the static mechanistic graph from training-window correlation
       only (no test/val data leaks into the graph).
    5. Trains epiSTAR with cross-sectional Mean Squared Error (MSE) on
       z-scored 5-day forward log returns.
    6. Evaluates Information Coefficient (IC) and Rank-IC on validation
       and test, with Normalized Discounted Cumulative Gain (NDCG) at 10
       and 50.
    7. Writes a JSON results file plus a retrieval-diagnostics CSV.

Usage:
    python -m src.v2.training.train_epistar --config configs/epistar.yaml \\
        --fold 1 --seed 42

Reproducibility: all seeds, hyperparameters, and run-time choices are
echoed to the JSON output.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from src.mtgn.training.graph_builder import GraphConfig, build_correlation_edges
from src.mtgn.training.panel_enriched import (
    EnrichedPanelConfig,
    build_enriched_panel,
    panel_to_tensors,
)
from src.mtgn.model.utils.patch_construction import (
    build_patches,
    precompute_top_neighbors,
)
from src.v2.data.episode_keys import (
    EpisodeKeyConfig,
    EPISODE_KEY_COLS,
    build_episode_keys,
)
from src.v2.model.episode_memory import EpisodeMemoryConfig
from src.v2.model.epistar import EpiSTAR, EpiSTARConfig
from src.v2.model.star_backbone import STARBackboneConfig
from src.v2.training.folds import FOLDS, fold_indices


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
    label_smoothing: float = 0.0
    output_dir: str = "results/epistar"
    # If True, replace the static mechanistic graph with the per-day
    # rolling-correlation neighbor list from DyReg-STAR. The combined
    # model (STAR + dynamic graph + episodic retrieval) is "epiSTAR-full".
    use_dynamic_graph: bool = False
    correlation_window: int = 60


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
    """Cross-sectional Mean Squared Error on z-scored true returns.

    Standardization is per-day across active tickers only; this is the
    same loss used by all v1 baselines.
    """
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
        # Shift relevance to positive (rank-style) by ordering on returns.
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


def build_episode_values(
    keys: np.ndarray, panel_x: np.ndarray, panel_mask: np.ndarray
) -> np.ndarray:
    """Build per-day episode values from raw inputs.

    Stage-1 episode values combine three components:
        (a) the raw episode key (shape K),
        (b) cross-sectional summary of the day's STAR input features
            (mean and std of selected columns across active tickers),
        (c) the day's active count normalized.

    This is intentionally lightweight; Stage 2 can replace this with a
    cached pass through a frozen STAR encoder.
    """
    t_total = keys.shape[0]
    feature_idx = [0, 1, 5, 6]  # log_return, log_return_5d, rv_20d, rv_60d
    n_summary = 2 * len(feature_idx) + 1
    out = np.zeros((t_total, n_summary), dtype=np.float32)
    for t in range(t_total):
        m = panel_mask[t]
        if m.sum() < 5:
            continue
        for j, fi in enumerate(feature_idx):
            x = panel_x[t, m, fi]
            out[t, 2 * j] = float(np.mean(x))
            out[t, 2 * j + 1] = float(np.std(x))
        out[t, -1] = float(m.sum()) / 250.0
    return np.concatenate([keys, out], axis=1)


def standardize_features(
    x: np.ndarray, mask: np.ndarray, train_idx: np.ndarray
) -> np.ndarray:
    """Per-feature z-score standardization using train-fold statistics.

    Inactive cells contribute NaN-equivalent values that are excluded from
    the mean/std calculation. The output replaces inactive cells with zero.
    """
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
    """Linear warmup for `warmup` steps followed by cosine decay to 0.1x."""
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))


def main(cfg_path: str, fold: int, seed: int, smoke: bool = False) -> None:
    """Train epiSTAR on one (fold, seed) pair and write a JSON report."""
    with open(cfg_path) as f:
        raw = yaml.safe_load(f)

    train_cfg = TrainConfig(**{**raw.get("train", {}), "fold": fold, "seed": seed})
    backbone_cfg = STARBackboneConfig(**raw.get("backbone", {}))
    memory_cfg = EpisodeMemoryConfig(**raw.get("memory", {}))
    epistar_cfg = EpiSTARConfig(
        backbone=backbone_cfg,
        memory=memory_cfg,
        episode_value_dim=raw.get("model", {}).get("episode_value_dim", 32),
        cross_attn_heads=raw.get("model", {}).get("cross_attn_heads", 4),
        gate_hidden_dim=raw.get("model", {}).get("gate_hidden_dim", 64),
        head_hidden_dim=raw.get("model", {}).get("head_hidden_dim", 64),
        head_dropout=raw.get("model", {}).get("head_dropout", 0.1),
        disable_gate=raw.get("ablation", {}).get("disable_gate", False),
        disable_retrieval=raw.get("ablation", {}).get("disable_retrieval", False),
    )
    if smoke:
        train_cfg.epochs = 2

    set_seeds(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[epiSTAR] fold={fold} seed={seed} device={device}")
    print(f"[epiSTAR] panel: {train_cfg.panel_start} to {train_cfg.panel_end}")

    panel_cfg = EnrichedPanelConfig(
        start_date=pd.Timestamp(train_cfg.panel_start),
        end_date=pd.Timestamp(train_cfg.panel_end),
        horizon_days=train_cfg.horizon_days,
        universe_csv=Path(train_cfg.universe_csv),
    )
    panel, tickers, dates = build_enriched_panel(panel_cfg)
    tensors = panel_to_tensors(panel, tickers, dates)
    x_raw = tensors["x"]  # [T, N, F]
    y = tensors["y"]  # [T, N]
    mask = tensors["mask"]  # [T, N]
    print(f"[epiSTAR] panel shape: T={x_raw.shape[0]}, N={x_raw.shape[1]}, F={x_raw.shape[2]}")
    if x_raw.shape[1] < 50:
        raise RuntimeError(f"Panel has only {x_raw.shape[1]} active tickers; aborting.")

    # Fold indices.
    train_idx, val_idx, test_idx = fold_indices(fold, dates)
    print(f"[epiSTAR] fold {fold}: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    # Standardize features using train-fold stats only.
    x = standardize_features(x_raw, mask, train_idx).astype(np.float32)

    # Graph construction. Two modes:
    #   (a) static (default): single mechanistic graph from train-window
    #       correlations, used unchanged across all days.
    #   (b) dynamic (use_dynamic_graph=True): per-day rolling-correlation
    #       neighbor list, like DyReg-STAR. The combined model
    #       (STAR + dynamic graph + episodic retrieval) is "epiSTAR-full".
    if train_cfg.use_dynamic_graph:
        from src.v2.graph.dynamic_edges import (
            DynamicGraphConfig,
            build_dynamic_neighbors,
        )
        dyn_cfg = DynamicGraphConfig(
            window_days=train_cfg.correlation_window,
            top_k=backbone_cfg.num_neighbors,
        )
        dyn_neighbors = build_dynamic_neighbors(
            returns=x_raw[..., 0], mask=mask, cfg=dyn_cfg, static_score=None,
        )
        dyn_neighbors_t = torch.from_numpy(dyn_neighbors).to(device)
        coverage = (dyn_neighbors >= 0).any(axis=-1).sum() / float(mask.sum())
        print(f"[epiSTAR] dynamic graph: window={dyn_cfg.window_days}d "
              f"top_K={dyn_cfg.top_k}, coverage on active cells: {coverage:.3f}")
        top_neighbors_t = None  # not used in dynamic mode
    else:
        graph_cfg = GraphConfig(correlation_window_days=60, correlation_threshold=0.3)
        edge_index, edge_weight = build_correlation_edges(x_raw[train_idx], graph_cfg)
        top_neighbors = precompute_top_neighbors(
            edge_index, edge_weight, num_nodes=x.shape[1], N=backbone_cfg.num_neighbors
        )
        top_neighbors_t = torch.from_numpy(top_neighbors).to(device)
        print(f"[epiSTAR] static mechanistic graph: |E|={edge_index.shape[1]}, top_N={backbone_cfg.num_neighbors}")
        dyn_neighbors_t = None

    # Build episode keys + values. Keys are raw (un-standardized); the
    # memory bank standardizes them using training-day stats internally.
    keys, key_cols = build_episode_keys(
        dates=dates,
        log_returns=x_raw[..., 0],
        mask=mask,
        cfg=EpisodeKeyConfig(),
    )
    values = build_episode_values(keys, x_raw, mask)
    print(f"[epiSTAR] episode keys: K={keys.shape[1]}, values: V={values.shape[1]}")

    epistar_cfg.episode_value_dim = values.shape[1]
    model = EpiSTAR(epistar_cfg, episode_key_dim=keys.shape[1]).to(device)
    model.memory.populate(
        keys=keys,
        values=values,
        day_indices=np.arange(len(dates)),
        train_day_indices=train_idx,
    )
    model.memory.to(device)

    allowed_train = torch.from_numpy(train_idx).long().to(device)

    optim = AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
    )
    total_steps = train_cfg.epochs * max(1, len(train_idx))
    scheduler = LambdaLR(
        optim, lr_lambda=lambda s: warmup_cosine_lr(s, train_cfg.warmup_steps, total_steps)
    )
    use_amp = device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    w = backbone_cfg.temporal_window
    best_val_ic = -1e9
    best_state = None
    patience = 0
    history: list[dict] = []

    def forward_one_day(t_idx: int) -> dict:
        """Run a forward pass at panel day index t_idx."""
        if t_idx < w:
            return {}
        x_window = torch.from_numpy(x[t_idx - w + 1 : t_idx + 1]).to(device)  # [W, N, F]
        mask_window = torch.from_numpy(mask[t_idx - w + 1 : t_idx + 1]).to(device)
        active_mask_t = torch.from_numpy(mask[t_idx]).to(device)
        if active_mask_t.sum() < 5:
            return {}
        active_idx = torch.nonzero(active_mask_t, as_tuple=False).squeeze(-1)

        # Pick the right neighbor list for this day.
        if dyn_neighbors_t is not None:
            top_neighbors_day = dyn_neighbors_t[t_idx]
        else:
            top_neighbors_day = top_neighbors_t
        patches, patch_mask = build_patches(
            x_window=x_window, mask_window=mask_window,
            top_neighbors=top_neighbors_day, active_idx=active_idx
        )

        query_raw_key = torch.from_numpy(keys[t_idx]).float().to(device)

        # Two regime scalars for the gate: VIX z-score and avg pairwise corr.
        regime_scalars = model.memory.standardize_query(query_raw_key)[[0, 9]].clone()
        if torch.isnan(regime_scalars).any():
            regime_scalars = torch.zeros(2, device=device)

        with autocast(enabled=use_amp, dtype=torch.float16):
            out = model.forward_day(
                patches=patches,
                patch_mask=patch_mask,
                active_mask=active_mask_t,
                query_raw_key=query_raw_key,
                query_day_idx=t_idx,
                allowed_day_indices=allowed_train,
                gate_regime_scalars=regime_scalars,
            )
        # Cast critical outputs back to float32 for the loss computation.
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
        retrieval_log: list[dict] = []
        with torch.no_grad():
            for t_idx in idx:
                out = forward_one_day(int(t_idx))
                if not out:
                    continue
                y_hat_all[t_idx] = out["y_hat"].detach().cpu().numpy()
                eval_mask[t_idx] = mask[t_idx]
                retrieval_log.append({
                    "day_idx": int(t_idx),
                    "date": str(dates[int(t_idx)]),
                    "alpha": float(out["alpha"].cpu().numpy()),
                    "top1_sim": float(out["top1_sim"].cpu().numpy()),
                    "retrieved_day_idx": out["retrieved_day_indices"].cpu().tolist(),
                })
        ic, ic_per_day = per_day_ic(y_hat_all, y, eval_mask, rank=False)
        rank_ic, _ = per_day_ic(y_hat_all, y, eval_mask, rank=True)
        ndcg10 = ndcg_at_k(y_hat_all, y, eval_mask, 10)
        ndcg50 = ndcg_at_k(y_hat_all, y, eval_mask, 50)
        return {
            "label": label,
            "ic": ic,
            "rank_ic": rank_ic,
            "ndcg10": ndcg10,
            "ndcg50": ndcg50,
            "n_days_scored": int(np.sum(~np.isnan(ic_per_day))),
            "retrieval_log": retrieval_log,
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
        print(f"[epiSTAR] epoch {epoch}: train_loss={train_loss:.4f} "
              f"val_ic={val_metrics['ic']:.4f} val_rank_ic={val_metrics['rank_ic']:.4f}")

        if val_metrics["ic"] > best_val_ic + 1e-5:
            best_val_ic = val_metrics["ic"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= train_cfg.early_stop_patience:
                print(f"[epiSTAR] early stop at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate(test_idx, "test")
    val_metrics_final = evaluate(val_idx, "val")

    out_dir = Path(train_cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save predictions for downstream diagnostic.
    model.eval()
    pred_path = out_dir / f"fold{fold}_seed{seed}_predictions.npz"
    y_hat_test = np.zeros((x.shape[0], x.shape[1]), dtype=np.float32)
    pred_mask = np.zeros((x.shape[0], x.shape[1]), dtype=bool)
    with torch.no_grad():
        for t_idx in test_idx:
            out = forward_one_day(int(t_idx))
            if not out:
                continue
            y_hat_test[t_idx] = out["y_hat"].detach().cpu().numpy()
            pred_mask[t_idx] = mask[t_idx]
    np.savez_compressed(
        pred_path,
        y_hat=y_hat_test,
        y_true=y,
        mask=pred_mask,
        test_idx=np.asarray(test_idx, dtype=np.int64),
        tickers=np.asarray(tickers, dtype=str),
        dates=np.asarray([str(d) for d in dates], dtype=str),
    )
    print(f"[epiSTAR] wrote {pred_path}")
    # Save best-state checkpoint for re-evaluation later.
    if best_state is not None:
        ckpt_path = out_dir / f"fold{fold}_seed{seed}_ckpt.pt"
        torch.save(best_state, ckpt_path)
        print(f"[epiSTAR] wrote {ckpt_path}")

    out_path = out_dir / f"fold{fold}_seed{seed}.json"
    payload = {
        "fold": fold,
        "seed": seed,
        "model": "epiSTAR",
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
        "history": history,
        "config": {
            "train": asdict(train_cfg),
            "backbone": asdict(backbone_cfg),
            "memory": asdict(memory_cfg),
            "ablation": {
                "disable_gate": epistar_cfg.disable_gate,
                "disable_retrieval": epistar_cfg.disable_retrieval,
            },
            "episode_key_cols": EPISODE_KEY_COLS,
            "episode_value_dim": values.shape[1],
        },
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"[epiSTAR] wrote {out_path}")

    diag_path = out_dir / f"fold{fold}_seed{seed}_retrieval.csv"
    diag_rows = []
    for r in test_metrics["retrieval_log"]:
        for j, d_idx in enumerate(r["retrieved_day_idx"]):
            diag_rows.append({
                "query_date": r["date"],
                "rank": j,
                "retrieved_day_idx": d_idx,
                "retrieved_date": str(dates[d_idx]) if d_idx >= 0 else None,
                "top1_sim": r["top1_sim"],
                "alpha": r["alpha"],
            })
    pd.DataFrame(diag_rows).to_csv(diag_path, index=False)
    print(f"[epiSTAR] wrote {diag_path}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/epistar.yaml")
    p.add_argument("--fold", type=int, choices=[1, 2, 3], required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true",
                   help="Smoke-test mode: 2 epochs, 80 steps per epoch.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.config, args.fold, args.seed, smoke=args.smoke)
