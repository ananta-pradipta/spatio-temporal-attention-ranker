"""Training loop for epiDyReg-STAR (full v1 spec).

Differs from train_epistar.py in three places:
    1. Per-day candidate scoring uses three sources (static mechanistic,
       rolling return-correlation, rolling residual-correlation), mixed
       with per-day softmax weights produced from the regime context.
    2. Age-aware: tickers with age < min_age_for_corr_edges are excluded
       from rolling-corr / residual-corr candidate pools.
    3. Episode key is augmented with a 6-dim per-day graph-summary
       vector (mean-abs-corr, PC1-share proxy, density, turnover,
       active-count norm, score std).

All other components (STAR backbone, episodic memory bank with
leakage-safe retrieval, cross-attention fusion, day-level confidence
gate, single rank head, cross-sectional MSE loss, AMP, prediction and
checkpoint saving) are reused from the existing v2 infrastructure.
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

from src.mtgn.model.utils.patch_construction import build_patches
from src.mtgn.training.panel_enriched import (
    EnrichedPanelConfig, build_enriched_panel, panel_to_tensors,
)
from src.v2.data.episode_keys import (
    EpisodeKeyConfig, EPISODE_KEY_COLS, build_episode_keys,
)
from src.v2.graph.multi_source_dynamic import (
    MultiSourceGraphConfig,
    build_static_score,
    correlation_window_matrix,
    residual_correlation_window_matrix,
    compute_age_days,
    gate_weighted_top_k,
    graph_summary_features,
)
from src.v2.model.episode_memory import EpisodeMemoryConfig
from src.v2.model.epidyreg_star import (
    EpiDyRegSTAR, EpiDyRegSTARConfig, GraphSourceMixerConfig,
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
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_steps: int = 500
    grad_clip: float = 1.0
    early_stop_patience: int = 3
    correlation_window: int = 60
    min_age_for_corr_edges: int = 60
    output_dir: str = "results/epidyreg_star"


def set_seeds(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cs_mse_loss(y_hat: torch.Tensor, y_true: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Cross-sectional MSE on per-day z-scored true returns."""
    m = mask.bool()
    yh = y_hat[m]; yt = y_true[m]
    if yt.numel() < 2:
        return torch.zeros((), device=y_hat.device, dtype=y_hat.dtype)
    mu = yt.mean(); sd = yt.std().clamp(min=1e-6)
    return ((yh - (yt - mu) / sd) ** 2).mean()


def per_day_ic(y_hat, y, mask, rank=False):
    t_total = y_hat.shape[0]
    ics = np.full(t_total, np.nan, dtype=np.float64)
    for t in range(t_total):
        m = mask[t]
        if m.sum() < 5:
            continue
        a = y_hat[t, m]; b = y[t, m]
        if rank:
            a = pd.Series(a).rank().to_numpy()
            b = pd.Series(b).rank().to_numpy()
        sa = a.std(); sb = b.std()
        if sa < 1e-9 or sb < 1e-9:
            continue
        ics[t] = float(np.corrcoef(a, b)[0, 1])
    return (np.nanmean(ics) if not np.all(np.isnan(ics)) else 0.0, ics)


def ndcg_at_k(y_hat, y, mask, k):
    out = []
    for t in range(y_hat.shape[0]):
        m = mask[t]
        if m.sum() < k + 1:
            continue
        scores = y_hat[t, m]; rels = y[t, m]
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


def standardize_features(x, mask, train_idx):
    flat_train_mask = mask[train_idx]; x_train = x[train_idx]
    out = np.zeros_like(x)
    for f in range(x.shape[2]):
        vals = x_train[..., f][flat_train_mask]
        if vals.size < 2:
            mu, sd = 0.0, 1.0
        else:
            mu = float(np.mean(vals)); sd = float(np.std(vals))
            if sd < 1e-6:
                sd = 1.0
        out[..., f] = (x[..., f] - mu) / sd
    out = out * mask[..., None]
    return out


def warmup_cosine_lr(step, warmup, total):
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))


def main(cfg_path: str, fold: int, seed: int, smoke: bool = False) -> None:
    """Train epiDyReg-STAR on one (fold, seed) pair."""
    with open(cfg_path) as f:
        raw = yaml.safe_load(f)
    train_cfg = TrainConfig(**{**raw.get("train", {}), "fold": fold, "seed": seed})
    backbone_cfg = STARBackboneConfig(**raw.get("backbone", {}))
    memory_cfg = EpisodeMemoryConfig(**raw.get("memory", {}))
    mixer_cfg = GraphSourceMixerConfig(**raw.get("mixer", {}))
    edyr_cfg = EpiDyRegSTARConfig(
        backbone=backbone_cfg, memory=memory_cfg, mixer=mixer_cfg,
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
    print(f"[epiDyReg-STAR] fold={fold} seed={seed} device={device}")

    panel_cfg = EnrichedPanelConfig(
        start_date=pd.Timestamp(train_cfg.panel_start),
        end_date=pd.Timestamp(train_cfg.panel_end),
        horizon_days=train_cfg.horizon_days,
        universe_csv=Path(train_cfg.universe_csv),
    )
    panel, tickers, dates = build_enriched_panel(panel_cfg)
    tens = panel_to_tensors(panel, tickers, dates)
    x_raw = tens["x"]; y = tens["y"]; mask = tens["mask"]
    print(f"[epiDyReg-STAR] panel: T={x_raw.shape[0]} N={x_raw.shape[1]} F={x_raw.shape[2]}")
    if x_raw.shape[1] < 50:
        raise RuntimeError(f"Panel has only {x_raw.shape[1]} active tickers; aborting.")

    train_idx, val_idx, test_idx = fold_indices(fold, dates)
    print(f"[epiDyReg-STAR] fold {fold}: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    x = standardize_features(x_raw, mask, train_idx).astype(np.float32)

    # Pre-compute static mechanistic score and per-day age.
    msg_cfg = MultiSourceGraphConfig(
        rolling_window_days=train_cfg.correlation_window,
        top_k_neighbors=backbone_cfg.num_neighbors,
        min_age_for_corr_edges=train_cfg.min_age_for_corr_edges,
    )
    static_score = build_static_score(tickers, msg_cfg)
    age_days = compute_age_days(mask)
    print(f"[epiDyReg-STAR] static-graph nonzero pairs: {int((static_score > 0).sum())}")

    # Episode keys: 14-dim risk+cs base, plus 6-dim graph summary appended later.
    base_keys, key_cols = build_episode_keys(
        dates=dates, log_returns=x_raw[..., 0], mask=mask, cfg=EpisodeKeyConfig(),
    )
    base_keys = base_keys.astype(np.float32)
    K_BASE = base_keys.shape[1]
    K_GRAPH = 6
    keys = np.zeros((len(dates), K_BASE + K_GRAPH), dtype=np.float32)
    keys[:, :K_BASE] = base_keys
    print(f"[epiDyReg-STAR] episode keys: K_base={K_BASE} + K_graph={K_GRAPH}")

    # Pre-compute static-source candidate scores for the per-day graph mixer.
    # We compute rolling-corr and residual-corr per day on-the-fly because
    # they depend on day-t-only data; static score is fixed.

    # Episode values (day-level summary): K_base + K_graph (= keys) plus
    # 8 STAR-feature summaries plus 1 active count.
    # We'll build this once after we have all per-day graph summaries.

    prev_neighbors_arr = np.full((x.shape[1], backbone_cfg.num_neighbors), -1, dtype=np.int64)
    daily_graph_summary = np.zeros((len(dates), K_GRAPH), dtype=np.float32)

    # Pre-compute per-day source scores in advance (CPU) so the training
    # loop only does the regime-conditioned mixing at GPU time.
    print(f"[epiDyReg-STAR] precomputing per-day source scores...")
    n_dates = len(dates)
    # We won't store all NxN matrices (too large). Instead we precompute,
    # per-day, the 3 per-source candidate top-K_per_source neighbor lists
    # plus their scores; mixing happens at training time.
    K_PER_SRC = msg_cfg.candidate_k_per_source
    static_top = _precompute_static_top(static_score, msg_cfg.top_k_neighbors * 2, mask, age_days, msg_cfg.min_age_for_corr_edges, age_relevant=False)

    print(f"[epiDyReg-STAR] starting training")
    edyr_cfg.episode_value_dim = K_BASE + K_GRAPH + 8 + 1
    model = EpiDyRegSTAR(edyr_cfg, episode_key_dim=K_BASE + K_GRAPH).to(device)
    allowed_train = torch.from_numpy(train_idx).long().to(device)
    optim = AdamW(model.parameters(), lr=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay)
    total_steps = train_cfg.epochs * max(1, len(train_idx))
    scheduler = LambdaLR(optim, lr_lambda=lambda s: warmup_cosine_lr(s, train_cfg.warmup_steps, total_steps))
    use_amp = device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    w = backbone_cfg.temporal_window

    def build_day_neighbors_and_summary(t_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (neighbors [N, K] int64, graph_summary [6])."""
        nonlocal prev_neighbors_arr
        active_t = mask[t_idx]
        if t_idx < train_cfg.correlation_window:
            # Insufficient history for rolling-corr graphs; use static only.
            neigh = gate_weighted_top_k(
                static_score, active_t, msg_cfg.top_k_neighbors,
                age_days=age_days[t_idx], min_age=0, age_relevant=False,
            )
            summary = graph_summary_features(static_score, active_t, prev_neighbors_arr)
            prev_neighbors_arr = neigh
            return neigh, summary

        corr = correlation_window_matrix(x_raw[..., 0], int(t_idx), train_cfg.correlation_window)
        resid = residual_correlation_window_matrix(x_raw[..., 0], int(t_idx), train_cfg.correlation_window)

        # Compute regime-source weights using the raw key for this day
        # (graph-summary slots are zero pre-computation; we plug in the
        # PREVIOUS day's summary as a regime cue, falling back to zeros
        # for the very first day).
        raw_key_t = keys[t_idx].copy()
        if t_idx > 0:
            raw_key_t[K_BASE:] = daily_graph_summary[t_idx - 1]
        rkey = torch.from_numpy(raw_key_t).float().to(device)
        with torch.no_grad():
            weights = model.gate_weights_for_day(rkey).cpu().numpy()

        # Mix sources. Each is N by N; product is the score for selection.
        score = (
            weights[0] * static_score
            + weights[1] * np.where(np.isfinite(corr), corr, 0.0)
            + weights[2] * np.where(np.isfinite(resid), resid, 0.0)
        )
        np.fill_diagonal(score, -np.inf)

        neigh = gate_weighted_top_k(
            score, active_t, msg_cfg.top_k_neighbors,
            age_days=age_days[t_idx], min_age=msg_cfg.min_age_for_corr_edges,
            age_relevant=True,
        )
        summary = graph_summary_features(score, active_t, prev_neighbors_arr)
        daily_graph_summary[t_idx] = summary
        keys[t_idx, K_BASE:] = summary
        prev_neighbors_arr = neigh
        return neigh, summary

    # First, populate the memory bank with day-level values. We need
    # graph summaries for every panel day; do a streaming pre-pass to
    # fill keys (with graph summary) and values.
    print(f"[epiDyReg-STAR] streaming graph summaries across {n_dates} days...")
    for t_idx in range(n_dates):
        if t_idx % 250 == 0:
            print(f"  pre-pass day {t_idx}/{n_dates}")
        if mask[t_idx].sum() < 5:
            continue
        if t_idx < train_cfg.correlation_window:
            summary = graph_summary_features(static_score, mask[t_idx], None)
        else:
            corr = correlation_window_matrix(x_raw[..., 0], int(t_idx), train_cfg.correlation_window)
            resid = residual_correlation_window_matrix(x_raw[..., 0], int(t_idx), train_cfg.correlation_window)
            mixed = static_score + np.where(np.isfinite(corr), corr, 0.0) + np.where(np.isfinite(resid), resid, 0.0)
            np.fill_diagonal(mixed, -np.inf)
            summary = graph_summary_features(mixed, mask[t_idx], None)
        daily_graph_summary[t_idx] = summary
        keys[t_idx, K_BASE:] = summary

    # Episode values: keys (K_BASE + K_GRAPH) plus 8 STAR-feature summaries plus 1 active count.
    feature_idx = [0, 1, 5, 6]
    n_summary = 2 * len(feature_idx) + 1
    values = np.zeros((n_dates, K_BASE + K_GRAPH + n_summary), dtype=np.float32)
    values[:, : K_BASE + K_GRAPH] = keys
    for t in range(n_dates):
        m = mask[t]
        if m.sum() < 5:
            continue
        for j, fi in enumerate(feature_idx):
            v = x_raw[t, m, fi]
            values[t, K_BASE + K_GRAPH + 2 * j] = float(np.mean(v))
            values[t, K_BASE + K_GRAPH + 2 * j + 1] = float(np.std(v))
        values[t, -1] = float(m.sum()) / 250.0

    edyr_cfg.episode_value_dim = values.shape[1]
    # Re-create the value projection layer with the new value dim.
    d = backbone_cfg.hidden_dim
    model.episode_value_proj = torch.nn.Linear(values.shape[1], d).to(device)
    new_mem = type(model.memory)(
        model.memory.cfg, key_dim=keys.shape[1], value_dim=values.shape[1]
    )
    new_mem.populate(
        keys=keys, values=values,
        day_indices=np.arange(n_dates),
        train_day_indices=train_idx,
    )
    model.memory = new_mem.to(device)

    best_val_ic = -1e9; best_state = None; patience = 0
    history: list[dict] = []

    def forward_one_day(t_idx: int) -> dict:
        if t_idx < w:
            return {}
        active_mask_t = torch.from_numpy(mask[t_idx]).to(device)
        if active_mask_t.sum() < 5:
            return {}
        active_idx = active_mask_t.nonzero(as_tuple=False).squeeze(-1)

        neigh, _ = build_day_neighbors_and_summary(int(t_idx))
        top_neighbors_day_t = torch.from_numpy(neigh).to(device)

        x_window = torch.from_numpy(x[t_idx - w + 1 : t_idx + 1]).to(device)
        mask_window = torch.from_numpy(mask[t_idx - w + 1 : t_idx + 1]).to(device)
        patches, patch_mask = build_patches(
            x_window=x_window, mask_window=mask_window,
            top_neighbors=top_neighbors_day_t, active_idx=active_idx,
        )
        query_raw_key = torch.from_numpy(keys[t_idx]).float().to(device)
        regime_scalars = model.memory.standardize_query(query_raw_key)[[0, 9]].clone()
        if torch.isnan(regime_scalars).any():
            regime_scalars = torch.zeros(2, device=device)

        with autocast(enabled=use_amp, dtype=torch.float16):
            out = model.forward_day(
                patches=patches, patch_mask=patch_mask, active_mask=active_mask_t,
                query_raw_key=query_raw_key, query_day_idx=int(t_idx),
                allowed_day_indices=allowed_train,
                gate_regime_scalars=regime_scalars,
            )
        out["y_hat"] = out["y_hat"].float()
        out["active_mask"] = active_mask_t
        out["t_idx"] = t_idx
        return out

    @torch.no_grad()
    def evaluate(idx):
        model.eval()
        T = x.shape[0]; N = x.shape[1]
        y_hat_all = np.zeros((T, N), dtype=np.float32)
        eval_mask = np.zeros((T, N), dtype=bool)
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
        return {"ic": ic, "rank_ic": rank_ic, "ndcg10": ndcg10, "ndcg50": ndcg50,
                "y_hat_all": y_hat_all, "eval_mask": eval_mask}

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
            loss = cs_mse_loss(out["y_hat"], y_true_t, mask_t)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            scaler.step(optim); scaler.update(); scheduler.step()
            epoch_losses.append(float(loss.item()))
            step += 1
            if smoke and step >= smoke_step_cap:
                break

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        val_metrics = evaluate(val_idx)
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_ic": val_metrics["ic"], "val_rank_ic": val_metrics["rank_ic"]})
        print(f"[epiDyReg-STAR] epoch {epoch}: train_loss={train_loss:.4f} "
              f"val_ic={val_metrics['ic']:.4f} val_rank_ic={val_metrics['rank_ic']:.4f}")
        if val_metrics["ic"] > best_val_ic + 1e-5:
            best_val_ic = val_metrics["ic"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= train_cfg.early_stop_patience:
                print(f"[epiDyReg-STAR] early stop at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate(test_idx)
    val_metrics_final = evaluate(val_idx)

    out_dir = Path(train_cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_path = out_dir / f"fold{fold}_seed{seed}_predictions.npz"
    np.savez_compressed(
        pred_path,
        y_hat=test_metrics["y_hat_all"], y_true=y, mask=test_metrics["eval_mask"],
        test_idx=np.asarray(test_idx, dtype=np.int64),
        tickers=np.asarray(tickers, dtype=str),
        dates=np.asarray([str(d) for d in dates], dtype=str),
        graph_summary=daily_graph_summary,
    )
    if best_state is not None:
        torch.save(best_state, out_dir / f"fold{fold}_seed{seed}_ckpt.pt")

    out_path = out_dir / f"fold{fold}_seed{seed}.json"
    payload = {
        "fold": fold, "seed": seed, "model": "epiDyReg-STAR",
        "panel_start": train_cfg.panel_start, "panel_end": train_cfg.panel_end,
        "n_tickers": int(x.shape[1]), "n_dates": int(x.shape[0]),
        "n_train": int(len(train_idx)), "n_val": int(len(val_idx)), "n_test": int(len(test_idx)),
        "ic": test_metrics["ic"], "rank_ic": test_metrics["rank_ic"],
        "ndcg10": test_metrics["ndcg10"], "ndcg50": test_metrics["ndcg50"],
        "val_ic": val_metrics_final["ic"], "val_rank_ic": val_metrics_final["rank_ic"],
        "best_val_ic": best_val_ic, "history": history,
        "config": {"train": asdict(train_cfg), "backbone": asdict(backbone_cfg),
                   "memory": asdict(memory_cfg), "mixer": asdict(mixer_cfg),
                   "model": {"head_hidden_dim": edyr_cfg.head_hidden_dim,
                             "head_dropout": edyr_cfg.head_dropout,
                             "episode_value_dim": values.shape[1]},
                   "episode_key_cols": EPISODE_KEY_COLS + ["graph_summary_dim_" + str(i) for i in range(K_GRAPH)]},
    }
    test_metrics.pop("y_hat_all", None); test_metrics.pop("eval_mask", None)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"[epiDyReg-STAR] wrote {out_path}")


def _precompute_static_top(static_score, k, mask, age_days, min_age, age_relevant):
    """Cache the static graph's top-K candidate set per ticker."""
    n = static_score.shape[0]
    out = np.full((n, k), -1, dtype=np.int64)
    s = static_score.copy()
    np.fill_diagonal(s, -np.inf)
    part = np.argpartition(-s, kth=min(k, n - 1), axis=1)[:, :k]
    for i in range(n):
        row_scores = s[i, part[i]]
        valid = np.isfinite(row_scores) & (row_scores > 0)
        chosen = part[i][valid]
        out[i, : len(chosen)] = chosen
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/epidyreg_star.yaml")
    p.add_argument("--fold", type=int, choices=[1, 2, 3], required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.config, args.fold, args.seed, smoke=args.smoke)
