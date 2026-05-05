"""Training loop for OW-epiSTAR v1 (focused survivorship-aware).

Differs from ``train_ow_epistar.py`` (v0) in five places:
    1. Builds the minimal mask family from the raw price panel
       (``src.v2.data.minimal_masks``) rather than relying on the
       feature-panel active mask. ``tradable_mask`` is used for graph
       construction, age computation, and as the model's "active" mask;
       ``loss_mask`` is used for the loss and for evaluation.
    2. Computes age features from ``tradable_mask`` so the early-life
       period of late IPOs is preserved.
    3. The dynamic graph is the reliability-shrunk graph from
       ``src.v2.graph.survivorship_dynamic_edges_v1`` with overlap
       counted against ``tradable_mask``.
    4. The IPO analogue key is the 22-dim ``IPO_ANALOGUE_KEY_COLS`` set
       (8 age/history + 11 panel + 3 macro features).
    5. The IPO retrieval gate input matches Section E of the spec
       exactly (11 dims).

Loss: cross-sectional MSE on per-day z-scored 5d forward log returns,
applied over ``loss_mask`` cells only. The model still emits scores
for every ``tradable_mask`` cell so the IPO retrieval path is exercised
on cells that are tradable but lack a 5d label (e.g. last 5 days).

Usage:
    python -m src.v2.training.train_ow_epistar_v1 \\
        --config configs/ow_epistar_v1.yaml --fold 1 --seed 42
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
from src.v2.data.minimal_masks import (
    MinimalMaskConfig, build_minimal_masks, compute_age_from_tradable,
)
from src.v2.graph.survivorship_dynamic_edges_v1 import (
    SurvivorshipGraphConfig, shrunk_correlation_neighbors_one_day,
)
from src.v2.model.episode_memory import EpisodeMemoryConfig
from src.v2.model.ipo_analogue_memory import (
    IPOMemoryConfig, IPO_ANALOGUE_KEY_COLS,
)
from src.v2.model.ow_epistar_v1 import (
    IPO_GATE_TICKER_FEATURES, OWEpiSTARV1, OWEpiSTARV1Config,
)
from src.v2.model.star_backbone import STARBackboneConfig
from src.v2.training.folds import fold_indices


@dataclass
class TrainConfig:
    """Top-level hyperparameters for OW-epiSTAR v1."""

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
    correlation_shrinkage_tau: float = 30.0
    min_overlap_absolute: int = 5
    output_dir: str = "results/ow_epistar_v1"
    # Ablation flags (mirrored from spec Section H).
    disable_ipo_retrieval: bool = False
    disable_ipo_gate: bool = False
    disable_correlation_shrinkage: bool = False
    random_ipo_retrieval: bool = False
    shuffle_age_features: bool = False


def set_seeds(seed: int) -> None:
    """Set seeds for Python, NumPy, and PyTorch."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cs_mse_loss(y_hat, y_true, mask):
    """Cross-sectional MSE on per-day z-scored true returns, masked."""
    m = mask.bool()
    yh = y_hat[m]; yt = y_true[m]
    if yt.numel() < 2:
        return torch.zeros((), device=y_hat.device, dtype=y_hat.dtype)
    mu = yt.mean(); sd = yt.std().clamp(min=1e-6)
    return ((yh - (yt - mu) / sd) ** 2).mean()


def per_day_ic(y_hat, y, mask, rank=False):
    """Daily-then-mean IC."""
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
    if np.all(np.isnan(ics)):
        return 0.0, ics
    return float(np.nanmean(ics)), ics


def ndcg_at_k(y_hat, y, mask, k):
    """Daily-mean NDCG at k."""
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


def cohort_ic(y_hat, y, eval_mask, age_days):
    """Per-cohort IC: all, fresh (<=60), young (61-252), seasoned (>252)."""
    out: dict[str, float] = {}
    cohorts = {
        "all": np.ones_like(eval_mask, dtype=bool),
        "fresh_ipo_60d": (age_days <= 60) & (age_days >= 1),
        "young_public_252d": (age_days > 60) & (age_days <= 252),
        "seasoned_253d": age_days > 252,
    }
    for label, cohort_mask in cohorts.items():
        ics = []
        for t in range(y_hat.shape[0]):
            m = eval_mask[t] & cohort_mask[t]
            if m.sum() < 5:
                continue
            a = y_hat[t, m]; b = y[t, m]
            sa = a.std(); sb = b.std()
            if sa < 1e-9 or sb < 1e-9:
                continue
            ics.append(float(np.corrcoef(a, b)[0, 1]))
        out[label] = float(np.mean(ics)) if ics else 0.0
    return out


def standardize_features(x, mask, train_idx):
    """Per-feature z-score using train-fold statistics; masked cells = 0."""
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
    """Linear warmup followed by cosine decay to 0.1x."""
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))


def build_age_features_from_tradable(
    tradable_mask: np.ndarray,
    history_valid_20d: np.ndarray,
    history_valid_60d: np.ndarray,
) -> np.ndarray:
    """8-dim age feature tensor [T, N, 8] using the cumsum age semantics.

    Columns (in order):
        0 age_trading_days
        1 log1p_age_trading_days
        2 age_bucket_0_20
        3 age_bucket_21_60
        4 age_bucket_61_252
        5 age_bucket_253_plus
        6 history_valid_ratio_20d
        7 history_valid_ratio_60d
    """
    age = compute_age_from_tradable(tradable_mask).astype(np.float32)
    out = np.zeros((*tradable_mask.shape, 8), dtype=np.float32)
    out[..., 0] = age
    out[..., 1] = np.log1p(age)
    out[..., 2] = ((age >= 1) & (age <= 20)).astype(np.float32)
    out[..., 3] = ((age > 20) & (age <= 60)).astype(np.float32)
    out[..., 4] = ((age > 60) & (age <= 252)).astype(np.float32)
    out[..., 5] = (age > 252).astype(np.float32)
    out[..., 6] = history_valid_20d
    out[..., 7] = history_valid_60d
    return out


def build_ipo_keys(
    x_raw: np.ndarray,
    age_feat: np.ndarray,
    risk_arr: np.ndarray,
    avg_corr_60d: np.ndarray,
) -> np.ndarray:
    """[T, N, 22] IPO analogue retrieval key per spec.

    Panel feature indices used:
        1:log_return_5d  2:log_return_20d  5:realized_vol_20d
        6:realized_vol_60d  9:st_volume_24h  11:st_bullish_ratio
        13:st_labeled_ratio  14:log_market_cap  15:cash_runway_q
        16:rd_intensity  18:cash_to_mc

    Risk feature columns used:
        vix (col 0) -> standardised externally to vix_z
        xbi_rv_20d (col 4)

    avg_corr_60d is the per-day cross-sectional avg pairwise correlation
    (broadcast across tickers).
    """
    t_total, n, _ = x_raw.shape
    keys = np.zeros((t_total, n, len(IPO_ANALOGUE_KEY_COLS)), dtype=np.float32)
    keys[..., 0:8] = age_feat
    keys[..., 8] = x_raw[..., 14]   # log_market_cap
    keys[..., 9] = x_raw[..., 15]   # cash_runway_q
    keys[..., 10] = x_raw[..., 18]  # cash_to_mc
    keys[..., 11] = x_raw[..., 16]  # rd_intensity
    keys[..., 12] = x_raw[..., 9]   # st_volume_24h
    keys[..., 13] = x_raw[..., 11]  # st_bullish_ratio
    keys[..., 14] = x_raw[..., 13]  # st_labeled_ratio
    keys[..., 15] = x_raw[..., 5]   # realized_vol_20d
    keys[..., 16] = x_raw[..., 6]   # realized_vol_60d
    keys[..., 17] = x_raw[..., 1]   # log_return_5d
    keys[..., 18] = x_raw[..., 2]   # log_return_20d

    vix = risk_arr[:, 0]
    vix_mu = float(np.nanmean(vix)); vix_sd = float(np.nanstd(vix))
    if vix_sd < 1e-6:
        vix_sd = 1.0
    vix_z = ((vix - vix_mu) / vix_sd).astype(np.float32)
    keys[..., 19] = np.broadcast_to(vix_z[:, None], (t_total, n))
    keys[..., 20] = np.broadcast_to(risk_arr[:, 4][:, None], (t_total, n))
    keys[..., 21] = np.broadcast_to(avg_corr_60d[:, None], (t_total, n))
    return keys


def main(cfg_path: str, fold: int, seed: int, smoke: bool = False) -> None:
    """Train OW-epiSTAR v1 on one (fold, seed) pair."""
    with open(cfg_path) as f:
        raw = yaml.safe_load(f)

    train_cfg = TrainConfig(**{**raw.get("train", {}), "fold": fold, "seed": seed})
    backbone_cfg = STARBackboneConfig(**raw.get("backbone", {}))
    day_mem_cfg = EpisodeMemoryConfig(**raw.get("day_memory", {}))
    # IPO memory config; honour the random-IPO-retrieval ablation flag.
    ipo_mem_kwargs = dict(raw.get("ipo_memory", {}))
    if train_cfg.random_ipo_retrieval:
        ipo_mem_kwargs["random_retrieval"] = True
    ipo_mem_cfg = IPOMemoryConfig(**ipo_mem_kwargs)
    ow_cfg = OWEpiSTARV1Config(
        backbone=backbone_cfg, day_memory=day_mem_cfg, ipo_memory=ipo_mem_cfg,
        episode_value_dim=raw.get("model", {}).get("episode_value_dim", 32),
        ipo_value_dim=raw.get("model", {}).get("ipo_value_dim", 32),
        cross_attn_heads=raw.get("model", {}).get("cross_attn_heads", 4),
        gate_hidden_dim=raw.get("model", {}).get("gate_hidden_dim", 64),
        head_hidden_dim=raw.get("model", {}).get("head_hidden_dim", 64),
        head_dropout=raw.get("model", {}).get("head_dropout", 0.1),
        disable_ipo_retrieval=train_cfg.disable_ipo_retrieval,
        disable_ipo_gate=train_cfg.disable_ipo_gate,
    )
    if smoke:
        train_cfg.epochs = 2

    set_seeds(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[OW-epiSTAR-v1] fold={fold} seed={seed} device={device}")

    panel_cfg = EnrichedPanelConfig(
        start_date=pd.Timestamp(train_cfg.panel_start),
        end_date=pd.Timestamp(train_cfg.panel_end),
        horizon_days=train_cfg.horizon_days,
        universe_csv=Path(train_cfg.universe_csv),
    )
    panel, tickers, dates = build_enriched_panel(panel_cfg)
    tens = panel_to_tensors(panel, tickers, dates)
    x_raw = tens["x"]; y = tens["y"]
    print(f"[OW-epiSTAR-v1] panel: T={x_raw.shape[0]} N={x_raw.shape[1]} F={x_raw.shape[2]}")
    if x_raw.shape[1] < 50:
        raise RuntimeError(f"Panel has only {x_raw.shape[1]} active tickers; aborting.")

    # Minimal mask family.
    mm = build_minimal_masks(
        dates, tickers,
        cfg=MinimalMaskConfig(horizon_days=train_cfg.horizon_days),
    )
    tradable = mm["tradable_mask"]
    label = mm["label_mask"]
    loss_mask = mm["loss_mask"]
    hist20 = mm["history_valid_20d"]
    hist60 = mm["history_valid_60d"]
    print(f"[OW-epiSTAR-v1] tradable_cells={int(tradable.sum())} "
          f"label_cells={int(label.sum())} loss_cells={int(loss_mask.sum())}")
    recovered = int((tradable & ~tens["mask"]).sum())
    print(f"[OW-epiSTAR-v1] recovered (tradable but excluded by old active mask): {recovered}")

    train_idx, val_idx, test_idx = fold_indices(fold, dates)
    print(f"[OW-epiSTAR-v1] fold {fold}: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    # Standardise features using train-fold tradable cells.
    x = standardize_features(x_raw, tradable, train_idx).astype(np.float32)

    # Age features from tradable mask (cumsum semantics).
    age_feat = build_age_features_from_tradable(tradable, hist20, hist60)
    age_days = age_feat[..., 0].astype(np.int64)
    if train_cfg.shuffle_age_features:
        rng = np.random.default_rng(seed)
        for f in range(age_feat.shape[-1]):
            for t in range(age_feat.shape[0]):
                idx_active = np.where(tradable[t])[0]
                if idx_active.size > 1:
                    perm = rng.permutation(idx_active)
                    vals = age_feat[t, idx_active, f].copy()
                    age_feat[t, perm, f] = vals
        print("[OW-epiSTAR-v1] age features SHUFFLED (sanity ablation)")

    # Day-level keys + values (same as epiSTAR-full, but use tradable
    # mask for the cross-sectional diagnostics).
    day_keys, _ = build_episode_keys(
        dates=dates, log_returns=x_raw[..., 0], mask=tradable,
        cfg=EpisodeKeyConfig(),
    )
    feature_idx = [0, 1, 5, 6]
    n_summary = 2 * len(feature_idx) + 1
    day_values = np.zeros((len(dates), day_keys.shape[1] + n_summary), dtype=np.float32)
    day_values[:, : day_keys.shape[1]] = day_keys
    for t in range(len(dates)):
        m = tradable[t]
        if m.sum() < 5:
            continue
        for j, fi in enumerate(feature_idx):
            v = x_raw[t, m, fi]
            day_values[t, day_keys.shape[1] + 2 * j] = float(np.mean(v))
            day_values[t, day_keys.shape[1] + 2 * j + 1] = float(np.std(v))
        day_values[t, -1] = float(m.sum()) / 250.0
    print(f"[OW-epiSTAR-v1] day-mem keys: {day_keys.shape[1]} dims, "
          f"values: {day_values.shape[1]} dims")

    # IPO 22-dim keys + values.
    risk_arr = pd.read_parquet("data/processed/risk_features.parquet").reindex(
        pd.to_datetime(dates)
    ).ffill().bfill().to_numpy(dtype=np.float32)
    avg_corr_60d = day_keys[:, EPISODE_KEY_COLS.index("cs_avg_pairwise_corr_60d")]
    ipo_keys = build_ipo_keys(x_raw, age_feat, risk_arr, avg_corr_60d)
    # IPO value: the IPO key itself + the realised 5d forward log return.
    ipo_value_extras = np.zeros((len(dates), x_raw.shape[1], 1), dtype=np.float32)
    ipo_value_extras[..., 0] = y
    ipo_values = np.concatenate([ipo_keys, ipo_value_extras], axis=-1)
    print(f"[OW-epiSTAR-v1] IPO keys: {ipo_keys.shape[-1]} dims, "
          f"values: {ipo_values.shape[-1]} dims")

    # IPO memory: tradable & label & age <= max_age, training-fold only.
    ipo_eligible = tradable & label & (age_days >= 1) & (age_days <= ipo_mem_cfg.max_age_days)
    train_set = set(int(t) for t in train_idx.tolist())
    flat_keys: list = []; flat_values: list = []
    flat_days: list = []; flat_tickers: list = []
    for t in range(len(dates)):
        if t not in train_set:
            continue
        active_tickers = np.where(ipo_eligible[t])[0]
        for i in active_tickers:
            flat_keys.append(ipo_keys[t, i])
            flat_values.append(ipo_values[t, i])
            flat_days.append(t)
            flat_tickers.append(int(i))
    flat_keys = np.asarray(flat_keys, dtype=np.float32) if flat_keys else np.zeros(
        (0, ipo_keys.shape[-1]), dtype=np.float32)
    flat_values = np.asarray(flat_values, dtype=np.float32) if flat_values else np.zeros(
        (0, ipo_values.shape[-1]), dtype=np.float32)
    flat_days = np.asarray(flat_days, dtype=np.int64) if len(flat_days) > 0 else np.zeros(
        (0,), dtype=np.int64)
    flat_tickers = np.asarray(flat_tickers, dtype=np.int64) if len(flat_tickers) > 0 else np.zeros(
        (0,), dtype=np.int64)
    print(f"[OW-epiSTAR-v1] IPO memory entries (train-fold, age<={ipo_mem_cfg.max_age_days}d): "
          f"{len(flat_keys)}")

    ow_cfg.episode_value_dim = day_values.shape[1]
    ow_cfg.ipo_value_dim = ipo_values.shape[-1]
    model = OWEpiSTARV1(
        ow_cfg, day_key_dim=day_keys.shape[1], ipo_key_dim=ipo_keys.shape[-1],
    ).to(device)
    model.day_memory.populate(
        keys=day_keys, values=day_values,
        day_indices=np.arange(len(dates)), train_day_indices=train_idx,
    )
    model.day_memory.to(device)
    model.ipo_memory.populate(
        keys=flat_keys, values=flat_values,
        day_indices=flat_days, ticker_indices=flat_tickers,
        train_day_indices=train_idx,
    )
    model.ipo_memory.to(device)

    allowed_train = torch.from_numpy(train_idx).long().to(device)

    optim = AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay,
    )
    total_steps = train_cfg.epochs * max(1, len(train_idx))
    scheduler = LambdaLR(
        optim, lr_lambda=lambda s: warmup_cosine_lr(s, train_cfg.warmup_steps, total_steps)
    )
    use_amp = device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    w = backbone_cfg.temporal_window
    cw = train_cfg.correlation_window
    best_val_ic = -1e9; best_state = None; patience = 0
    history: list[dict] = []
    graph_cfg = SurvivorshipGraphConfig(
        corr_window=cw,
        tau=train_cfg.correlation_shrinkage_tau,
        top_k=backbone_cfg.num_neighbors,
        min_overlap_absolute=train_cfg.min_overlap_absolute,
    )

    def forward_one_day(t_idx: int) -> dict:
        if t_idx < max(w, cw):
            return {}
        active_mask_t = torch.from_numpy(tradable[t_idx]).to(device)
        if active_mask_t.sum() < 5:
            return {}
        active_idx = active_mask_t.nonzero(as_tuple=False).squeeze(-1)

        if train_cfg.disable_correlation_shrinkage:
            # Plain rolling-correlation graph (no shrinkage). Reuse the
            # function but with tau=0.
            cfg_no_shrink = SurvivorshipGraphConfig(
                corr_window=cw, tau=0.0, top_k=backbone_cfg.num_neighbors,
                min_overlap_absolute=train_cfg.min_overlap_absolute,
            )
            neigh, _ = shrunk_correlation_neighbors_one_day(
                x_raw[..., 0], tradable, int(t_idx), cfg_no_shrink,
            )
        else:
            neigh, _ = shrunk_correlation_neighbors_one_day(
                x_raw[..., 0], tradable, int(t_idx), graph_cfg,
            )
        top_neighbors_day_t = torch.from_numpy(neigh).to(device)
        x_window = torch.from_numpy(x[t_idx - w + 1 : t_idx + 1]).to(device)
        mask_window = torch.from_numpy(tradable[t_idx - w + 1 : t_idx + 1]).to(device)
        patches, patch_mask = build_patches(
            x_window=x_window, mask_window=mask_window,
            top_neighbors=top_neighbors_day_t, active_idx=active_idx,
        )

        day_query_key = torch.from_numpy(day_keys[t_idx]).float().to(device)
        regime_scalars = model.day_memory.standardize_query(day_query_key)[[0, 9]].clone()
        if torch.isnan(regime_scalars).any():
            regime_scalars = torch.zeros(2, device=device)

        active_idx_np = active_idx.detach().cpu().numpy()
        ipo_query_keys = torch.from_numpy(
            ipo_keys[t_idx, active_idx_np]
        ).float().to(device)
        # ipo_gate_features: 7 per-ticker columns per the spec.
        # Indices into age_feat: 0=age, 1=log1p_age, 6=hist20, 7=hist60.
        # Panel feature index for has_fundamentals = 21.
        gate_age = age_feat[t_idx, active_idx_np][:, [0, 1, 6, 7]]
        has_fund = x_raw[t_idx, active_idx_np, 21:22]
        st_lab = x_raw[t_idx, active_idx_np, 13:14]
        rv20 = x_raw[t_idx, active_idx_np, 5:6]
        gate_feats_np = np.concatenate([gate_age, has_fund, st_lab, rv20], axis=-1)
        ipo_gate_feats = torch.from_numpy(gate_feats_np).float().to(device)

        with autocast(enabled=use_amp, dtype=torch.float16):
            out = model.forward_day(
                patches=patches, patch_mask=patch_mask, active_mask=active_mask_t,
                day_query_key=day_query_key, ipo_query_keys=ipo_query_keys,
                ipo_gate_features=ipo_gate_feats,
                query_day_idx=int(t_idx), allowed_day_indices=allowed_train,
                gate_regime_scalars=regime_scalars,
            )
        out["y_hat"] = out["y_hat"].float()
        out["alpha_ipo"] = out["alpha_ipo"].float()
        out["active_mask"] = active_mask_t
        out["t_idx"] = t_idx
        out["active_idx"] = active_idx
        return out

    @torch.no_grad()
    def evaluate(idx, eval_mask_arr):
        """Evaluate with a passed-in eval mask (loss_mask for IC calc)."""
        model.eval()
        T = x.shape[0]; N = x.shape[1]
        y_hat_all = np.zeros((T, N), dtype=np.float32)
        emask = np.zeros((T, N), dtype=bool)
        alpha_ipo_log: list[np.ndarray] = []
        for t_idx in idx:
            out = forward_one_day(int(t_idx))
            if not out:
                continue
            y_hat_all[t_idx] = out["y_hat"].detach().cpu().numpy()
            emask[t_idx] = eval_mask_arr[t_idx]
            ai = out["alpha_ipo"].detach().cpu().numpy()
            if ai.size > 0:
                alpha_ipo_log.append(ai)
        ic, _ = per_day_ic(y_hat_all, y, emask, rank=False)
        rank_ic, _ = per_day_ic(y_hat_all, y, emask, rank=True)
        ndcg10 = ndcg_at_k(y_hat_all, y, emask, 10)
        ndcg50 = ndcg_at_k(y_hat_all, y, emask, 50)
        coh = cohort_ic(y_hat_all, y, emask, age_days)
        alpha_mean = float(np.mean(np.concatenate(alpha_ipo_log))) if alpha_ipo_log else 0.0
        return {"ic": ic, "rank_ic": rank_ic, "ndcg10": ndcg10, "ndcg50": ndcg50,
                "cohort_ic": coh, "alpha_ipo_mean": alpha_mean,
                "y_hat_all": y_hat_all, "eval_mask": emask}

    step = 0; smoke_step_cap = 80
    for epoch in range(train_cfg.epochs):
        model.train()
        np.random.seed(seed + epoch)
        perm = np.random.permutation(train_idx)
        epoch_losses: list[float] = []
        for t_idx in perm:
            t_idx = int(t_idx)
            if t_idx < max(w, cw):
                continue
            out = forward_one_day(t_idx)
            if not out:
                continue
            # Loss uses loss_mask, NOT tradable_mask (per spec).
            loss_mask_t = torch.from_numpy(loss_mask[t_idx]).to(device)
            y_true_t = torch.from_numpy(y[t_idx]).to(device)
            loss = cs_mse_loss(out["y_hat"], y_true_t, loss_mask_t)
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
        val_metrics = evaluate(val_idx, eval_mask_arr=loss_mask)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_ic": val_metrics["ic"],
            "val_rank_ic": val_metrics["rank_ic"],
            "val_alpha_ipo_mean": val_metrics["alpha_ipo_mean"],
        })
        print(f"[OW-epiSTAR-v1] epoch {epoch}: train_loss={train_loss:.4f} "
              f"val_ic={val_metrics['ic']:.4f} val_rank_ic={val_metrics['rank_ic']:.4f} "
              f"alpha_ipo={val_metrics['alpha_ipo_mean']:.3f}")
        if val_metrics["ic"] > best_val_ic + 1e-5:
            best_val_ic = val_metrics["ic"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= train_cfg.early_stop_patience:
                print(f"[OW-epiSTAR-v1] early stop at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate(test_idx, eval_mask_arr=loss_mask)
    val_metrics_final = evaluate(val_idx, eval_mask_arr=loss_mask)

    out_dir = Path(train_cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / f"fold{fold}_seed{seed}_predictions.npz"
    np.savez_compressed(
        pred_path,
        y_hat=test_metrics["y_hat_all"], y_true=y,
        loss_mask=test_metrics["eval_mask"],
        tradable_mask=tradable,
        test_idx=np.asarray(test_idx, dtype=np.int64),
        tickers=np.asarray(tickers, dtype=str),
        dates=np.asarray([str(d) for d in dates], dtype=str),
        age_days=age_days.astype(np.int32),
    )
    if best_state is not None:
        torch.save(best_state, out_dir / f"fold{fold}_seed{seed}_ckpt.pt")
    out_path = out_dir / f"fold{fold}_seed{seed}.json"

    test_metrics.pop("y_hat_all", None); test_metrics.pop("eval_mask", None)
    val_metrics_final.pop("y_hat_all", None); val_metrics_final.pop("eval_mask", None)

    payload = {
        "fold": fold, "seed": seed, "model": "OW-epiSTAR-v1",
        "panel_start": train_cfg.panel_start, "panel_end": train_cfg.panel_end,
        "n_tickers": int(x.shape[1]), "n_dates": int(x.shape[0]),
        "n_train": int(len(train_idx)), "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "n_ipo_memory_entries": int(len(flat_keys)),
        "tradable_cells": int(tradable.sum()),
        "label_cells": int(label.sum()),
        "loss_cells": int(loss_mask.sum()),
        "recovered_cells_vs_old_active": int(recovered),
        "ic": test_metrics["ic"], "rank_ic": test_metrics["rank_ic"],
        "ndcg10": test_metrics["ndcg10"], "ndcg50": test_metrics["ndcg50"],
        "test_cohort_ic": test_metrics["cohort_ic"],
        "test_alpha_ipo_mean": test_metrics["alpha_ipo_mean"],
        "val_ic": val_metrics_final["ic"], "val_rank_ic": val_metrics_final["rank_ic"],
        "val_cohort_ic": val_metrics_final["cohort_ic"],
        "best_val_ic": best_val_ic, "history": history,
        "config": {"train": asdict(train_cfg),
                   "backbone": asdict(backbone_cfg),
                   "day_memory": asdict(day_mem_cfg),
                   "ipo_memory": asdict(ipo_mem_cfg),
                   "model": {"head_hidden_dim": ow_cfg.head_hidden_dim,
                             "head_dropout": ow_cfg.head_dropout,
                             "episode_value_dim": ow_cfg.episode_value_dim,
                             "ipo_value_dim": ow_cfg.ipo_value_dim,
                             "ipo_gate_ticker_features": IPO_GATE_TICKER_FEATURES,
                             "ipo_key_cols": IPO_ANALOGUE_KEY_COLS},
                   "graph": asdict(graph_cfg)},
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"[OW-epiSTAR-v1] wrote {out_path}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/ow_epistar_v1.yaml")
    p.add_argument("--fold", type=int, choices=[1, 2, 3], required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.config, args.fold, args.seed, smoke=args.smoke)
