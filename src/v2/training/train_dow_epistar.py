"""DOW-epiSTAR v2 trainer (Phase 1: macro-duration head only).

Builds on the OW-epiSTAR v1 trainer with three additions:

    1. Loads the macro_duration_features parquet and standardises with
       train-fold statistics.
    2. Loads the rolling_macro_betas parquet and broadcasts to a
       [T, N, 10] tensor.
    3. At each forward, passes per-ticker duration_input and per-day
       macro_input + macro_gate_input to DOWEpiSTAR.

End-to-end training (no staged freezing in v2.0). If end-to-end is
unstable, the spec's Stage 0-5 staged training is the v2.1 plan.
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
from src.v2.data.macro_duration_features import (
    MACRO_FEATURE_COLS_FULL, MACRO_GATE_COLS, MacroDurationConfig,
    build_macro_duration_features, standardize_macro_duration,
)
from src.v2.data.minimal_masks import (
    MinimalMaskConfig, build_minimal_masks, compute_age_from_tradable,
)
from src.v2.data.rolling_macro_betas import (
    ROLLING_BETA_COLS, RollingBetaConfig, betas_to_tensor, build_rolling_betas,
)
from src.v2.graph.duration_dynamic_edges import (
    DURATION_GRAPH_FEATURE_COLS, build_duration_similarity,
    merge_corr_and_duration,
)
from src.v2.graph.survivorship_dynamic_edges_v1 import (
    SurvivorshipGraphConfig, shrunk_correlation_neighbors_one_day,
)
from src.v2.model.dow_epistar import DOWEpiSTAR, DOWEpiSTARConfig
from src.v2.model.duration_exposure import DurationExposureConfig
from src.v2.model.episode_memory import EpisodeMemoryConfig
from src.v2.model.ipo_analogue_memory import (
    IPOMemoryConfig, IPO_ANALOGUE_KEY_COLS,
)
from src.v2.model.macro_state import MacroStateConfig
from src.v2.model.ow_epistar_v1 import (
    IPO_GATE_TICKER_FEATURES, OWEpiSTARV1Config,
)
from src.v2.model.rate_shock_memory import (
    RATE_SHOCK_KEY_COLS, RATE_VALUE_CS_SUMMARY_COLS,
    RATE_VALUE_DURATION_SUMMARY_COLS, RateShockMemoryConfig,
)
from src.v2.model.star_backbone import STARBackboneConfig
from src.v2.training.folds import fold_indices


@dataclass
class TrainConfig:
    """Top-level hyperparameters for DOW-epiSTAR v2 training."""

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
    output_dir: str = "results/dow_epistar_v2"
    # Ablation flags (subset of spec Section K).
    disable_macro_duration: bool = False
    disable_lambda_macro: bool = False
    disable_score_duration: bool = False
    shuffle_macro_state: bool = False
    shuffle_duration_input: bool = False
    random_ipo_retrieval: bool = False
    shuffle_age_features: bool = False
    disable_correlation_shrinkage: bool = False
    use_rate_memory: bool = True
    disable_alpha_rate: bool = False
    disable_score_rate: bool = False
    random_rate_retrieval: bool = False
    use_duration_graph: bool = True
    duration_graph_min_warmup_days: int = 252  # need DurationEnc warmed up
    # v2.3 patch E: pick which duration-similarity source to use.
    #   "deterministic_features" (default v2.3): hand-engineered 10-dim
    #     features standardised with train-fold stats.
    #   "learned_encoder" (v2.2 default): cosine over the live
    #     DurationExposureEncoder output.
    duration_graph_source: str = "deterministic_features"
    # ===== Graph-blending ablation knobs (added 2026-05-04 for Section 6
    # ablation in the paper). graph_mode in:
    #   "learned_blend" (default; current headline): macro-conditioned
    #     softmax over (A_corr, A_rate); requires use_duration_graph=True.
    #   "fixed_blend": fixed (fixed_blend_w_corr, fixed_blend_w_rate)
    #     with no learned gate.
    #   "corr_only": ignore A_rate; use reliability-shrunk correlation
    #     graph alone.
    #   "rate_only": ignore A_corr; use rate-sensitivity graph alone.
    #   "no_graph": no spatial neighbours at all (each ticker uses only
    #     its own temporal window).
    #   "random_graph": top-K random neighbours per active ticker.
    graph_mode: str = "learned_blend"
    fixed_blend_w_corr: float = 0.8
    fixed_blend_w_rate: float = 0.2
    # When True, after computing the rate-sensitivity graph adjacency,
    # permute its rows/cols across active tickers per day. Tests whether
    # the actual rate-sensitive topology matters.
    shuffle_rate_neighbours: bool = False
    # When True, permute the per-day macro_gate_input across days.
    # Tests whether the GraphSourceGate uses real macro regime info.
    shuffle_macro_gate: bool = False


def set_seeds(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cs_mse_loss(y_hat, y_true, mask):
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
        if a.std() < 1e-9 or b.std() < 1e-9:
            continue
        ics[t] = float(np.corrcoef(a, b)[0, 1])
    if np.all(np.isnan(ics)):
        return 0.0, ics
    return float(np.nanmean(ics)), ics


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


def cohort_ic(y_hat, y, eval_mask, age_days):
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
            if a.std() < 1e-9 or b.std() < 1e-9:
                continue
            ics.append(float(np.corrcoef(a, b)[0, 1]))
        out[label] = float(np.mean(ics)) if ics else 0.0
    return out


def standardize_features(x, mask, train_idx):
    flat_train_mask = mask[train_idx]; x_train = x[train_idx]
    out = np.zeros_like(x)
    for f in range(x.shape[2]):
        vals = x_train[..., f][flat_train_mask]
        if vals.size < 2:
            mu, sd = 0.0, 1.0
        else:
            mu = float(np.mean(vals)); sd = float(np.std(vals))
            if sd < 1e-6: sd = 1.0
        out[..., f] = (x[..., f] - mu) / sd
    out = out * mask[..., None]
    return out


def warmup_cosine_lr(step, warmup, total):
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))


def build_age_features_from_tradable(tradable_mask, hist20, hist60):
    """[T, N, 8] age tensor with cumsum-age semantics."""
    age = compute_age_from_tradable(tradable_mask).astype(np.float32)
    out = np.zeros((*tradable_mask.shape, 8), dtype=np.float32)
    out[..., 0] = age
    out[..., 1] = np.log1p(age)
    out[..., 2] = ((age >= 1) & (age <= 20)).astype(np.float32)
    out[..., 3] = ((age > 20) & (age <= 60)).astype(np.float32)
    out[..., 4] = ((age > 60) & (age <= 252)).astype(np.float32)
    out[..., 5] = (age > 252).astype(np.float32)
    out[..., 6] = hist20
    out[..., 7] = hist60
    return out


def build_ipo_keys(x_raw, age_feat, risk_arr, avg_corr_60d):
    """[T, N, 22] IPO retrieval key (same as OW v1)."""
    t_total, n, _ = x_raw.shape
    keys = np.zeros((t_total, n, len(IPO_ANALOGUE_KEY_COLS)), dtype=np.float32)
    keys[..., 0:8] = age_feat
    keys[..., 8] = x_raw[..., 14]
    keys[..., 9] = x_raw[..., 15]
    keys[..., 10] = x_raw[..., 18]
    keys[..., 11] = x_raw[..., 16]
    keys[..., 12] = x_raw[..., 9]
    keys[..., 13] = x_raw[..., 11]
    keys[..., 14] = x_raw[..., 13]
    keys[..., 15] = x_raw[..., 5]
    keys[..., 16] = x_raw[..., 6]
    keys[..., 17] = x_raw[..., 1]
    keys[..., 18] = x_raw[..., 2]
    vix = risk_arr[:, 0]
    vix_mu = float(np.nanmean(vix)); vix_sd = float(np.nanstd(vix))
    if vix_sd < 1e-6: vix_sd = 1.0
    vix_z = ((vix - vix_mu) / vix_sd).astype(np.float32)
    keys[..., 19] = np.broadcast_to(vix_z[:, None], (t_total, n))
    keys[..., 20] = np.broadcast_to(risk_arr[:, 4][:, None], (t_total, n))
    keys[..., 21] = np.broadcast_to(avg_corr_60d[:, None], (t_total, n))
    return keys


# Per-active-ticker feature column indices from x_raw (panel) used by
# DurationExposureEncoder. Spec Section C input list mapped to panel
# 22-feature schema:
#     0:log_return  1:log_return_5d  2:log_return_20d  3:log_volume
#     4:log_volume_ratio_20d  5:rv_20d  6:rv_60d  7:hl_range
#     8:close_to_high  9:st_volume_24h  10:st_volume_change_30d
#     11:st_bullish_ratio  12:st_sentiment_dispersion  13:st_labeled_ratio
#     14:log_market_cap  15:cash_runway_q  16:rd_intensity
#     17:revenue_growth_yoy  18:cash_to_mc  19:shares_outstanding_yoy
#     20:total_assets_growth  21:has_fundamentals
DURATION_PANEL_COL_IDX = [
    14, 15, 16, 17, 18, 19, 20, 21,   # fundamentals (8)
    5, 6, 4, 7,                        # risk/liquidity (4)
    9, 10, 11, 12, 13,                 # social (5)
]
# Followed by 8 age features and 10 rolling-beta features assembled at
# runtime. Total: 17 + 8 + 10 = 35.
DURATION_INPUT_DIM = len(DURATION_PANEL_COL_IDX) + 8 + len(ROLLING_BETA_COLS)


def main(cfg_path: str, fold: int, seed: int, smoke: bool = False) -> None:
    """Train DOW-epiSTAR v2 on one (fold, seed) pair."""
    with open(cfg_path) as f:
        raw = yaml.safe_load(f)

    train_kwargs = dict(raw.get("train", {}))
    # Pop ablation flags that are not part of TrainConfig dataclass.
    disable_day_retrieval_flag = train_kwargs.pop("disable_day_retrieval", False)
    disable_ipo_retrieval_flag = train_kwargs.pop("disable_ipo_retrieval", False)
    train_cfg = TrainConfig(**{**train_kwargs, "fold": fold, "seed": seed})
    backbone_cfg = STARBackboneConfig(**raw.get("backbone", {}))
    day_mem_cfg = EpisodeMemoryConfig(**raw.get("day_memory", {}))
    ipo_mem_kwargs = dict(raw.get("ipo_memory", {}))
    if train_cfg.random_ipo_retrieval:
        ipo_mem_kwargs["random_retrieval"] = True
    ipo_mem_cfg = IPOMemoryConfig(**ipo_mem_kwargs)
    duration_cfg = DurationExposureConfig(
        input_dim=DURATION_INPUT_DIM,
        hidden_dim=raw.get("model", {}).get("duration_hidden_dim", 64),
        out_dim=raw.get("model", {}).get("duration_dim", 32),
        dropout=raw.get("model", {}).get("duration_dropout", 0.1),
    )
    macro_cfg = MacroStateConfig(
        input_dim=len(MACRO_FEATURE_COLS_FULL),
        hidden_dim=raw.get("model", {}).get("macro_hidden_dim", 64),
        out_dim=raw.get("model", {}).get("macro_dim", 32),
        gate_state_dim=raw.get("model", {}).get("macro_gate_state_dim", 16),
        dropout=raw.get("model", {}).get("macro_dropout", 0.1),
    )
    ow_cfg = OWEpiSTARV1Config(
        backbone=backbone_cfg, day_memory=day_mem_cfg, ipo_memory=ipo_mem_cfg,
        episode_value_dim=raw.get("model", {}).get("episode_value_dim", 32),
        ipo_value_dim=raw.get("model", {}).get("ipo_value_dim", 32),
        cross_attn_heads=raw.get("model", {}).get("cross_attn_heads", 4),
        gate_hidden_dim=raw.get("model", {}).get("gate_hidden_dim", 64),
        head_hidden_dim=raw.get("model", {}).get("head_hidden_dim", 64),
        head_dropout=raw.get("model", {}).get("head_dropout", 0.1),
        disable_day_retrieval=disable_day_retrieval_flag,
        disable_ipo_retrieval=disable_ipo_retrieval_flag,
    )
    rate_mem_kwargs = dict(raw.get("rate_memory", {}))
    if train_cfg.random_rate_retrieval:
        rate_mem_kwargs["random_retrieval"] = True
    rate_mem_cfg = RateShockMemoryConfig(**rate_mem_kwargs)
    dow_cfg = DOWEpiSTARConfig(
        ow=ow_cfg, duration=duration_cfg, macro=macro_cfg,
        rate_memory=rate_mem_cfg,
        macro_gate_input_dim=len(MACRO_GATE_COLS),
        rate_gate_input_dim=5,   # see RateShock gate input below
        head_hidden_dim=raw.get("model", {}).get("head_hidden_dim", 64),
        head_dropout=raw.get("model", {}).get("head_dropout", 0.1),
        cross_attn_heads=raw.get("model", {}).get("cross_attn_heads", 4),
        use_macro_duration_head=not train_cfg.disable_macro_duration,
        use_rate_memory=train_cfg.use_rate_memory,
        use_duration_graph=train_cfg.use_duration_graph,
        disable_lambda_macro=train_cfg.disable_lambda_macro,
        disable_score_duration=train_cfg.disable_score_duration,
        disable_alpha_rate=train_cfg.disable_alpha_rate,
        disable_score_rate=train_cfg.disable_score_rate,
    )
    if smoke:
        train_cfg.epochs = 2

    set_seeds(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DOW-epiSTAR-v2] fold={fold} seed={seed} device={device}")

    panel_cfg = EnrichedPanelConfig(
        start_date=pd.Timestamp(train_cfg.panel_start),
        end_date=pd.Timestamp(train_cfg.panel_end),
        horizon_days=train_cfg.horizon_days,
        universe_csv=Path(train_cfg.universe_csv),
    )
    panel, tickers, dates = build_enriched_panel(panel_cfg)
    tens = panel_to_tensors(panel, tickers, dates)
    x_raw = tens["x"]; y = tens["y"]
    print(f"[DOW-epiSTAR-v2] panel: T={x_raw.shape[0]} N={x_raw.shape[1]} F={x_raw.shape[2]}")
    if x_raw.shape[1] < 50:
        raise RuntimeError("Panel too small")

    mm = build_minimal_masks(
        dates, tickers, MinimalMaskConfig(horizon_days=train_cfg.horizon_days)
    )
    tradable = mm["tradable_mask"]; loss_mask = mm["loss_mask"]
    hist20 = mm["history_valid_20d"]; hist60 = mm["history_valid_60d"]
    print(f"[DOW-epiSTAR-v2] tradable_cells={int(tradable.sum())} loss_cells={int(loss_mask.sum())}")

    train_idx, val_idx, test_idx = fold_indices(fold, dates)
    print(f"[DOW-epiSTAR-v2] fold {fold}: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    x = standardize_features(x_raw, tradable, train_idx).astype(np.float32)

    age_feat = build_age_features_from_tradable(tradable, hist20, hist60)
    age_days = age_feat[..., 0].astype(np.int64)
    if train_cfg.shuffle_age_features:
        rng = np.random.default_rng(seed)
        for fi in range(age_feat.shape[-1]):
            for t in range(age_feat.shape[0]):
                idx_a = np.where(tradable[t])[0]
                if idx_a.size > 1:
                    perm = rng.permutation(idx_a)
                    vals = age_feat[t, idx_a, fi].copy()
                    age_feat[t, perm, fi] = vals
        print("[DOW-epiSTAR-v2] age features SHUFFLED (sanity ablation)")

    # Day-level keys + values (same as OW v1).
    day_keys, _ = build_episode_keys(
        dates=dates, log_returns=x_raw[..., 0], mask=tradable, cfg=EpisodeKeyConfig(),
    )
    feature_idx = [0, 1, 5, 6]
    n_summary = 2 * len(feature_idx) + 1
    day_values = np.zeros((len(dates), day_keys.shape[1] + n_summary), dtype=np.float32)
    day_values[:, : day_keys.shape[1]] = day_keys
    for t in range(len(dates)):
        m = tradable[t]
        if m.sum() < 5: continue
        for j, fi in enumerate(feature_idx):
            v = x_raw[t, m, fi]
            day_values[t, day_keys.shape[1] + 2 * j] = float(np.mean(v))
            day_values[t, day_keys.shape[1] + 2 * j + 1] = float(np.std(v))
        day_values[t, -1] = float(m.sum()) / 250.0

    # IPO keys + values.
    risk_arr = pd.read_parquet("data/processed/risk_features.parquet").reindex(
        pd.to_datetime(dates)
    ).ffill().bfill().to_numpy(dtype=np.float32)
    avg_corr_60d = day_keys[:, EPISODE_KEY_COLS.index("cs_avg_pairwise_corr_60d")]
    ipo_keys = build_ipo_keys(x_raw, age_feat, risk_arr, avg_corr_60d)
    ipo_value_extras = np.zeros((len(dates), x_raw.shape[1], 1), dtype=np.float32)
    ipo_value_extras[..., 0] = y
    ipo_values = np.concatenate([ipo_keys, ipo_value_extras], axis=-1)

    ipo_eligible = tradable & mm["label_mask"] & (age_days >= 1) & (age_days <= ipo_mem_cfg.max_age_days)
    train_set = set(int(t) for t in train_idx.tolist())
    flat_keys: list = []; flat_values: list = []
    flat_days: list = []; flat_tickers: list = []
    for t in range(len(dates)):
        if t not in train_set: continue
        for i in np.where(ipo_eligible[t])[0]:
            flat_keys.append(ipo_keys[t, i]); flat_values.append(ipo_values[t, i])
            flat_days.append(t); flat_tickers.append(int(i))
    flat_keys = np.asarray(flat_keys, dtype=np.float32) if flat_keys else np.zeros((0, ipo_keys.shape[-1]), dtype=np.float32)
    flat_values = np.asarray(flat_values, dtype=np.float32) if flat_values else np.zeros((0, ipo_values.shape[-1]), dtype=np.float32)
    flat_days = np.asarray(flat_days, dtype=np.int64) if len(flat_days) > 0 else np.zeros((0,), dtype=np.int64)
    flat_tickers = np.asarray(flat_tickers, dtype=np.int64) if len(flat_tickers) > 0 else np.zeros((0,), dtype=np.int64)
    print(f"[DOW-epiSTAR-v2] IPO memory entries: {len(flat_keys)}")

    # DOW v2 macro features.
    macro_path = Path("data/processed/macro_duration_features.parquet")
    if not macro_path.exists():
        print("[DOW-epiSTAR-v2] macro parquet missing; building...")
        build_macro_duration_features()
    macro = pd.read_parquet(macro_path)
    macro_arr, macro_cols, macro_stats = standardize_macro_duration(macro, dates, train_idx)
    if train_cfg.shuffle_macro_state:
        rng2 = np.random.default_rng(seed + 1000)
        perm = rng2.permutation(macro_arr.shape[0])
        macro_arr = macro_arr[perm]
        print("[DOW-epiSTAR-v2] macro state SHUFFLED (sanity ablation)")
    print(f"[DOW-epiSTAR-v2] macro features: {len(macro_cols)} dims")

    # v2.3 patch B: 9-dim macro gate input. Add avg_pairwise_corr_60d and
    # cross_sectional_dispersion (read from episode keys, not the macro
    # parquet) to the 7 macro scalars.
    gate_indices = [macro_cols.index(c) for c in MACRO_GATE_COLS if c in macro_cols]
    if len(gate_indices) != len(MACRO_GATE_COLS):
        missing = [c for c in MACRO_GATE_COLS if c not in macro_cols]
        print(f"[DOW-epiSTAR-v2] WARN missing gate cols: {missing}")
    macro_gate_macro = macro_arr[:, gate_indices].astype(np.float32)
    avg_corr_idx_dk = EPISODE_KEY_COLS.index("cs_avg_pairwise_corr_60d")
    cs_disp_idx_dk = EPISODE_KEY_COLS.index("cs_dispersion")
    # Standardise these two with train-fold stats.
    avg_corr = day_keys[:, avg_corr_idx_dk].astype(np.float32)
    cs_disp = day_keys[:, cs_disp_idx_dk].astype(np.float32)
    avg_corr_train = avg_corr[train_idx]; cs_disp_train = cs_disp[train_idx]
    avg_corr_z = ((avg_corr - avg_corr_train.mean()) /
                  max(avg_corr_train.std(), 1e-6)).astype(np.float32)
    cs_disp_z = ((cs_disp - cs_disp_train.mean()) /
                 max(cs_disp_train.std(), 1e-6)).astype(np.float32)
    macro_gate_arr = np.concatenate(
        [macro_gate_macro, avg_corr_z[:, None], cs_disp_z[:, None]], axis=1
    ).astype(np.float32)
    print(f"[DOW-epiSTAR-v2] macro_gate input: {macro_gate_arr.shape[1]} dims "
          f"(v2.3 expanded from 7 to 9)")
    dow_cfg.macro_gate_input_dim = macro_gate_arr.shape[1]
    # macro_arr_for_graph_gate: copy of macro_arr used only by the
    # GraphSourceGate (via the MacroStateEncoder gate-state output).
    # When shuffle_macro_gate is True we permute days; the rest of the
    # forward path (including the MacroRateSensitivityHead) keeps the
    # unshuffled macro_arr so we can isolate the graph-gate's reliance
    # on macro regime info from the head's reliance.
    if train_cfg.shuffle_macro_gate:
        rng_mgs = np.random.default_rng(seed + 9001)
        macro_perm = rng_mgs.permutation(macro_arr.shape[0])
        macro_arr_for_graph_gate = macro_arr[macro_perm].copy()
        print("[DOW-epiSTAR-v2] macro input to GraphSourceGate SHUFFLED "
              "across days (graph-gate ablation control)")
    else:
        macro_arr_for_graph_gate = macro_arr

    # Rolling betas.
    betas_path = Path("data/processed/rolling_macro_betas.parquet")
    if not betas_path.exists():
        print("[DOW-epiSTAR-v2] rolling betas parquet missing; building...")
        build_rolling_betas()
    betas_long = pd.read_parquet(betas_path)
    betas_tensor = betas_to_tensor(betas_long, dates, tickers)
    # Standardise betas using train-fold cells only (per-feature).
    bt_train = betas_tensor[train_idx]
    train_mask = tradable[train_idx]
    betas_std = np.zeros_like(betas_tensor)
    for fi in range(betas_tensor.shape[-1]):
        vals = bt_train[..., fi][train_mask]
        if vals.size < 2:
            mu, sd = 0.0, 1.0
        else:
            mu = float(np.mean(vals)); sd = float(np.std(vals))
            if sd < 1e-6: sd = 1.0
        betas_std[..., fi] = (betas_tensor[..., fi] - mu) / sd
    betas_std = betas_std * tradable[..., None]
    print(f"[DOW-epiSTAR-v2] rolling betas: shape={betas_std.shape}")

    # Build duration input tensor [T, N, DURATION_INPUT_DIM].
    duration_input_full = np.concatenate(
        [x[..., DURATION_PANEL_COL_IDX], age_feat, betas_std], axis=-1
    ).astype(np.float32)
    if train_cfg.shuffle_duration_input:
        rng3 = np.random.default_rng(seed + 2000)
        for t in range(duration_input_full.shape[0]):
            idx_a = np.where(tradable[t])[0]
            if idx_a.size > 1:
                perm = rng3.permutation(idx_a)
                duration_input_full[t, perm] = duration_input_full[t, idx_a]
        print("[DOW-epiSTAR-v2] duration input SHUFFLED (sanity ablation)")
    print(f"[DOW-epiSTAR-v2] duration input dim: {duration_input_full.shape[-1]}")

    # v2.3 patch C: explicit 17-dim rate-shock key (15 macro + 2 cs).
    # The RATE_SHOCK_KEY_COLS list now spells out all 17 columns.
    rate_keys = np.zeros((len(dates), len(RATE_SHOCK_KEY_COLS)), dtype=np.float32)
    for j, col in enumerate(RATE_SHOCK_KEY_COLS):
        if col == "avg_pairwise_corr_60d":
            rate_keys[:, j] = avg_corr_z
        elif col == "cross_sectional_dispersion":
            rate_keys[:, j] = cs_disp_z
        elif col in macro_cols:
            rate_keys[:, j] = macro_arr[:, macro_cols.index(col)]
        else:
            print(f"[DOW-epiSTAR-v2] WARN rate-shock col '{col}' missing")
    assert rate_keys.shape[1] == len(RATE_SHOCK_KEY_COLS), (
        f"rate_keys dim {rate_keys.shape[1]} != "
        f"len(RATE_SHOCK_KEY_COLS)={len(RATE_SHOCK_KEY_COLS)}"
    )
    print(f"[DOW-epiSTAR-v2] rate-shock keys: {rate_keys.shape[1]} dims "
          f"(asserted == {len(RATE_SHOCK_KEY_COLS)})")

    # Rate values: rate_key + day-level cs summaries + duration
    # distribution summaries (v2.3 patch D).
    feat_for_summary = [0, 1, 5]   # log_return, log_return_5d, rv_20d
    rate_value_extras = np.zeros((len(dates), 2 * len(feat_for_summary) + 1), dtype=np.float32)
    for t in range(len(dates)):
        m = tradable[t]
        if m.sum() < 5: continue
        for j, fi in enumerate(feat_for_summary):
            v = x_raw[t, m, fi]
            rate_value_extras[t, 2 * j] = float(np.mean(v))
            rate_value_extras[t, 2 * j + 1] = float(np.std(v))
        rate_value_extras[t, -1] = float(m.sum()) / 250.0

    # v2.3 patch D: duration-distribution summaries are built AFTER
    # duration_input_full is constructed (a few sections below). We
    # assemble rate_values there with the correct duration-summary
    # columns concatenated.

    # Section F: 5-d rate-gate scalar tensor (top1+entropy are appended
    # inside the model). Spec scalars: |delta_10y_20d|, delta_hy_spread_20d,
    # xbi_ret_20d, xbi_rv_20d, avg_pairwise_corr_60d.
    rate_gate_indices_macro = [macro_cols.index(c) for c in
                                ["delta_10y_20d", "delta_hy_spread_20d",
                                 "xbi_ret_20d", "xbi_rv_20d"] if c in macro_cols]
    rate_gate_arr = np.concatenate([
        np.abs(macro_arr[:, [macro_cols.index("delta_10y_20d")]]
               if "delta_10y_20d" in macro_cols else np.zeros((macro_arr.shape[0], 1))),
        macro_arr[:, [macro_cols.index("delta_hy_spread_20d")]]
        if "delta_hy_spread_20d" in macro_cols else np.zeros((macro_arr.shape[0], 1)),
        macro_arr[:, [macro_cols.index("xbi_ret_20d")]]
        if "xbi_ret_20d" in macro_cols else np.zeros((macro_arr.shape[0], 1)),
        macro_arr[:, [macro_cols.index("xbi_rv_20d")]]
        if "xbi_rv_20d" in macro_cols else np.zeros((macro_arr.shape[0], 1)),
        day_keys[:, [avg_corr_idx_dk]].astype(np.float32),
    ], axis=1).astype(np.float32)
    print(f"[DOW-epiSTAR-v2] rate-gate scalars: {rate_gate_arr.shape[1]} dims")

    # v2.3 patch D: per-day duration-distribution summaries to append
    # to rate_values. These let rate retrieval match days with similar
    # rate-shock AND similar cross-sectional duration vulnerability.
    # We use ||duration_input|| as a duration_norm proxy (encoder is
    # not yet trained when memory is populated).
    dur_norm = np.linalg.norm(duration_input_full, axis=-1)        # [T, N]
    dur_summary = np.zeros(
        (len(dates), len(RATE_VALUE_DURATION_SUMMARY_COLS)), dtype=np.float32,
    )
    rate_beta_col = ROLLING_BETA_COLS.index("rolling_rate_beta_60d")
    credit_beta_col = ROLLING_BETA_COLS.index("rolling_credit_beta_60d")
    for t in range(len(dates)):
        m = tradable[t]
        if m.sum() < 5:
            continue
        dn = dur_norm[t, m]
        dur_summary[t, 0] = float(np.mean(dn))
        dur_summary[t, 1] = float(np.std(dn))
        sorted_dn = np.sort(dn)
        n_dec = max(1, len(sorted_dn) // 10)
        dur_summary[t, 2] = float(np.mean(sorted_dn[-n_dec:]))   # top decile
        dur_summary[t, 3] = float(np.mean(sorted_dn[:n_dec]))    # bottom decile
        rb = betas_std[t, m, rate_beta_col]
        cb = betas_std[t, m, credit_beta_col]
        dur_summary[t, 4] = float(np.mean(rb)); dur_summary[t, 5] = float(np.std(rb))
        dur_summary[t, 6] = float(np.mean(cb)); dur_summary[t, 7] = float(np.std(cb))
        cash_runway = x[t, m, 15]
        cash_to_mc = x[t, m, 18]
        rd_int = x[t, m, 16]
        dur_summary[t, 8] = float(np.mean(cash_runway))
        sorted_cr = np.sort(cash_runway)
        dur_summary[t, 9] = float(np.mean(sorted_cr[:n_dec]))
        dur_summary[t, 10] = float(np.mean(cash_to_mc))
        dur_summary[t, 11] = float(np.mean(rd_int))

    rate_values = np.concatenate(
        [rate_keys, rate_value_extras, dur_summary], axis=1,
    ).astype(np.float32)
    print(f"[DOW-epiSTAR-v2] rate-shock values: {rate_values.shape[1]} dims "
          f"(key {rate_keys.shape[1]} + cs {rate_value_extras.shape[1]} + "
          f"duration {dur_summary.shape[1]})")

    # v2.3 patch E: deterministic duration-graph features [T, N, 10].
    # Standardise each feature with train-fold cell statistics only.
    duration_graph_feats_raw = np.zeros(
        (len(dates), len(tickers), len(DURATION_GRAPH_FEATURE_COLS)), dtype=np.float32,
    )
    # Map columns to data sources:
    #   panel cols (use raw, then z-score below):
    #     cash_runway_q -> 15, cash_to_mc -> 18, rd_intensity -> 16
    #     log_market_cap -> 14, realized_vol_60d -> 6
    #   rolling_*_beta_60d -> betas_tensor (already standardised in betas_std)
    #   age_trading_days -> age_feat[..., 0]
    #   history_valid_ratio_60d -> age_feat[..., 7]
    panel_col_map = {
        "cash_runway_q": 15, "cash_to_mc": 18, "rd_intensity": 16,
        "log_market_cap": 14, "realized_vol_60d": 6,
    }
    for j, col in enumerate(DURATION_GRAPH_FEATURE_COLS):
        if col in panel_col_map:
            duration_graph_feats_raw[..., j] = x_raw[..., panel_col_map[col]]
        elif col in ROLLING_BETA_COLS:
            duration_graph_feats_raw[..., j] = betas_tensor[..., ROLLING_BETA_COLS.index(col)]
        elif col == "age_trading_days":
            duration_graph_feats_raw[..., j] = age_feat[..., 0]
        elif col == "history_valid_ratio_60d":
            duration_graph_feats_raw[..., j] = age_feat[..., 7]
        else:
            print(f"[DOW-epiSTAR-v2] WARN duration-graph col '{col}' missing")
    duration_graph_feats = standardize_features(
        duration_graph_feats_raw, tradable, train_idx,
    ).astype(np.float32)
    print(f"[DOW-epiSTAR-v2] deterministic duration-graph features: "
          f"{duration_graph_feats.shape[-1]} dims (v2.3 patch E)")

    # Model.
    ow_cfg.episode_value_dim = day_values.shape[1]
    ow_cfg.ipo_value_dim = ipo_values.shape[-1]
    dow_cfg.ow = ow_cfg
    dow_cfg.rate_key_dim = rate_keys.shape[1]
    dow_cfg.rate_value_dim = rate_values.shape[1]
    model = DOWEpiSTAR(
        dow_cfg, day_key_dim=day_keys.shape[1], ipo_key_dim=ipo_keys.shape[-1],
    ).to(device)
    if dow_cfg.use_rate_memory:
        model.rate_memory.populate(
            keys=rate_keys, values=rate_values,
            day_indices=np.arange(len(dates)), train_day_indices=train_idx,
        )
        model.rate_memory.to(device)
    model.ow.day_memory.populate(
        keys=day_keys, values=day_values,
        day_indices=np.arange(len(dates)), train_day_indices=train_idx,
    )
    model.ow.day_memory.to(device)
    model.ow.ipo_memory.populate(
        keys=flat_keys, values=flat_values,
        day_indices=flat_days, ticker_indices=flat_tickers,
        train_day_indices=train_idx,
    )
    model.ow.ipo_memory.to(device)

    allowed_train = torch.from_numpy(train_idx).long().to(device)
    optim = AdamW(model.parameters(), lr=train_cfg.learning_rate,
                  weight_decay=train_cfg.weight_decay)
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
        corr_window=cw, tau=train_cfg.correlation_shrinkage_tau,
        top_k=backbone_cfg.num_neighbors,
        min_overlap_absolute=train_cfg.min_overlap_absolute,
    )

    def _shrunk_corr_full(t: int, cfg_use: SurvivorshipGraphConfig) -> np.ndarray:
        """Compute full [N, N] reliability-shrunk correlation for day t.

        Mirrors shrunk_correlation_neighbors_one_day but returns the
        score matrix instead of top-K.
        """
        ww = cfg_use.corr_window
        win_returns = x_raw[..., 0][t - ww + 1 : t + 1]
        win_mask = tradable[t - ww + 1 : t + 1]
        nan_filled = np.where(win_mask, np.where(np.isnan(win_returns), 0.0, win_returns), 0.0)
        valid_count = win_mask.sum(axis=0).astype(np.float32)
        x_w = nan_filled - nan_filled.mean(axis=0, keepdims=True)
        sd = x_w.std(axis=0, keepdims=True)
        sd = np.where(sd < 1e-8, 1e-8, sd)
        x_norm = x_w / sd
        rho = (x_norm.T @ x_norm) / float(ww)
        overlap = np.minimum(valid_count[:, None], valid_count[None, :])
        if cfg_use.tau > 0:
            shrinkage = overlap / (overlap + cfg_use.tau)
        else:
            shrinkage = np.ones_like(overlap)
        rho_shrunk = rho * shrinkage
        rho_shrunk[overlap < cfg_use.min_overlap_absolute] = -np.inf
        np.fill_diagonal(rho_shrunk, -np.inf)
        rho_shrunk[~tradable[t], :] = -np.inf
        rho_shrunk[:, ~tradable[t]] = -np.inf
        return rho_shrunk.astype(np.float32)

    def forward_one_day(t_idx: int) -> dict:
        if t_idx < max(w, cw):
            return {}
        active_mask_t = torch.from_numpy(tradable[t_idx]).to(device)
        if active_mask_t.sum() < 5:
            return {}
        active_idx = active_mask_t.nonzero(as_tuple=False).squeeze(-1)
        active_idx_np = active_idx.detach().cpu().numpy()

        cfg_use = SurvivorshipGraphConfig(
            corr_window=cw,
            tau=0.0 if train_cfg.disable_correlation_shrinkage else train_cfg.correlation_shrinkage_tau,
            top_k=backbone_cfg.num_neighbors,
            min_overlap_absolute=train_cfg.min_overlap_absolute,
        )

        # Section G: graph-blending ablation modes.
        gmode = train_cfg.graph_mode
        if gmode == "no_graph":
            # No spatial neighbours: each ticker uses only its own
            # temporal window. We pass an all-(-1) neighbour matrix so
            # patch construction emits self-only patches.
            n_panel = active_mask_t.shape[0]
            top_neighbors_day_t = torch.full(
                (n_panel, backbone_cfg.num_neighbors), -1,
                dtype=torch.long, device=device,
            )
        elif gmode == "random_graph":
            # Random top-K active neighbours per active ticker (deterministic
            # per (t, seed) via numpy.random with seed offset).
            rng_t = np.random.default_rng(train_cfg.seed * 1000003 + int(t_idx))
            n_panel = active_mask_t.shape[0]
            top = np.full((n_panel, backbone_cfg.num_neighbors), -1, dtype=np.int64)
            active_pool = active_idx_np
            if active_pool.size >= 2:
                k_eff = min(backbone_cfg.num_neighbors, active_pool.size - 1)
                for src_pos, src in enumerate(active_pool):
                    others = active_pool[active_pool != src]
                    if others.size == 0:
                        continue
                    top[src, :k_eff] = rng_t.choice(others, size=k_eff, replace=False)
            top_neighbors_day_t = torch.from_numpy(top).to(device)
        elif gmode in ("corr_only", "rate_only", "fixed_blend", "learned_blend"):
            # Compute A_corr and (if needed) A_rate, then blend.
            with torch.no_grad():
                m_in = torch.from_numpy(macro_arr_for_graph_gate[t_idx]).float().to(device)
                _, m_gate_state = model.compute_macro(m_in)

                # A_corr is always available.
                a_corr_np = _shrunk_corr_full(int(t_idx), cfg_use)
                a_corr_t = torch.from_numpy(a_corr_np).to(device)

                # A_rate (rate-sensitivity): only needed if graph_mode
                # uses it. For learned_blend we honour the gate's weights.
                need_rate = gmode in ("rate_only", "fixed_blend", "learned_blend")
                if need_rate:
                    if train_cfg.duration_graph_source == "deterministic_features":
                        dgraph_feats = torch.from_numpy(
                            duration_graph_feats[t_idx, active_idx_np]
                        ).float().to(device)
                        a_rate_t = build_duration_similarity(
                            dgraph_feats, active_mask_t,
                        )
                    else:
                        # learned_encoder needs the duration encoder warm.
                        if t_idx < train_cfg.duration_graph_min_warmup_days:
                            need_rate = False
                            a_rate_t = torch.zeros_like(a_corr_t)
                        else:
                            dur_in_full = torch.from_numpy(
                                duration_input_full[t_idx, active_idx_np]
                            ).float().to(device)
                            d_exp_full = model.compute_duration_exposure(dur_in_full)
                            a_rate_t = build_duration_similarity(
                                d_exp_full.float(), active_mask_t,
                            )
                    if train_cfg.shuffle_rate_neighbours and a_rate_t.numel() > 0:
                        rng_s = np.random.default_rng(
                            train_cfg.seed * 7919 + int(t_idx) + 31337,
                        )
                        idx_a = active_idx_np
                        if idx_a.size > 1:
                            perm = rng_s.permutation(idx_a)
                            a_rate_np = a_rate_t.detach().cpu().numpy()
                            a_rate_np[idx_a, :] = a_rate_np[perm, :]
                            a_rate_np[:, idx_a] = a_rate_np[:, perm]
                            a_rate_t = torch.from_numpy(a_rate_np).to(device)
                else:
                    a_rate_t = torch.zeros_like(a_corr_t)

                # Determine blend weights.
                if gmode == "corr_only":
                    w_corr, w_rate = 1.0, 0.0
                elif gmode == "rate_only":
                    w_corr, w_rate = 0.0, 1.0
                elif gmode == "fixed_blend":
                    w_corr = train_cfg.fixed_blend_w_corr
                    w_rate = train_cfg.fixed_blend_w_rate
                else:  # learned_blend
                    graph_w = model.compute_graph_weights(m_gate_state)
                    w_corr = float(graph_w[0].item())
                    w_rate = float(graph_w[1].item())

                neigh_t = merge_corr_and_duration(
                    a_corr_t, a_rate_t, w_corr, w_rate,
                    top_k=backbone_cfg.num_neighbors,
                    active_mask=active_mask_t,
                )
            top_neighbors_day_t = neigh_t
        else:
            # Fallback: legacy behaviour (use_duration_graph flag).
            use_dgraph = (train_cfg.use_duration_graph
                          and dow_cfg.use_duration_graph)
            if use_dgraph and train_cfg.duration_graph_source == "learned_encoder":
                use_dgraph = use_dgraph and t_idx >= train_cfg.duration_graph_min_warmup_days
            if use_dgraph:
                with torch.no_grad():
                    m_in = torch.from_numpy(macro_arr_for_graph_gate[t_idx]).float().to(device)
                    _, m_gate_state = model.compute_macro(m_in)
                    graph_w = model.compute_graph_weights(m_gate_state)
                    w_corr = float(graph_w[0].item())
                    w_duration = float(graph_w[1].item())
                    a_corr_np = _shrunk_corr_full(int(t_idx), cfg_use)
                    a_corr_t = torch.from_numpy(a_corr_np).to(device)
                    if train_cfg.duration_graph_source == "deterministic_features":
                        dgraph_feats = torch.from_numpy(
                            duration_graph_feats[t_idx, active_idx_np]
                        ).float().to(device)
                        a_dur_t = build_duration_similarity(
                            dgraph_feats, active_mask_t,
                        )
                    else:
                        dur_in_full = torch.from_numpy(
                            duration_input_full[t_idx, active_idx_np]
                        ).float().to(device)
                        d_exp_full = model.compute_duration_exposure(dur_in_full)
                        a_dur_t = build_duration_similarity(
                            d_exp_full.float(), active_mask_t,
                        )
                    neigh_t = merge_corr_and_duration(
                        a_corr_t, a_dur_t, w_corr, w_duration,
                        top_k=backbone_cfg.num_neighbors,
                        active_mask=active_mask_t,
                    )
                top_neighbors_day_t = neigh_t
            else:
                neigh, _ = shrunk_correlation_neighbors_one_day(
                    x_raw[..., 0], tradable, int(t_idx), cfg_use,
                )
                top_neighbors_day_t = torch.from_numpy(neigh).to(device)
        x_window = torch.from_numpy(x[t_idx - w + 1 : t_idx + 1]).to(device)
        mask_window = torch.from_numpy(tradable[t_idx - w + 1 : t_idx + 1]).to(device)
        patches, patch_mask = build_patches(
            x_window=x_window, mask_window=mask_window,
            top_neighbors=top_neighbors_day_t, active_idx=active_idx,
        )
        day_query_key = torch.from_numpy(day_keys[t_idx]).float().to(device)
        regime_scalars = model.ow.day_memory.standardize_query(day_query_key)[[0, 9]].clone()
        if torch.isnan(regime_scalars).any():
            regime_scalars = torch.zeros(2, device=device)
        ipo_query_keys = torch.from_numpy(ipo_keys[t_idx, active_idx_np]).float().to(device)
        gate_age = age_feat[t_idx, active_idx_np][:, [0, 1, 6, 7]]
        has_fund = x_raw[t_idx, active_idx_np, 21:22]
        st_lab = x_raw[t_idx, active_idx_np, 13:14]
        rv20 = x_raw[t_idx, active_idx_np, 5:6]
        gate_feats_np = np.concatenate([gate_age, has_fund, st_lab, rv20], axis=-1)
        ipo_gate_feats = torch.from_numpy(gate_feats_np).float().to(device)

        # DOW v2 specific tensors.
        dur_in = torch.from_numpy(duration_input_full[t_idx, active_idx_np]).float().to(device)
        macro_in = torch.from_numpy(macro_arr[t_idx]).float().to(device)
        macro_gate_in = torch.from_numpy(macro_gate_arr[t_idx]).float().to(device)
        # Section F: rate-shock query key + gate scalars.
        rate_q_key = torch.from_numpy(rate_keys[t_idx]).float().to(device)
        rate_gate_in = torch.from_numpy(rate_gate_arr[t_idx]).float().to(device)

        with autocast(enabled=use_amp, dtype=torch.float16):
            out = model.forward_day(
                patches=patches, patch_mask=patch_mask, active_mask=active_mask_t,
                day_query_key=day_query_key, ipo_query_keys=ipo_query_keys,
                ipo_gate_features=ipo_gate_feats,
                query_day_idx=int(t_idx), allowed_day_indices=allowed_train,
                gate_regime_scalars=regime_scalars,
                duration_input=dur_in, macro_input=macro_in,
                macro_gate_input=macro_gate_in,
                rate_query_key=rate_q_key, rate_gate_input=rate_gate_in,
            )
        out["y_hat"] = out["y_hat"].float()
        out["active_mask"] = active_mask_t
        out["t_idx"] = t_idx
        out["active_idx"] = active_idx
        return out

    @torch.no_grad()
    def evaluate(idx, eval_mask_arr):
        model.eval()
        T = x.shape[0]; N = x.shape[1]
        y_hat_all = np.zeros((T, N), dtype=np.float32)
        score_idio_all = np.zeros((T, N), dtype=np.float32)
        score_dur_all = np.zeros((T, N), dtype=np.float32)
        score_rate_all = np.zeros((T, N), dtype=np.float32)
        lambda_log: list[float] = []
        alpha_rate_log: list[float] = []
        emask = np.zeros((T, N), dtype=bool)
        for t_idx in idx:
            out = forward_one_day(int(t_idx))
            if not out:
                continue
            y_hat_all[t_idx] = out["y_hat"].detach().cpu().numpy()
            if "score_idio" in out:
                score_idio_all[t_idx] = out["score_idio"].detach().cpu().numpy()
                score_dur_all[t_idx] = out["score_duration"].detach().cpu().numpy()
                score_rate_all[t_idx] = out.get(
                    "score_rate", torch.zeros_like(out["score_idio"])
                ).detach().cpu().numpy()
                lm = out["lambda_macro"]
                if lm.dim() == 0:
                    lambda_log.append(float(lm.item()))
                ar = out.get("alpha_rate")
                if ar is not None and ar.dim() == 0:
                    alpha_rate_log.append(float(ar.item()))
            emask[t_idx] = eval_mask_arr[t_idx]
        ic, _ = per_day_ic(y_hat_all, y, emask, rank=False)
        rank_ic, _ = per_day_ic(y_hat_all, y, emask, rank=True)
        ndcg10 = ndcg_at_k(y_hat_all, y, emask, 10)
        ndcg50 = ndcg_at_k(y_hat_all, y, emask, 50)
        coh = cohort_ic(y_hat_all, y, emask, age_days)
        return {"ic": ic, "rank_ic": rank_ic, "ndcg10": ndcg10, "ndcg50": ndcg50,
                "cohort_ic": coh,
                "lambda_macro_mean": float(np.mean(lambda_log)) if lambda_log else 0.0,
                "alpha_rate_mean": float(np.mean(alpha_rate_log)) if alpha_rate_log else 0.0,
                "y_hat_all": y_hat_all,
                "score_idio_all": score_idio_all,
                "score_duration_all": score_dur_all,
                "score_rate_all": score_rate_all,
                "eval_mask": emask}

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
            "epoch": epoch, "train_loss": train_loss,
            "val_ic": val_metrics["ic"], "val_rank_ic": val_metrics["rank_ic"],
            "val_lambda_macro_mean": val_metrics["lambda_macro_mean"],
        })
        print(f"[DOW-epiSTAR-v2] epoch {epoch}: train_loss={train_loss:.4f} "
              f"val_ic={val_metrics['ic']:.4f} val_rank_ic={val_metrics['rank_ic']:.4f} "
              f"lambda_macro={val_metrics['lambda_macro_mean']:.3f}")
        if val_metrics["ic"] > best_val_ic + 1e-5:
            best_val_ic = val_metrics["ic"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= train_cfg.early_stop_patience:
                print(f"[DOW-epiSTAR-v2] early stop at epoch {epoch}")
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
        y_hat=test_metrics["y_hat_all"],
        score_idio=test_metrics["score_idio_all"],
        score_duration=test_metrics["score_duration_all"],
        score_rate=test_metrics["score_rate_all"],
        y_true=y,
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
    for tm in (test_metrics, val_metrics_final):
        for k in ("y_hat_all", "score_idio_all", "score_duration_all",
                  "score_rate_all", "eval_mask"):
            tm.pop(k, None)

    payload = {
        "fold": fold, "seed": seed, "model": "DOW-epiSTAR-v2.2",
        "panel_start": train_cfg.panel_start, "panel_end": train_cfg.panel_end,
        "n_tickers": int(x.shape[1]), "n_dates": int(x.shape[0]),
        "n_train": int(len(train_idx)), "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "n_ipo_memory_entries": int(len(flat_keys)),
        "ic": test_metrics["ic"], "rank_ic": test_metrics["rank_ic"],
        "ndcg10": test_metrics["ndcg10"], "ndcg50": test_metrics["ndcg50"],
        "test_cohort_ic": test_metrics["cohort_ic"],
        "test_lambda_macro_mean": test_metrics["lambda_macro_mean"],
        "test_alpha_rate_mean": test_metrics["alpha_rate_mean"],
        "val_ic": val_metrics_final["ic"], "val_rank_ic": val_metrics_final["rank_ic"],
        "val_cohort_ic": val_metrics_final["cohort_ic"],
        "best_val_ic": best_val_ic, "history": history,
        "config": {
            "train": asdict(train_cfg),
            "backbone": asdict(backbone_cfg),
            "day_memory": asdict(day_mem_cfg),
            "ipo_memory": asdict(ipo_mem_cfg),
            "rate_memory": asdict(rate_mem_cfg),
            "duration": asdict(duration_cfg),
            "macro": asdict(macro_cfg),
            "macro_cols": macro_cols,
            "macro_gate_cols": MACRO_GATE_COLS,
            "rate_shock_key_cols": RATE_SHOCK_KEY_COLS,
            "duration_input_dim": int(DURATION_INPUT_DIM),
            "use_rate_memory": bool(dow_cfg.use_rate_memory),
            "use_duration_graph": bool(dow_cfg.use_duration_graph),
        },
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"[DOW-epiSTAR-v2] wrote {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/dow_epistar_v2.yaml")
    p.add_argument("--fold", type=int, choices=[1, 2, 3], required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.config, args.fold, args.seed, smoke=args.smoke)
