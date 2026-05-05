"""RCFB v1 trainer.

Regime-Conditional Factor Blending. Per spec
``docs/specs/rcfb_implementation_prompt.md``:

    DOW backbone (OW v1 + duration graph) -> score_dow per ticker
    RCFB layer:
        score_dow_z   = per-day z(score_dow)
        factor_combo  = 0.45*z(rev) + 0.25*z(soc) + 0.20*z(cash) + 0.10*z(lowvol)
                        (FIXED weights; rev=-log_return_5d,
                         soc=st_volume_change_30d, cash=cash_runway_q,
                         lowvol=-realized_vol_60d)
        g_t           = sigmoid(GateMLP([m_state[16], cs_struct[4]]))
        score_total   = (1 - g_t) * score_dow_z + g_t * factor_combo

DOW backbone is preserved unchanged (no CSID, no DurationExposureEncoder,
no rate memory). Only the post-rank-head stage is replaced.
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
from src.v2.data.cs_struct import CS_STRUCT_FEATURE_COLS, build_cs_struct
from src.v2.data.episode_keys import (
    EpisodeKeyConfig, EPISODE_KEY_COLS, build_episode_keys,
)
from src.v2.data.minimal_masks import (
    MinimalMaskConfig, build_minimal_masks, compute_age_from_tradable,
)
from src.v2.data.rolling_macro_betas import (
    ROLLING_BETA_COLS, betas_to_tensor, build_rolling_betas,
)
from src.v2.graph.duration_dynamic_edges import (
    DURATION_GRAPH_FEATURE_COLS, build_duration_similarity, merge_corr_and_duration,
)
from src.v2.graph.survivorship_dynamic_edges_v1 import (
    SurvivorshipGraphConfig, shrunk_correlation_neighbors_one_day,
)
from src.v2.model.rcfb import RCFB
from src.v2.model.episode_memory import EpisodeMemoryConfig
from src.v2.model.ipo_analogue_memory import IPOMemoryConfig, IPO_ANALOGUE_KEY_COLS
from src.v2.model.macro_state import MacroStateConfig, MacroStateEncoder
from src.v2.model.ow_epistar_v1 import OWEpiSTARV1, OWEpiSTARV1Config
from src.v2.model.star_backbone import STARBackboneConfig
from src.v2.training.folds import fold_indices


@dataclass
class TrainConfig:
    """Top-level RCFB v1 training hyperparameters."""

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
    output_dir: str = "results/rcfb_v1"
    use_duration_graph: bool = True
    fixed_w_corr: float = 0.7
    fixed_w_duration: float = 0.3
    random_ipo_retrieval: bool = False
    disable_correlation_shrinkage: bool = False


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
        if m.sum() < 5: continue
        a = y_hat[t, m]; b = y[t, m]
        if rank:
            a = pd.Series(a).rank().to_numpy()
            b = pd.Series(b).rank().to_numpy()
        if a.std() < 1e-9 or b.std() < 1e-9: continue
        ics[t] = float(np.corrcoef(a, b)[0, 1])
    if np.all(np.isnan(ics)):
        return 0.0, ics
    return float(np.nanmean(ics)), ics


def ndcg_at_k(y_hat, y, mask, k):
    out = []
    for t in range(y_hat.shape[0]):
        m = mask[t]
        if m.sum() < k + 1: continue
        scores = y_hat[t, m]; rels = y[t, m]
        rels_pos = rels - rels.min() + 1e-9
        order = np.argsort(-scores)[:k]
        gains = rels_pos[order]
        discounts = 1.0 / np.log2(np.arange(2, k + 2))
        dcg = float((gains * discounts).sum())
        ideal = np.sort(rels_pos)[::-1][:k]
        idcg = float((ideal * discounts).sum())
        if idcg < 1e-9: continue
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
            if m.sum() < 5: continue
            a = y_hat[t, m]; b = y[t, m]
            if a.std() < 1e-9 or b.std() < 1e-9: continue
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


def build_age_feature_tensor(tradable_mask, hist20, hist60):
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


def main(cfg_path: str, fold: int, seed: int, smoke: bool = False) -> None:
    """Train RCFB v1 on one (fold, seed) pair."""
    with open(cfg_path) as f:
        raw = yaml.safe_load(f)

    train_cfg = TrainConfig(**{**raw.get("train", {}), "fold": fold, "seed": seed})
    backbone_cfg = STARBackboneConfig(**raw.get("backbone", {}))
    day_mem_cfg = EpisodeMemoryConfig(**raw.get("day_memory", {}))
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
    )
    rcfb_cfg = raw.get("rcfb", {})
    macro_cfg = MacroStateConfig(
        input_dim=raw.get("macro", {}).get("input_dim", 28),
        hidden_dim=raw.get("macro", {}).get("hidden_dim", 64),
        out_dim=raw.get("macro", {}).get("out_dim", 32),
        gate_state_dim=rcfb_cfg.get("m_state_dim", 16),
        dropout=raw.get("macro", {}).get("dropout", 0.1),
    )
    if smoke:
        train_cfg.epochs = 2

    set_seeds(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[RCFB] fold={fold} seed={seed} device={device}")

    panel_cfg = EnrichedPanelConfig(
        start_date=pd.Timestamp(train_cfg.panel_start),
        end_date=pd.Timestamp(train_cfg.panel_end),
        horizon_days=train_cfg.horizon_days,
        universe_csv=Path(train_cfg.universe_csv),
    )
    panel, tickers, dates = build_enriched_panel(panel_cfg)
    tens = panel_to_tensors(panel, tickers, dates)
    x_raw = tens["x"]; y = tens["y"]
    print(f"[RCFB] panel: T={x_raw.shape[0]} N={x_raw.shape[1]} F={x_raw.shape[2]}")
    if x_raw.shape[1] < 50:
        raise RuntimeError("Panel too small")

    mm = build_minimal_masks(
        dates, tickers, MinimalMaskConfig(horizon_days=train_cfg.horizon_days)
    )
    tradable = mm["tradable_mask"]; loss_mask = mm["loss_mask"]
    hist20 = mm["history_valid_20d"]; hist60 = mm["history_valid_60d"]

    train_idx, val_idx, test_idx = fold_indices(fold, dates)
    print(f"[RCFB] fold {fold}: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    x = standardize_features(x_raw, tradable, train_idx).astype(np.float32)
    age_feat = build_age_feature_tensor(tradable, hist20, hist60)
    age_days = age_feat[..., 0].astype(np.int64)

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

    risk_arr = pd.read_parquet("data/processed/risk_features.parquet").reindex(
        pd.to_datetime(dates)
    ).ffill().bfill().to_numpy(dtype=np.float32)
    avg_corr_60d = day_keys[:, EPISODE_KEY_COLS.index("cs_avg_pairwise_corr_60d")]
    cs_disp = day_keys[:, EPISODE_KEY_COLS.index("cs_dispersion")]
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
    print(f"[RCFB] IPO memory entries: {len(flat_keys)}")

    # Duration graph features (10-d) for the regime-blended graph.
    rolling_betas_path = Path("data/processed/rolling_macro_betas.parquet")
    if not rolling_betas_path.exists():
        build_rolling_betas()
    betas_long = pd.read_parquet(rolling_betas_path)
    betas_tensor = betas_to_tensor(betas_long, dates, tickers)
    panel_col_map = {
        "cash_runway_q": 15, "cash_to_mc": 18, "rd_intensity": 16,
        "log_market_cap": 14, "realized_vol_60d": 6,
    }
    duration_feats_raw = np.zeros(
        (len(dates), len(tickers), len(DURATION_GRAPH_FEATURE_COLS)), dtype=np.float32,
    )
    for j, col in enumerate(DURATION_GRAPH_FEATURE_COLS):
        if col in panel_col_map:
            duration_feats_raw[..., j] = x_raw[..., panel_col_map[col]]
        elif col in ROLLING_BETA_COLS:
            duration_feats_raw[..., j] = betas_tensor[..., ROLLING_BETA_COLS.index(col)]
        elif col == "age_trading_days":
            duration_feats_raw[..., j] = age_feat[..., 0]
        elif col == "history_valid_ratio_60d":
            duration_feats_raw[..., j] = age_feat[..., 7]
    duration_feats = standardize_features(
        duration_feats_raw, tradable, train_idx,
    ).astype(np.float32)

    # Macro features for the MacroStateEncoder input + cs_struct (CSID).
    from src.v2.data.macro_duration_features import (
        MACRO_FEATURE_COLS_FULL, build_macro_duration_features, standardize_macro_duration,
    )
    macro_path = Path("data/processed/macro_duration_features.parquet")
    if not macro_path.exists():
        build_macro_duration_features()
    macro = pd.read_parquet(macro_path)
    macro_arr, macro_cols, _ = standardize_macro_duration(macro, dates, train_idx)
    if macro_cfg.input_dim != macro_arr.shape[1]:
        macro_cfg.input_dim = macro_arr.shape[1]

    # cs_struct (4-d): pc1_share_21d + avg_pairwise_corr_60d + dispersion_5d + market_return_5d.
    xbi_close_path = Path("data/raw/xbi_close.csv")
    xbi_df = pd.read_csv(xbi_close_path, parse_dates=["date"]).set_index("date")
    xbi_close = xbi_df["close"].astype(float)
    cs_struct, cs_struct_cols = build_cs_struct(
        log_returns=x_raw[..., 0], tradable_mask=tradable,
        avg_pairwise_corr_60d=avg_corr_60d, cs_dispersion=cs_disp,
        xbi_close=xbi_close, panel_dates=dates, train_idx=train_idx,
    )
    print(f"[RCFB] cs_struct dims: {cs_struct.shape[1]} cols={cs_struct_cols}")

    # Build models.
    ow_cfg.episode_value_dim = day_values.shape[1]
    ow_cfg.ipo_value_dim = ipo_values.shape[-1]
    ow_model = OWEpiSTARV1(
        ow_cfg, day_key_dim=day_keys.shape[1], ipo_key_dim=ipo_keys.shape[-1],
    ).to(device)
    ow_model.day_memory.populate(
        keys=day_keys, values=day_values,
        day_indices=np.arange(len(dates)), train_day_indices=train_idx,
    )
    ow_model.day_memory.to(device)
    ow_model.ipo_memory.populate(
        keys=flat_keys, values=flat_values,
        day_indices=flat_days, ticker_indices=flat_tickers,
        train_day_indices=train_idx,
    )
    ow_model.ipo_memory.to(device)

    # MacroStateEncoder (re-used from DOW v2.3 architecture; only the
    # 16-d gate_state output feeds RCFB's gate MLP).
    macro_encoder = MacroStateEncoder(macro_cfg).to(device)

    # RCFB layer.
    fw_dict = rcfb_cfg.get("factor_weights", {})
    factor_weights = (
        float(fw_dict.get("rev", 0.45)),
        float(fw_dict.get("soc", 0.25)),
        float(fw_dict.get("cash", 0.20)),
        float(fw_dict.get("lowvol", 0.10)),
    )
    rcfb = RCFB(
        cs_struct_dim=cs_struct.shape[1],
        m_state_dim=rcfb_cfg.get("m_state_dim", 16),
        gate_hidden=rcfb_cfg.get("gate_hidden", 32),
        gate_init_bias=rcfb_cfg.get("gate_init_bias", -3.0),
        factor_weights=factor_weights,
        eps=rcfb_cfg.get("eps", 1e-6),
    ).to(device)

    allowed_train = torch.from_numpy(train_idx).long().to(device)
    params = list(ow_model.parameters()) + list(macro_encoder.parameters()) + list(rcfb.parameters())
    optim = AdamW(params, lr=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay)
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

    def _shrunk_corr_full(t_idx: int, cfg_use: SurvivorshipGraphConfig) -> np.ndarray:
        ww = cfg_use.corr_window
        win_returns = x_raw[..., 0][t_idx - ww + 1 : t_idx + 1]
        win_mask = tradable[t_idx - ww + 1 : t_idx + 1]
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
        rho_shrunk[~tradable[t_idx], :] = -np.inf
        rho_shrunk[:, ~tradable[t_idx]] = -np.inf
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
        if train_cfg.use_duration_graph:
            a_corr_np = _shrunk_corr_full(int(t_idx), cfg_use)
            a_corr_t = torch.from_numpy(a_corr_np).to(device)
            dgraph_feats = torch.from_numpy(
                duration_feats[t_idx, active_idx_np]
            ).float().to(device)
            a_dur_t = build_duration_similarity(dgraph_feats, active_mask_t)
            top_neighbors_day_t = merge_corr_and_duration(
                a_corr_t, a_dur_t,
                train_cfg.fixed_w_corr, train_cfg.fixed_w_duration,
                top_k=backbone_cfg.num_neighbors, active_mask=active_mask_t,
            )
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
        regime_scalars = ow_model.day_memory.standardize_query(day_query_key)[[0, 9]].clone()
        if torch.isnan(regime_scalars).any():
            regime_scalars = torch.zeros(2, device=device)
        ipo_query_keys = torch.from_numpy(ipo_keys[t_idx, active_idx_np]).float().to(device)
        gate_age = age_feat[t_idx, active_idx_np][:, [0, 1, 6, 7]]
        has_fund = x_raw[t_idx, active_idx_np, 21:22]
        st_lab = x_raw[t_idx, active_idx_np, 13:14]
        rv20 = x_raw[t_idx, active_idx_np, 5:6]
        gate_feats_np = np.concatenate([gate_age, has_fund, st_lab, rv20], axis=-1)
        ipo_gate_feats = torch.from_numpy(gate_feats_np).float().to(device)

        macro_in = torch.from_numpy(macro_arr[t_idx]).float().to(device)
        cs_t = torch.from_numpy(cs_struct[t_idx]).float().to(device)

        # Per-(active-ticker) raw factor inputs from the panel.
        # Spec Section 6: panel cols are 1=log_return_5d, 5=realized_vol_60d
        # (we use 6=realized_vol_60d), 10=st_volume_change_30d, 15=cash_runway_q.
        rev_active = torch.from_numpy(
            -x_raw[t_idx, active_idx_np, 1]
        ).float().to(device)
        soc_active = torch.from_numpy(
            x_raw[t_idx, active_idx_np, 10]
        ).float().to(device)
        cash_active = torch.from_numpy(
            x_raw[t_idx, active_idx_np, 15]
        ).float().to(device)
        lowvol_active = torch.from_numpy(
            -x_raw[t_idx, active_idx_np, 6]
        ).float().to(device)

        with autocast(enabled=use_amp, dtype=torch.float16):
            # Compute m_state via macro encoder (16-d gate state).
            _m_main, m_gate_state = macro_encoder(macro_in)

            # DOW backbone unchanged: no CSID hook, no z_idio splice.
            ow_out = ow_model.forward_day(
                patches=patches, patch_mask=patch_mask, active_mask=active_mask_t,
                day_query_key=day_query_key, ipo_query_keys=ipo_query_keys,
                ipo_gate_features=ipo_gate_feats,
                query_day_idx=int(t_idx), allowed_day_indices=allowed_train,
                gate_regime_scalars=regime_scalars,
            )
            # OW v1's y_hat is shape [N] with zeros on inactive cells.
            score_dow_full = ow_out["y_hat"].float()
            score_dow_active = score_dow_full[active_idx]

            # Apply RCFB on the active cross-section.
            final_score_active, g_t, factor_combo_active = rcfb(
                score_dow=score_dow_active.float(),
                rev=rev_active, soc=soc_active,
                cash=cash_active, lowvol=lowvol_active,
                m_state=m_gate_state.float(), cs_struct=cs_t,
            )

            # Place final_score into the full [N] tensor for the loss.
            score_total_full = torch.zeros_like(score_dow_full)
            score_total_full[active_idx] = final_score_active.float()
            score_total_full = score_total_full * active_mask_t.float()

            # Adapt the OW out dict for downstream code.
            ow_out["y_hat"] = score_total_full
            ow_out["hook_g_rcfb"] = g_t.detach()   # legacy field reused for logging
            ow_out["hook_factor_combo_std"] = factor_combo_active.std().detach() if factor_combo_active.numel() > 0 else torch.zeros((), device=device)
        out = {
            "y_hat": ow_out["y_hat"].float(),
            "active_mask": active_mask_t,
            "t_idx": t_idx,
            "active_idx": active_idx,
            "g_rcfb": ow_out.get("hook_g_rcfb", torch.zeros((), device=device)),
            "factor_combo_std": ow_out.get("hook_factor_combo_std", torch.zeros((), device=device)),
        }
        return out

    @torch.no_grad()
    def evaluate(idx, eval_mask_arr):
        ow_model.eval(); macro_encoder.eval(); rcfb.eval()
        T = x.shape[0]; N = x.shape[1]
        y_hat_all = np.zeros((T, N), dtype=np.float32)
        emask = np.zeros((T, N), dtype=bool)
        alpha_log: list[float] = []
        v_norm_log: list[float] = []
        alpha_arr = np.zeros(T, dtype=np.float32)
        for t_idx in idx:
            out = forward_one_day(int(t_idx))
            if not out: continue
            y_hat_all[t_idx] = out["y_hat"].detach().cpu().numpy()
            emask[t_idx] = eval_mask_arr[t_idx]
            a_v = out["g_rcfb"]
            if isinstance(a_v, torch.Tensor) and a_v.dim() == 0:
                alpha_log.append(float(a_v.item()))
                alpha_arr[t_idx] = float(a_v.item())
            v_v = out["factor_combo_std"]
            if isinstance(v_v, torch.Tensor) and v_v.dim() == 0:
                v_norm_log.append(float(v_v.item()))
        ic, _ = per_day_ic(y_hat_all, y, emask, rank=False)
        rank_ic, _ = per_day_ic(y_hat_all, y, emask, rank=True)
        ndcg10 = ndcg_at_k(y_hat_all, y, emask, 10)
        ndcg50 = ndcg_at_k(y_hat_all, y, emask, 50)
        coh = cohort_ic(y_hat_all, y, emask, age_days)
        return {"ic": ic, "rank_ic": rank_ic, "ndcg10": ndcg10, "ndcg50": ndcg50,
                "cohort_ic": coh,
                "g_rcfb_mean": float(np.mean(alpha_log)) if alpha_log else 0.0,
                "g_rcfb_std": float(np.std(alpha_log)) if alpha_log else 0.0,
                "factor_combo_std_mean": float(np.mean(v_norm_log)) if v_norm_log else 0.0,
                "g_rcfb_arr": alpha_arr,
                "y_hat_all": y_hat_all,
                "eval_mask": emask}

    step = 0; smoke_step_cap = 80
    for epoch in range(train_cfg.epochs):
        ow_model.train(); macro_encoder.train(); rcfb.train()
        np.random.seed(seed + epoch)
        perm = np.random.permutation(train_idx)
        epoch_losses: list[float] = []
        for t_idx in perm:
            t_idx = int(t_idx)
            if t_idx < max(w, cw): continue
            out = forward_one_day(t_idx)
            if not out: continue
            loss_mask_t = torch.from_numpy(loss_mask[t_idx]).to(device)
            y_true_t = torch.from_numpy(y[t_idx]).to(device)
            loss = cs_mse_loss(out["y_hat"], y_true_t, loss_mask_t)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(params, train_cfg.grad_clip)
            scaler.step(optim); scaler.update(); scheduler.step()
            epoch_losses.append(float(loss.item()))
            step += 1
            if smoke and step >= smoke_step_cap: break

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        val_metrics = evaluate(val_idx, eval_mask_arr=loss_mask)
        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "val_ic": val_metrics["ic"], "val_rank_ic": val_metrics["rank_ic"],
            "val_g_rcfb_mean": val_metrics["g_rcfb_mean"],
            "val_factor_combo_std_mean": val_metrics["factor_combo_std_mean"],
        })
        print(f"[RCFB] epoch {epoch}: loss={train_loss:.4f} "
              f"val_ic={val_metrics['ic']:.4f} "
              f"alpha={val_metrics['g_rcfb_mean']:.3f} "
              f"v_norm={val_metrics['factor_combo_std_mean']:.3f}")
        if val_metrics["ic"] > best_val_ic + 1e-5:
            best_val_ic = val_metrics["ic"]
            best_state = {
                "ow": {k: v.detach().clone() for k, v in ow_model.state_dict().items()},
                "macro": {k: v.detach().clone() for k, v in macro_encoder.state_dict().items()},
                "rcfb": {k: v.detach().clone() for k, v in rcfb.state_dict().items()},
            }
            patience = 0
        else:
            patience += 1
            if patience >= train_cfg.early_stop_patience:
                print(f"[RCFB] early stop at epoch {epoch}")
                break

    if best_state is not None:
        ow_model.load_state_dict(best_state["ow"])
        macro_encoder.load_state_dict(best_state["macro"])
        rcfb.load_state_dict(best_state["rcfb"])

    test_metrics = evaluate(test_idx, eval_mask_arr=loss_mask)
    val_metrics_final = evaluate(val_idx, eval_mask_arr=loss_mask)

    out_dir = Path(train_cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / f"fold{fold}_seed{seed}_predictions.npz"
    np.savez_compressed(
        pred_path,
        y_hat=test_metrics["y_hat_all"], y_true=y,
        loss_mask=test_metrics["eval_mask"], tradable_mask=tradable,
        test_idx=np.asarray(test_idx, dtype=np.int64),
        tickers=np.asarray(tickers, dtype=str),
        dates=np.asarray([str(d) for d in dates], dtype=str),
        age_days=age_days.astype(np.int32),
        g_rcfb=test_metrics["g_rcfb_arr"],
        cs_struct=cs_struct,
    )
    if best_state is not None:
        torch.save(best_state, out_dir / f"fold{fold}_seed{seed}_ckpt.pt")
    out_path = out_dir / f"fold{fold}_seed{seed}.json"
    for tm in (test_metrics, val_metrics_final):
        for k in ("y_hat_all", "eval_mask", "g_rcfb_arr"):
            tm.pop(k, None)
    payload = {
        "fold": fold, "seed": seed, "model": "RCFB-v1",
        "panel_start": train_cfg.panel_start, "panel_end": train_cfg.panel_end,
        "n_tickers": int(x.shape[1]), "n_dates": int(x.shape[0]),
        "n_train": int(len(train_idx)), "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "n_ipo_memory_entries": int(len(flat_keys)),
        "ic": test_metrics["ic"], "rank_ic": test_metrics["rank_ic"],
        "ndcg10": test_metrics["ndcg10"], "ndcg50": test_metrics["ndcg50"],
        "test_cohort_ic": test_metrics["cohort_ic"],
        "test_g_rcfb_mean": test_metrics["g_rcfb_mean"],
        "test_g_rcfb_std": test_metrics["g_rcfb_std"],
        "test_factor_combo_std_mean": test_metrics["factor_combo_std_mean"],
        "val_ic": val_metrics_final["ic"], "val_rank_ic": val_metrics_final["rank_ic"],
        "val_cohort_ic": val_metrics_final["cohort_ic"],
        "best_val_ic": best_val_ic, "history": history,
        "config": {"train": asdict(train_cfg),
                   "backbone": asdict(backbone_cfg),
                   "day_memory": asdict(day_mem_cfg),
                   "ipo_memory": asdict(ipo_mem_cfg),
                   "rcfb": rcfb_cfg,
                   "macro": asdict(macro_cfg),
                   "cs_struct_cols": cs_struct_cols},
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"[RCFB] wrote {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/rcfb_v1.yaml")
    p.add_argument("--fold", type=int, choices=[1, 2, 3], required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.config, args.fold, args.seed, smoke=args.smoke)
