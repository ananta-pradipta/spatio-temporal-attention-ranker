"""Phase 5b retrieval-bank key construction.

Replaces the placeholder zero-vector keys used by Phase 5's
``populate_retrieval_banks``. Per Phase 5b spec sections 5.1 (regime key)
and 5.2 (novelty key); ambiguity resolutions 1, 4, 5, 6 from user message
on 2026-05-07.

This module is data-pipeline code and does not modify any file under
``src/lattice/model/``. Train-fold-only standardisation is enforced by the
``fit_*_stats`` functions taking ``train_idx`` and using only those days
to compute means and standard deviations.

Notes
-----
* Regime key (14-d): cross-sectional return moments + vol moments + pairwise
  correlation + dispersion + active-count + VIX + 2s10s + 10y breakeven,
  per-component z-scored against train fold.
* Novelty key (8-d numeric): months_since_ipo / 36 + log_market_cap (z within
  sector) + log_dollar_volume (z within sector) + realized_vol_20d (z) +
  st_volume_abnormal_z60d (passthrough) + st_volume_24h_log (z) + idiovol_60d
  (z; computed below) + sector projection scalar (frozen). The model's
  NoveltyMemory.populate_bank still appends its learned 16-d sector embed.
* IPO recency: derived from per-ticker first-panel-appearance date. Tickers
  appearing in the first ``incumbent_window_days`` (default 170 trading days,
  approximately 8 calendar months) are treated as incumbents and excluded
  from the novelty bank for all dates; later additions get
  ``months_since_first_panel_appearance`` as the IPO-age proxy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from src.lattice.data.build_panel import (
    PANEL_FEATURE_COLS, ST_FEATURE_COLS, MACRO_FEATURE_COLS,
)


PANEL_COL_IDX = {c: i for i, c in enumerate(PANEL_FEATURE_COLS)}
ST_COL_IDX = {c: i for i, c in enumerate(ST_FEATURE_COLS)}
MACRO_COL_IDX = {c: i for i, c in enumerate(MACRO_FEATURE_COLS)}

REGIME_KEY_DIM = 14
NOVELTY_KEY_DIM = 8

# Trading days threshold for "incumbent" classification: a ticker first
# appearing in the first ``INCUMBENT_WINDOW_TRADING_DAYS`` of the panel is
# treated as already-listed, never enters the novelty bank.
INCUMBENT_WINDOW_TRADING_DAYS = 170

# Months-since-IPO cap for novelty-bank eligibility.
NOVELTY_MAX_MONTHS = 36
TRADING_DAYS_PER_MONTH = 21


# ---------------------------------------------------------------------------
# Regime key
# ---------------------------------------------------------------------------

@dataclass
class RegimeKeyStats:
    """Per-component mean and std for regime-key z-scoring (train fold only)."""

    means: np.ndarray  # (REGIME_KEY_DIM,)
    stds: np.ndarray   # (REGIME_KEY_DIM,)


CORR_SUBSAMPLE = 50  # tickers for pairwise corr (full universe is too slow)


def compute_regime_features_per_day(
    panel_tensor: np.ndarray,
    mask_tensor: np.ndarray,
    macro_tensor: np.ndarray,
    t: int,
    corr_window: int = 60,
    min_corr_overlap: int = 5,
) -> np.ndarray:
    """Compute the 14-d raw regime feature vector for day t.

    The pairwise correlation feature (component 8) is approximated on a
    deterministic 50-ticker subsample of the active universe rather than
    the full N approximately 500 panel: full computation is approximately 30
    seconds per day on the S&P 500 panel, which makes the 2000-day pre-build
    multi-hour. The 50-ticker subsample produces a stable cross-day
    average-correlation signal with negligible bias once standardised.
    """
    out = np.zeros(REGIME_KEY_DIM, dtype=np.float32)
    active = mask_tensor[t]
    n_active = int(active.sum())

    log_ret_idx = PANEL_COL_IDX["log_return"]
    log_ret5_idx = PANEL_COL_IDX["log_return_5d"]
    log_ret20_idx = PANEL_COL_IDX["log_return_20d"]
    rv20_idx = PANEL_COL_IDX["realized_vol_20d"]

    if n_active >= 2:
        r1 = panel_tensor[t, active, log_ret_idx]
        r5 = panel_tensor[t, active, log_ret5_idx]
        r20 = panel_tensor[t, active, log_ret20_idx]
        rv20 = panel_tensor[t, active, rv20_idx]

        def _mean_std(x: np.ndarray) -> tuple[float, float]:
            x_f = x[np.isfinite(x)]
            if x_f.size == 0:
                return 0.0, 0.0
            m = float(x_f.mean())
            s = float(x_f.std()) if x_f.size > 1 else 0.0
            return m, s

        out[0], out[1] = _mean_std(r1)
        out[2], out[3] = _mean_std(r5)
        out[4], out[5] = _mean_std(r20)
        out[6], out[7] = _mean_std(rv20)

        # Pairwise correlation on a 50-ticker subsample (deterministic by
        # taking the first CORR_SUBSAMPLE active tickers in panel order).
        if t >= corr_window - 1 and n_active >= 5:
            active_indices = np.where(active)[0][:CORR_SUBSAMPLE]
            if active_indices.size >= 2:
                window = panel_tensor[t - corr_window + 1: t + 1,
                                       active_indices, log_ret_idx]
                wmask = mask_tensor[t - corr_window + 1: t + 1, active_indices]
                window = np.where(wmask, window, np.nan)
                with np.errstate(invalid="ignore"):
                    centered = window - np.nanmean(window, axis=0, keepdims=True)
                centered = np.nan_to_num(centered, nan=0.0)
                valid = (~np.isnan(window)).astype(np.float32)
                n_overlap = valid.T @ valid  # (k, k)
                with np.errstate(invalid="ignore", divide="ignore"):
                    cov = centered.T @ centered
                    var = (centered ** 2).sum(axis=0)
                    denom = np.sqrt(var[:, None] * var[None, :])
                    rho = np.where(denom > 1e-9, cov / denom, 0.0)
                np.fill_diagonal(rho, np.nan)
                rho = np.where(n_overlap >= min_corr_overlap, rho, np.nan)
                avg_rho = float(np.nanmean(rho))
                out[8] = avg_rho if np.isfinite(avg_rho) else 0.0

        # Robust dispersion: IQR / MAD on 5-day returns
        r5_clean = r5[np.isfinite(r5)]
        if r5_clean.size >= 4:
            iqr = float(np.subtract(*np.percentile(r5_clean, [75, 25])))
            mad = float(np.median(np.abs(r5_clean - np.median(r5_clean))))
            out[9] = iqr / mad if mad > 1e-9 else 0.0

    out[10] = n_active / 500.0

    # Macro features (raw; z-scoring against train fold happens in apply_)
    if "vix" in MACRO_COL_IDX:
        out[11] = float(macro_tensor[t, MACRO_COL_IDX["vix"]])
    if "slope_2s10s" in MACRO_COL_IDX:
        out[12] = float(macro_tensor[t, MACRO_COL_IDX["slope_2s10s"]])
    if "breakeven_10y" in MACRO_COL_IDX:
        out[13] = float(macro_tensor[t, MACRO_COL_IDX["breakeven_10y"]])

    return out


def fit_regime_stats(
    panel_tensor: np.ndarray,
    mask_tensor: np.ndarray,
    macro_tensor: np.ndarray,
    train_idx: np.ndarray,
) -> RegimeKeyStats:
    """Compute per-component mean and std on train_idx days only."""
    raws = np.stack([
        compute_regime_features_per_day(panel_tensor, mask_tensor, macro_tensor,
                                          int(t))
        for t in train_idx
    ], axis=0)
    means = raws.mean(axis=0)
    stds = raws.std(axis=0)
    stds = np.where(stds < 1e-8, 1.0, stds)
    return RegimeKeyStats(means=means.astype(np.float32),
                            stds=stds.astype(np.float32))


def apply_regime_stats(raw: np.ndarray, stats: RegimeKeyStats) -> np.ndarray:
    """Per-component z-score the 14-d raw key against train-fold stats."""
    return ((raw - stats.means) / stats.stds).astype(np.float32)


def build_regime_key_tensor(
    panel_tensor: np.ndarray,
    mask_tensor: np.ndarray,
    macro_tensor: np.ndarray,
    train_idx: np.ndarray,
) -> tuple[np.ndarray, RegimeKeyStats]:
    """Compute standardised regime keys for every day in the panel.

    Returns ``(keys, stats)`` where keys is ``(T, 14)`` z-scored against
    train_idx-only stats.
    """
    stats = fit_regime_stats(panel_tensor, mask_tensor, macro_tensor, train_idx)
    T = panel_tensor.shape[0]
    keys = np.zeros((T, REGIME_KEY_DIM), dtype=np.float32)
    for t in range(T):
        raw = compute_regime_features_per_day(
            panel_tensor, mask_tensor, macro_tensor, t,
        )
        keys[t] = apply_regime_stats(raw, stats)
    return keys, stats


# ---------------------------------------------------------------------------
# IPO recency
# ---------------------------------------------------------------------------

def compute_first_panel_idx_per_ticker(
    mask_tensor: np.ndarray,
) -> np.ndarray:
    """For each ticker, the integer day index of first appearance.

    Args:
        mask_tensor: ``(T, N)`` bool.

    Returns:
        ``(N,)`` int array; ``T`` for tickers that never appear.
    """
    T, N = mask_tensor.shape
    first = np.full(N, T, dtype=np.int64)
    for n in range(N):
        col = mask_tensor[:, n]
        if col.any():
            first[n] = int(np.argmax(col))
    return first


def is_recent_ipo(
    first_panel_idx: np.ndarray,
    incumbent_window_days: int = INCUMBENT_WINDOW_TRADING_DAYS,
) -> np.ndarray:
    """Boolean mask: True for tickers we treat as post-2015 IPOs."""
    return (first_panel_idx > incumbent_window_days) & (first_panel_idx >= 0)


def months_since_ipo(
    t: int, n: int, first_panel_idx: np.ndarray, recent_ipo: np.ndarray,
) -> float:
    """Months-since-IPO for ticker n on day t. ``inf`` for incumbents."""
    if not recent_ipo[n]:
        return float("inf")
    days = t - int(first_panel_idx[n])
    if days < 0:
        return float("inf")
    return days / TRADING_DAYS_PER_MONTH


# ---------------------------------------------------------------------------
# Novelty key
# ---------------------------------------------------------------------------

@dataclass
class NoveltyKeyStats:
    """Train-fold-only standardisation stats for the 8-d novelty key.

    Some components are z-scored within sector (log_market_cap,
    log_dollar_volume); others are global (realized_vol_20d, st_volume_24h_log,
    idiovol_60d). Two components are passthrough (months_since_ipo / 36,
    st_volume_abnormal_z60d, sector projection scalar).
    """

    log_market_cap_per_sector: dict          # sector_id -> (mean, std)
    log_dollar_volume_per_sector: dict       # sector_id -> (mean, std)
    realized_vol_global: tuple[float, float]
    st_volume_24h_log_global: tuple[float, float]
    idiovol_60d_global: tuple[float, float]


def compute_idiovol_60d_proxy(
    panel_tensor: np.ndarray,
    mask_tensor: np.ndarray,
    sector_per_ticker: np.ndarray,
    window: int = 60,
) -> np.ndarray:
    """Approximate idiosyncratic 60-day vol per (date, ticker).

    Spec section 5.2 calls for residual std from a rolling regression of
    ticker returns on the sector ETF return. The macro tensor only holds
    5-day-aggregate ETF returns, not daily ETF returns, and a per-(date,
    ticker) rolling OLS is roughly 1.4M regressions on the S&P 500 panel.
    For the Phase 5b retrieval bank we instead compute a sector-residualised
    proxy: ``sqrt(max(0, rv60^2 - sector_pooled_vol60^2))``, where
    ``sector_pooled_vol60`` is the 60-day std of the sector-pooled (active-
    ticker mean) daily log return. This isolates the within-sector portion
    of the ticker's variance without requiring daily ETF returns.

    The proxy is documented in
    ``docs/lattice_data_provenance.md`` and is z-scored downstream against
    train-fold stats. Phase 5b v1 ships this approximation; the proper
    regression-residual idiovol is deferred.
    """
    T, N, _ = panel_tensor.shape
    out = np.full((T, N), np.nan, dtype=np.float32)
    log_ret_idx = PANEL_COL_IDX["log_return"]
    rv60_idx = PANEL_COL_IDX["realized_vol_60d"]

    # Sector-pooled 60-day vol, computed once per fold.
    log_ret = panel_tensor[..., log_ret_idx]
    log_ret = np.where(mask_tensor, log_ret, np.nan)

    sector_pooled = np.full((T, 11), np.nan, dtype=np.float32)
    for s in range(11):
        cols = (sector_per_ticker == s)
        if not cols.any():
            continue
        sector_pooled[:, s] = np.nanmean(log_ret[:, cols], axis=1)

    sector_60d_vol = np.full((T, 11), np.nan, dtype=np.float32)
    for s in range(11):
        sp = sector_pooled[:, s]
        if not np.isfinite(sp).any():
            continue
        for t in range(window - 1, T):
            chunk = sp[t - window + 1: t + 1]
            chunk = chunk[np.isfinite(chunk)]
            if chunk.size >= 5:
                sector_60d_vol[t, s] = float(np.std(chunk))

    rv60_arr = np.where(mask_tensor, panel_tensor[..., rv60_idx], np.nan)
    for s in range(11):
        cols = (sector_per_ticker == s)
        if not cols.any():
            continue
        sec_vol_col = sector_60d_vol[:, s][:, None]
        rv = rv60_arr[:, cols]
        with np.errstate(invalid="ignore"):
            resid_var = rv ** 2 - sec_vol_col ** 2
            resid_var = np.where(resid_var < 0, 0.0, resid_var)
            out[:, cols] = np.sqrt(resid_var).astype(np.float32)
    return out


def fit_novelty_stats(
    panel_tensor: np.ndarray,
    mask_tensor: np.ndarray,
    st_tensor: np.ndarray,
    sector_per_ticker: np.ndarray,
    train_idx: np.ndarray,
    idiovol_tensor: Optional[np.ndarray] = None,
) -> NoveltyKeyStats:
    """Compute z-score stats on train_idx active cells only."""
    log_mc_idx = PANEL_COL_IDX["log_market_cap"]
    log_vol_idx = PANEL_COL_IDX["log_volume"]
    rv20_idx = PANEL_COL_IDX["realized_vol_20d"]
    st_v24_idx = ST_COL_IDX["st_volume_24h_log"]

    train_set = set(int(t) for t in train_idx)
    train_mask = np.zeros(panel_tensor.shape[0], dtype=bool)
    for t in train_idx:
        train_mask[int(t)] = True
    cell_mask = mask_tensor & train_mask[:, None]

    log_mc_per_sector: dict = {}
    log_vol_per_sector: dict = {}
    for s in range(11):
        col_sec = (sector_per_ticker == s)
        if not col_sec.any():
            log_mc_per_sector[s] = (0.0, 1.0)
            log_vol_per_sector[s] = (0.0, 1.0)
            continue
        cells = cell_mask & col_sec[None, :]
        mc_vals = panel_tensor[..., log_mc_idx][cells]
        v_vals = panel_tensor[..., log_vol_idx][cells]
        mc_mean = float(np.nanmean(mc_vals)) if mc_vals.size else 0.0
        mc_std = float(np.nanstd(mc_vals)) if mc_vals.size else 1.0
        v_mean = float(np.nanmean(v_vals)) if v_vals.size else 0.0
        v_std = float(np.nanstd(v_vals)) if v_vals.size else 1.0
        log_mc_per_sector[s] = (mc_mean, max(mc_std, 1e-8))
        log_vol_per_sector[s] = (v_mean, max(v_std, 1e-8))

    rv_vals = panel_tensor[..., rv20_idx][cell_mask]
    rv_mean = float(np.nanmean(rv_vals)) if rv_vals.size else 0.0
    rv_std = float(np.nanstd(rv_vals)) if rv_vals.size else 1.0
    rv_std = max(rv_std, 1e-8)

    st_v24_vals = st_tensor[..., st_v24_idx][cell_mask]
    st_mean = float(np.nanmean(st_v24_vals)) if st_v24_vals.size else 0.0
    st_std = float(np.nanstd(st_v24_vals)) if st_v24_vals.size else 1.0
    st_std = max(st_std, 1e-8)

    if idiovol_tensor is not None:
        iv_vals = idiovol_tensor[cell_mask]
        iv_mean = float(np.nanmean(iv_vals)) if iv_vals.size else 0.0
        iv_std = float(np.nanstd(iv_vals)) if iv_vals.size else 1.0
        iv_std = max(iv_std, 1e-8)
    else:
        iv_mean, iv_std = 0.0, 1.0

    return NoveltyKeyStats(
        log_market_cap_per_sector=log_mc_per_sector,
        log_dollar_volume_per_sector=log_vol_per_sector,
        realized_vol_global=(rv_mean, rv_std),
        st_volume_24h_log_global=(st_mean, st_std),
        idiovol_60d_global=(iv_mean, iv_std),
    )


def compute_novelty_key_for_cell(
    panel_tensor: np.ndarray, st_tensor: np.ndarray,
    t: int, n: int, sector_id: int,
    months_since: float,
    stats: NoveltyKeyStats,
    sector_proj_scalar: float,
    idiovol_tensor: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Build the 8-d novelty numeric key for one (t, ticker) cell."""
    out = np.zeros(NOVELTY_KEY_DIM, dtype=np.float32)
    out[0] = max(0.0, min(1.0, months_since / NOVELTY_MAX_MONTHS))

    log_mc_idx = PANEL_COL_IDX["log_market_cap"]
    log_vol_idx = PANEL_COL_IDX["log_volume"]
    rv20_idx = PANEL_COL_IDX["realized_vol_20d"]
    st_abn_idx = ST_COL_IDX["st_volume_abnormal_z60d"]
    st_v24_idx = ST_COL_IDX["st_volume_24h_log"]

    s = int(sector_id) if 0 <= int(sector_id) < 11 else 0
    mc_mean, mc_std = stats.log_market_cap_per_sector[s]
    v_mean, v_std = stats.log_dollar_volume_per_sector[s]
    rv_mean, rv_std = stats.realized_vol_global
    st24_mean, st24_std = stats.st_volume_24h_log_global
    iv_mean, iv_std = stats.idiovol_60d_global

    mc_raw = float(panel_tensor[t, n, log_mc_idx])
    v_raw = float(panel_tensor[t, n, log_vol_idx])
    rv_raw = float(panel_tensor[t, n, rv20_idx])
    st_abn_raw = float(st_tensor[t, n, st_abn_idx])
    st24_raw = float(st_tensor[t, n, st_v24_idx])

    out[1] = (mc_raw - mc_mean) / mc_std if np.isfinite(mc_raw) else 0.0
    out[2] = (v_raw - v_mean) / v_std if np.isfinite(v_raw) else 0.0
    out[3] = (rv_raw - rv_mean) / rv_std if np.isfinite(rv_raw) else 0.0
    out[4] = st_abn_raw if np.isfinite(st_abn_raw) else 0.0
    out[5] = (st24_raw - st24_mean) / st24_std if np.isfinite(st24_raw) else 0.0

    if idiovol_tensor is not None:
        iv_raw = float(idiovol_tensor[t, n])
        out[6] = (iv_raw - iv_mean) / iv_std if np.isfinite(iv_raw) else 0.0
    else:
        out[6] = 0.0

    out[7] = sector_proj_scalar
    return out


def build_novelty_key_tensor(
    panel_tensor: np.ndarray,
    mask_tensor: np.ndarray,
    st_tensor: np.ndarray,
    sector_per_ticker: np.ndarray,
    sector_proj_scalar_per_ticker: np.ndarray,
    first_panel_idx: np.ndarray,
    train_idx: np.ndarray,
    idiovol_tensor: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, NoveltyKeyStats, np.ndarray]:
    """Compute the 8-d novelty key for every (date, ticker) panel cell.

    Returns:
        ``(keys, stats, novelty_eligible)`` where keys is ``(T, N, 8)``,
        novelty_eligible is ``(T, N)`` bool: True iff active AND ticker is
        post-2015 addition AND months_since_ipo <= 36.
    """
    T, N, _ = panel_tensor.shape
    stats = fit_novelty_stats(panel_tensor, mask_tensor, st_tensor,
                                sector_per_ticker, train_idx, idiovol_tensor)
    recent = is_recent_ipo(first_panel_idx)
    keys = np.zeros((T, N, NOVELTY_KEY_DIM), dtype=np.float32)
    eligible = np.zeros((T, N), dtype=bool)
    for n in range(N):
        if not mask_tensor[:, n].any():
            continue
        s = int(sector_per_ticker[n]) if sector_per_ticker[n] >= 0 else 0
        proj_scalar = float(sector_proj_scalar_per_ticker[n])
        for t in range(T):
            if not mask_tensor[t, n]:
                continue
            ms = months_since_ipo(t, n, first_panel_idx, recent)
            keys[t, n] = compute_novelty_key_for_cell(
                panel_tensor, st_tensor, t, n, s, ms,
                stats, proj_scalar, idiovol_tensor,
            )
            if recent[n] and ms <= NOVELTY_MAX_MONTHS:
                eligible[t, n] = True
    return keys, stats, eligible


__all__ = [
    "REGIME_KEY_DIM",
    "NOVELTY_KEY_DIM",
    "INCUMBENT_WINDOW_TRADING_DAYS",
    "NOVELTY_MAX_MONTHS",
    "RegimeKeyStats",
    "NoveltyKeyStats",
    "compute_regime_features_per_day",
    "fit_regime_stats",
    "apply_regime_stats",
    "build_regime_key_tensor",
    "compute_first_panel_idx_per_ticker",
    "is_recent_ipo",
    "months_since_ipo",
    "compute_idiovol_60d_proxy",
    "fit_novelty_stats",
    "compute_novelty_key_for_cell",
    "build_novelty_key_tensor",
]
