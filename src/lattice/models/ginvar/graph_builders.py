"""G-InVAR graph builders.

Per spec section 2:

  A_corr  : reliability-shrunk rolling 60-day correlation graph.
  A_sector: point-in-time GICS sector / subsector graph.
  A_factor: cosine similarity over standardised factor exposures.
  A_social: cosine similarity over StockTwits feature vectors.
  A_beta  : optional rolling beta similarity (deferred to v2).

Phase 1 (this commit) ships the sector graph. Other graph builders are
stubbed and will be filled in in subsequent commits per the implementation
order in section 14.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch


SECTOR_TO_ID = {
    "Communication Services": 0, "Consumer Discretionary": 1,
    "Consumer Staples": 2, "Energy": 3, "Financials": 4,
    "Health Care": 5, "Industrials": 6, "Information Technology": 7,
    "Materials": 8, "Real Estate": 9, "Utilities": 10,
}


def build_sector_graph_per_day(
    constituents_path: Path,
    tickers: list[str],
    dates: list[pd.Timestamp],
) -> np.ndarray:
    """Build a per-day point-in-time GICS sector / subsector graph.

    Args:
        constituents_path : path to sp500_constituents_pit.parquet
        tickers           : list of N panel ticker strings (column order)
        dates             : list of T panel dates (date-only Timestamps)

    Returns:
        ``(T, N, N)`` float32 array. Edge weight = 1.0 for same subsector,
        0.5 for same sector only, 0.0 otherwise. Diagonal is zero.
        Inactive (date, ticker) cells imply zero edges (handled by
        ``active_mask`` at attention time).
    """
    df = pd.read_parquet(constituents_path)
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["end_date"] = pd.to_datetime(df["end_date"])
    ticker_to_idx = {t: i for i, t in enumerate(tickers)}
    T = len(dates)
    N = len(tickers)
    out = np.zeros((T, N, N), dtype=np.float32)

    # Per-ticker membership intervals.
    intervals: dict[int, list[tuple[pd.Timestamp, pd.Timestamp, str, str]]] = {}
    for _, row in df.iterrows():
        t = row["ticker"]
        if t not in ticker_to_idx:
            continue
        i = ticker_to_idx[t]
        intervals.setdefault(i, []).append((
            row["start_date"], row["end_date"],
            str(row.get("gics_sector", "") or ""),
            str(row.get("gics_subsector", "") or ""),
        ))

    def active_at(i: int, date: pd.Timestamp) -> tuple[bool, str, str]:
        for start, end, sec, subsec in intervals.get(i, []):
            if start <= date <= end:
                return True, sec, subsec
        return False, "", ""

    for d_idx, date in enumerate(dates):
        active_pairs: list[tuple[int, str, str]] = []
        for i in range(N):
            ok, sec, subsec = active_at(i, date)
            if ok:
                active_pairs.append((i, sec, subsec))
        # Edge weights
        for a in range(len(active_pairs)):
            i, sec_i, sub_i = active_pairs[a]
            for b in range(a + 1, len(active_pairs)):
                j, sec_j, sub_j = active_pairs[b]
                if sec_i == sec_j and sec_i:
                    if sub_i == sub_j and sub_i:
                        w = 1.0
                    else:
                        w = 0.5
                else:
                    w = 0.0
                if w > 0:
                    out[d_idx, i, j] = w
                    out[d_idx, j, i] = w
    return out


def row_normalise(A: np.ndarray, eps: float = 1.0e-8) -> np.ndarray:
    """Row-normalise an ``(*, N)`` adjacency matrix (last axis sums to 1)."""
    s = A.sum(axis=-1, keepdims=True)
    return A / (s + eps)


# ---------------------------------------------------------------------------
# Step 6: reliability-shrunk rolling correlation graph
# ---------------------------------------------------------------------------

def build_correlation_graph_per_day(
    log_returns: np.ndarray,
    mask: np.ndarray,
    window: int = 60,
    min_overlap: int = 20,
    tau: float = 30.0,
    keep_positive_only: bool = True,
) -> np.ndarray:
    """Reliability-shrunk rolling pairwise correlation graph.

    Args:
        log_returns : (T, N) float, NaN where the cell is inactive.
        mask        : (T, N) bool, True where the cell is active on day t.
        window      : trailing window length in trading days (default 60).
        min_overlap : minimum pair-overlap days for a valid correlation.
        tau         : reliability-shrinkage anchor; A = (overlap / (overlap + tau)) * corr.
        keep_positive_only : if True, clip negative correlations to 0.

    Returns:
        (T, N, N) float32. Diagonal zeroed; NaN replaced with 0; days
        with insufficient lookback (t < window - 1) emit zero matrices.
    """
    T, N = log_returns.shape
    out = np.zeros((T, N, N), dtype=np.float32)
    log_returns = np.where(mask, log_returns, np.nan)
    for t in range(window - 1, T):
        win = log_returns[t - window + 1: t + 1]            # (W, N)
        valid = (~np.isnan(win)).astype(np.float32)         # (W, N) 1.0 where present
        # Mean / std per column over the present cells.
        with np.errstate(invalid="ignore"):
            sums = np.where(np.isnan(win), 0.0, win)
            counts = valid.sum(axis=0).clip(min=1.0)
            means = sums.sum(axis=0) / counts
            centered = sums - means[None, :] * valid
            sq_sum = (centered ** 2).sum(axis=0)
            stds = np.sqrt(sq_sum / counts)
        valid_n = valid.T @ valid                           # (N, N) overlap counts
        cov = centered.T @ centered                          # (N, N)
        denom = stds[:, None] * stds[None, :] * valid_n
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.where(denom > 1e-9, cov / np.where(denom == 0, 1.0, denom), 0.0)
            corr = np.where(np.isfinite(corr), corr, 0.0)
        # Reliability shrinkage by overlap.
        shrink = valid_n / (valid_n + tau)
        A = shrink * corr
        if keep_positive_only:
            A = np.clip(A, a_min=0.0, a_max=None)
        np.fill_diagonal(A, 0.0)
        # Mask out inactive tickers on day t (no edges from inactive nodes).
        active_t = mask[t]
        A = A * active_t[:, None] * active_t[None, :]
        out[t] = A.astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# Step 7: factor-similarity graph
# ---------------------------------------------------------------------------

def build_factor_graph_per_day(
    factor_features: np.ndarray,
    mask: np.ndarray,
    keep_positive_only: bool = True,
    eps: float = 1.0e-9,
) -> np.ndarray:
    """Cosine similarity over standardised factor exposures, per day.

    Args:
        factor_features : (T, N, F_factor) float, NaN where missing.
        mask            : (T, N) bool active mask.
        keep_positive_only : clip negative cosine to 0.
        eps             : numerical stability.

    Per-day cross-sectional standardisation on the active subset, then
    cosine similarity between standardised ticker exposure vectors.

    Returns ``(T, N, N) float32`` with diagonal zeroed.
    """
    T, N, F = factor_features.shape
    out = np.zeros((T, N, N), dtype=np.float32)
    for t in range(T):
        active = mask[t]
        if active.sum() < 2:
            continue
        x = factor_features[t]                              # (N, F)
        x_active = x[active]
        # Standardise each feature on the active subset; impute NaN -> 0.
        with np.errstate(invalid="ignore"):
            mu = np.nanmean(x_active, axis=0)
            sd = np.nanstd(x_active, axis=0)
            sd = np.where(sd < 1e-8, 1.0, sd)
            xz = (x - mu[None, :]) / sd[None, :]
        xz = np.nan_to_num(xz, nan=0.0)
        # Zero out inactive rows so they don't contribute to similarities.
        xz = xz * active[:, None]
        norms = np.linalg.norm(xz, axis=1, keepdims=True).clip(min=eps)
        xn = xz / norms
        sim = xn @ xn.T
        if keep_positive_only:
            sim = np.clip(sim, a_min=0.0, a_max=None)
        np.fill_diagonal(sim, 0.0)
        sim = sim * active[:, None] * active[None, :]
        out[t] = sim.astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# Step 9: StockTwits attention/disagreement graph
# ---------------------------------------------------------------------------

def build_social_graph_per_day(
    st_features: np.ndarray,
    has_st: np.ndarray,
    mask: np.ndarray,
    keep_positive_only: bool = True,
    eps: float = 1.0e-9,
) -> np.ndarray:
    """Cosine similarity over StockTwits feature vectors.

    Args:
        st_features : (T, N, F_st) float, 0 where ticker has no ST data.
        has_st      : (T, N) bool, True where ticker has ST data on day t.
        mask        : (T, N) bool active mask.

    Returns ``(T, N, N) float32`` with edges only between
    has_st-AND-active pairs.
    """
    T, N, F = st_features.shape
    out = np.zeros((T, N, N), dtype=np.float32)
    for t in range(T):
        active_st = mask[t] & has_st[t]
        if active_st.sum() < 2:
            continue
        x = st_features[t]
        # Standardise on the active-ST subset.
        x_act = x[active_st]
        with np.errstate(invalid="ignore"):
            mu = np.nanmean(x_act, axis=0)
            sd = np.nanstd(x_act, axis=0)
            sd = np.where(sd < 1e-8, 1.0, sd)
            xz = (x - mu[None, :]) / sd[None, :]
        xz = np.nan_to_num(xz, nan=0.0)
        xz = xz * active_st[:, None]
        norms = np.linalg.norm(xz, axis=1, keepdims=True).clip(min=eps)
        xn = xz / norms
        sim = xn @ xn.T
        if keep_positive_only:
            sim = np.clip(sim, a_min=0.0, a_max=None)
        np.fill_diagonal(sim, 0.0)
        sim = sim * active_st[:, None] * active_st[None, :]
        out[t] = sim.astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# Step 8: regime-aware deterministic blend
# ---------------------------------------------------------------------------

def compute_stress_per_day(
    macro_tensor: np.ndarray,
    panel_log_returns: np.ndarray,
    mask: np.ndarray,
    macro_feature_names: list[str],
    train_idx: np.ndarray,
    panel_corr_window: int = 60,
) -> np.ndarray:
    """Per-day stress index (mean z-score of macro + panel stress signals).

    Per spec section 3:
      stress inputs (macro): vix, vix_term_slope, move_proxy, dgs10,
        slope_2s10s, slope_3m10y, hyg_5d_ret, tlt_5d_ret, spy_5d_ret,
        qqq_5d_ret, iwm_5d_ret, market_breadth_proxy.
      stress inputs (panel): avg pairwise correlation (60d),
        cross-sectional return dispersion, cross-sectional vol dispersion,
        active ticker count.

    Standardisation is fitted on ``train_idx`` only.
    """
    T, _ = macro_tensor.shape
    macro_inputs = [
        "vix", "vix_term_slope", "move_proxy", "dgs10",
        "slope_2s10s", "slope_3m10y",
        "hyg_5d_ret", "tlt_5d_ret",
        "spy_5d_ret", "qqq_5d_ret", "iwm_5d_ret",
        "market_breadth_proxy",
    ]
    sign = {
        "vix": +1, "vix_term_slope": +1, "move_proxy": +1,
        "dgs10": -1, "slope_2s10s": -1, "slope_3m10y": -1,
        "hyg_5d_ret": -1, "tlt_5d_ret": -1,
        "spy_5d_ret": -1, "qqq_5d_ret": -1, "iwm_5d_ret": -1,
        "market_breadth_proxy": -1,
    }
    macro_idx = {n: i for i, n in enumerate(macro_feature_names)}
    macro_signals = []
    for name in macro_inputs:
        if name not in macro_idx:
            continue
        col = macro_tensor[:, macro_idx[name]]
        train_col = col[train_idx]
        finite = train_col[np.isfinite(train_col)]
        if finite.size == 0:
            continue
        mu = float(finite.mean())
        sd = float(finite.std()) or 1.0
        z = (col - mu) / sd
        macro_signals.append(sign.get(name, +1) * z)
    macro_signals = np.stack(macro_signals, axis=0) if macro_signals else np.zeros((1, T))

    # Panel-derived stress: cross-sectional dispersion and active count.
    panel_signals = np.zeros((4, T), dtype=np.float32)
    for t in range(T):
        active = mask[t]
        if active.sum() < 2:
            continue
        ret = panel_log_returns[t, active]
        ret = ret[np.isfinite(ret)]
        if ret.size:
            panel_signals[0, t] = float(np.std(ret))            # cs return dispersion
        if t >= panel_corr_window - 1:
            win = panel_log_returns[t - panel_corr_window + 1: t + 1, active]
            win = np.where(np.isnan(win), 0.0, win)
            with np.errstate(invalid="ignore"):
                centered = win - win.mean(axis=0, keepdims=True)
                cov = centered.T @ centered
                stds = np.sqrt(np.diag(cov)).clip(min=1e-9)
                corr = cov / (stds[:, None] * stds[None, :])
            np.fill_diagonal(corr, np.nan)
            avg_corr = float(np.nanmean(corr))
            panel_signals[1, t] = avg_corr if np.isfinite(avg_corr) else 0.0
        panel_signals[2, t] = float(active.sum() / 500.0) * -1.0  # negative: more active = calmer
        # vol dispersion: std of per-ticker realized vol proxy within window
        if t >= 5:
            recent_vol = np.nanstd(
                panel_log_returns[t - 5: t + 1, active], axis=0,
            )
            recent_vol = recent_vol[np.isfinite(recent_vol)]
            if recent_vol.size:
                panel_signals[3, t] = float(np.std(recent_vol))

    # Z-score each panel signal against train_idx.
    for i in range(panel_signals.shape[0]):
        col = panel_signals[i]
        train_col = col[train_idx]
        finite = train_col[np.isfinite(train_col)]
        if finite.size == 0:
            continue
        mu = float(finite.mean())
        sd = float(finite.std()) or 1.0
        panel_signals[i] = (col - mu) / sd

    all_signals = np.concatenate([macro_signals, panel_signals], axis=0)
    stress = all_signals.mean(axis=0)
    return stress.astype(np.float32)


def regime_blend_weights(
    stress: np.ndarray,
    use_corr: bool, use_sector: bool, use_factor: bool, use_social: bool,
) -> dict[str, np.ndarray]:
    """Return deterministic per-day blend weights per spec section 3.

    Weights normalised to sum to 1 over the active graph sources.
    """
    T = stress.shape[0]
    sigm = 1.0 / (1.0 + np.exp(-stress))                    # sigmoid(stress)
    weights: dict[str, np.ndarray] = {}
    if use_corr:
        weights["corr"] = np.clip(0.35 - 0.10 * sigm, 0.20, 0.40).astype(np.float32)
    if use_sector:
        weights["sector"] = np.full(T, 0.20, dtype=np.float32)
    if use_factor:
        weights["factor"] = np.clip(0.25 + 0.15 * sigm, 0.25, 0.45).astype(np.float32)
    if use_social:
        weights["social"] = np.full(T, 0.10, dtype=np.float32)
    if not weights:
        return weights
    total = sum(w for w in weights.values())
    total = np.where(total < 1e-9, 1.0, total)
    return {k: w / total for k, w in weights.items()}


def blend_graphs(
    graphs: dict[str, np.ndarray], weights: dict[str, np.ndarray],
) -> np.ndarray:
    """Compute the blended graph A_graph[t] = sum_g w_g[t] * A_g[t]."""
    if not graphs:
        raise ValueError("blend_graphs requires at least one graph")
    out = None
    for name, A in graphs.items():
        if name not in weights:
            continue
        w_t = weights[name][:, None, None]                  # (T, 1, 1)
        contribution = (w_t * A).astype(np.float32)
        out = contribution if out is None else out + contribution
    return out


def build_beta_graph_per_day(*args, **kwargs):
    raise NotImplementedError("A_beta is optional; deferred to v2")


__all__ = [
    "SECTOR_TO_ID",
    "build_sector_graph_per_day",
    "build_correlation_graph_per_day",
    "build_factor_graph_per_day",
    "build_social_graph_per_day",
    "compute_stress_per_day",
    "regime_blend_weights",
    "blend_graphs",
    "row_normalise",
]
