"""Cross-sectional structure features for CSID v1.

Per spec Section 7. Builds the 4-d daily cs_struct vector consumed by
the CSID gate MLP:

    pc1_share_21d:        leading-eigenvalue share of the 21d return
                          correlation matrix over the active universe.
    avg_pairwise_corr_60d: already in episode_keys (cs_avg_pairwise_corr_60d).
    dispersion_5d:         5d trailing std of cs_dispersion (the daily
                          cross-sectional return std).
    market_return_5d:      5d log return of the broad biotech index XBI
                          (read from data/raw/xbi_close.csv).

All entries are z-scored using train-fold statistics only. NaNs (from
warmup periods) are forward-filled then zero-filled.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


CS_STRUCT_FEATURE_COLS = [
    "pc1_share_21d",
    "avg_pairwise_corr_60d",
    "dispersion_5d",
    "market_return_5d",
]


def compute_pc1_share_21d(
    log_returns: np.ndarray, tradable_mask: np.ndarray, window: int = 21,
    min_universe: int = 20,
) -> np.ndarray:
    """Per-day leading-eigenvalue share over the trailing window.

    Args:
        log_returns: [T, N] panel of daily log returns.
        tradable_mask: [T, N] bool active mask.
        window: trailing-window length (default 21).
        min_universe: minimum number of full-window-active tickers
            required to compute pc1; falls back to NaN otherwise.

    Returns:
        [T] float32 per-day pc1 share. Days with insufficient history
        or universe are NaN.
    """
    t_total, n = log_returns.shape
    out = np.full(t_total, np.nan, dtype=np.float32)
    for t in range(window - 1, t_total):
        win_returns = log_returns[t - window + 1 : t + 1]
        win_mask = tradable_mask[t - window + 1 : t + 1]
        full_active = win_mask.all(axis=0)
        if full_active.sum() < min_universe:
            continue
        x = win_returns[:, full_active]
        # Standardise per ticker over the window.
        mu = x.mean(axis=0, keepdims=True)
        sd = x.std(axis=0, keepdims=True)
        sd = np.where(sd < 1e-8, 1e-8, sd)
        x_z = (x - mu) / sd
        if x_z.shape[1] < min_universe:
            continue
        # Correlation = (1 / window) * x_z^T x_z.
        c = (x_z.T @ x_z) / float(window)
        try:
            eig = np.linalg.eigvalsh(c)
        except np.linalg.LinAlgError:
            continue
        eig = np.clip(eig, 0.0, None)
        total = float(eig.sum())
        if total < 1e-12:
            continue
        out[t] = float(eig.max() / total)
    return out


def build_cs_struct(
    log_returns: np.ndarray, tradable_mask: np.ndarray,
    avg_pairwise_corr_60d: np.ndarray,   # [T] from episode_keys
    cs_dispersion: np.ndarray,           # [T] from episode_keys
    xbi_close: pd.Series,                # XBI daily close series
    panel_dates: list[pd.Timestamp],
    train_idx: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """Build the 4-d cs_struct[T, 4] tensor (train-fold standardised)."""
    t_total = log_returns.shape[0]

    # 1. pc1_share_21d.
    pc1 = compute_pc1_share_21d(log_returns, tradable_mask, window=21)

    # 2. avg_pairwise_corr_60d (already a daily series).
    avg = np.asarray(avg_pairwise_corr_60d, dtype=np.float32).copy()

    # 3. dispersion_5d: 5-day trailing mean of cs_dispersion.
    s = pd.Series(cs_dispersion)
    disp5 = s.rolling(5, min_periods=1).mean().to_numpy(dtype=np.float32)

    # 4. market_return_5d: 5-day log return of XBI.
    panel_index = pd.DatetimeIndex(pd.to_datetime(panel_dates).normalize())
    xbi_aligned = xbi_close.reindex(panel_index).ffill(limit=5)
    mkt = np.log(xbi_aligned / xbi_aligned.shift(5)).to_numpy(dtype=np.float32)

    raw = np.stack([pc1, avg, disp5, mkt], axis=1).astype(np.float32)
    raw = np.where(np.isnan(raw), 0.0, raw)

    # Train-fold z-score per column.
    out = np.zeros_like(raw)
    train_mask = np.zeros(t_total, dtype=bool)
    train_mask[train_idx] = True
    for k in range(raw.shape[1]):
        train_vals = raw[train_mask, k]
        train_vals = train_vals[np.isfinite(train_vals)]
        if train_vals.size < 5:
            mu, sd = 0.0, 1.0
        else:
            mu = float(train_vals.mean())
            sd = float(train_vals.std())
            if sd < 1e-6:
                sd = 1.0
        out[:, k] = (raw[:, k] - mu) / sd
    return out, list(CS_STRUCT_FEATURE_COLS)


__all__ = [
    "CS_STRUCT_FEATURE_COLS",
    "compute_pc1_share_21d",
    "build_cs_struct",
]
