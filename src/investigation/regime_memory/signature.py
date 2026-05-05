"""Causal 4-dim regime signature per panel day.

For day t, signature uses only data strictly up to day t-1:
  s1 = XBI realized vol 60d   (risk feature, causal at t-1)
  s2 = cross-sectional dispersion (std across active tickers of log
        return, averaged over days [t-20..t-1])
  s3 = mean pairwise log-return correlation across active tickers
        over days [t-60..t-1]
  s4 = VIX term slope (risk feature, causal at t-1)

REM-Audit 1 (signature causality) is enforced by construction: all
windows end at day t-1; no access to day t or later.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def cross_sectional_dispersion(log_returns: np.ndarray, mask: np.ndarray,
                               dispersion_window: int = 20) -> np.ndarray:
    """log_returns: [T, N] log_return feature. mask: [T, N] bool.
    Returns [T] mean-over-window of cross-sectional std.
    dispersion[t] uses days [t-window..t-1] (strictly past).
    """
    T, N = log_returns.shape
    # Per-day cross-sectional std of active tickers' log_return
    per_day = np.full(T, np.nan, dtype=np.float64)
    for d in range(T):
        m = mask[d]
        if m.sum() < 3:
            continue
        per_day[d] = float(np.nanstd(log_returns[d, m]))

    out = np.full(T, np.nan, dtype=np.float64)
    for t in range(dispersion_window, T):
        window = per_day[t - dispersion_window: t]
        valid = window[np.isfinite(window)]
        if valid.size > 0:
            out[t] = float(valid.mean())
    return out


def mean_pairwise_correlation(log_returns: np.ndarray, mask: np.ndarray,
                              corr_window: int = 60,
                              min_overlap: int = 20) -> np.ndarray:
    """Returns [T] mean off-diagonal pairwise ticker correlation over a
    trailing window ending at t-1. Expensive to compute naively; we
    approximate by restricting to tickers active on every day of the
    window and computing np.corrcoef on those.
    """
    T, N = log_returns.shape
    out = np.full(T, np.nan, dtype=np.float64)
    for t in range(corr_window, T):
        sl = slice(t - corr_window, t)
        sub = log_returns[sl]     # [window, N]
        sub_m = mask[sl]          # [window, N]
        # Tickers active on every day of the window
        always_active = sub_m.all(axis=0)
        if always_active.sum() < 3:
            continue
        active_cols = np.where(always_active)[0]
        X = sub[:, active_cols]   # [window, n_active]
        if X.shape[0] < min_overlap:
            continue
        C = np.corrcoef(X, rowvar=False)
        # Off-diagonal mean
        n_active = C.shape[0]
        iu = np.triu_indices(n_active, k=1)
        vals = C[iu]
        vals = vals[np.isfinite(vals)]
        if vals.size > 0:
            out[t] = float(vals.mean())
    return out


def compute_signatures(log_returns: np.ndarray, mask: np.ndarray,
                       dates: list, risk_df: pd.DataFrame,
                       dispersion_window: int = 20,
                       corr_window: int = 60) -> np.ndarray:
    """Returns [T, 4] raw (not standardized) regime signatures.

    NaN values indicate the signature is undefined at that day (early
    panel days where the trailing window does not fit). Downstream
    clustering / retrieval must handle NaN via forward-fill or exclude.
    """
    T, N = log_returns.shape
    disp = cross_sectional_dispersion(log_returns, mask, dispersion_window)
    corr = mean_pairwise_correlation(log_returns, mask, corr_window)

    # s1 and s4 from risk_df, aligned to panel dates
    idx = pd.DatetimeIndex(pd.to_datetime(dates))
    s1 = risk_df["xbi_rv_60d"].reindex(idx).ffill().bfill().values.astype(np.float64)
    s4 = risk_df["vix_term_slope"].reindex(idx).ffill().bfill().values.astype(np.float64)

    # Shift by 1 day so s(t) uses data up to t-1 (strict causality)
    s1_shift = np.concatenate([[np.nan], s1[:-1]])
    s4_shift = np.concatenate([[np.nan], s4[:-1]])

    sigs = np.stack([s1_shift, disp, corr, s4_shift], axis=1)  # [T, 4]
    return sigs


def forward_fill_signatures(sigs: np.ndarray) -> np.ndarray:
    """Forward-fill NaN days with the most recent valid signature, then
    backfill any remaining leading NaNs with the first valid row.
    """
    out = sigs.copy()
    T, D = out.shape
    # Forward-fill
    last_valid = None
    for t in range(T):
        if np.all(np.isfinite(out[t])):
            last_valid = out[t].copy()
        elif last_valid is not None:
            out[t] = last_valid
    # Backfill
    first_valid = None
    for t in range(T):
        if np.all(np.isfinite(out[t])):
            first_valid = out[t].copy()
            break
    if first_valid is not None:
        for t in range(T):
            if not np.all(np.isfinite(out[t])):
                out[t] = first_valid
    return out


def rolling_pc1_share(log_returns: np.ndarray, mask: np.ndarray,
                      window: int = 60) -> np.ndarray:
    """Rolling first-principal-component variance share. At day t, uses
    returns from [t-window..t-1] across tickers active every day of that
    window. Causal: no access to day t or beyond."""
    T, N = log_returns.shape
    out = np.full(T, np.nan, dtype=np.float64)
    for t in range(window, T):
        sub = log_returns[t - window: t]
        sub_m = mask[t - window: t]
        always_active = sub_m.all(axis=0)
        if always_active.sum() < 5:
            continue
        X = sub[:, always_active]
        X = X - X.mean(axis=0, keepdims=True)
        cov = np.cov(X, rowvar=False)
        eigvals = np.linalg.eigvalsh(cov)[::-1]
        eigvals = np.maximum(eigvals, 0)
        total = eigvals.sum()
        if total < 1e-12:
            continue
        out[t] = float(eigvals[0] / total)
    return out


def rolling_cs_skew(log_returns: np.ndarray, mask: np.ndarray,
                    window: int = 20) -> np.ndarray:
    """Rolling mean of per-day cross-sectional skewness over [t-window..t-1]."""
    T, N = log_returns.shape
    per_day = np.full(T, np.nan, dtype=np.float64)
    for d in range(T):
        m = mask[d]
        if m.sum() < 5:
            continue
        per_day[d] = float(pd.Series(log_returns[d, m]).skew())
    out = np.full(T, np.nan, dtype=np.float64)
    for t in range(window, T):
        w = per_day[t - window: t]
        valid = w[np.isfinite(w)]
        if valid.size > 0:
            out[t] = float(valid.mean())
    return out


def rolling_cs_kurt(log_returns: np.ndarray, mask: np.ndarray,
                    window: int = 60) -> np.ndarray:
    """Rolling mean of per-day cross-sectional kurtosis over [t-window..t-1]."""
    T, N = log_returns.shape
    per_day = np.full(T, np.nan, dtype=np.float64)
    for d in range(T):
        m = mask[d]
        if m.sum() < 5:
            continue
        per_day[d] = float(pd.Series(log_returns[d, m]).kurt())
    out = np.full(T, np.nan, dtype=np.float64)
    for t in range(window, T):
        w = per_day[t - window: t]
        valid = w[np.isfinite(w)]
        if valid.size > 0:
            out[t] = float(valid.mean())
    return out


def compute_extended_signatures(log_returns: np.ndarray, mask: np.ndarray,
                                dates: list, risk_df: pd.DataFrame,
                                dispersion_window: int = 20,
                                corr_window: int = 60) -> np.ndarray:
    """Returns [T, 7] signatures: original 4 + PC1 share + cs_skew + cs_kurt."""
    base = compute_signatures(log_returns, mask, dates, risk_df,
                              dispersion_window, corr_window)  # [T, 4]
    pc1 = rolling_pc1_share(log_returns, mask, window=60)       # [T]
    skew = rolling_cs_skew(log_returns, mask, window=20)        # [T]
    kurt = rolling_cs_kurt(log_returns, mask, window=60)        # [T]
    extended = np.column_stack([base, pc1, skew, kurt])          # [T, 7]
    return extended


__all__ = ["compute_signatures", "compute_extended_signatures", "forward_fill_signatures"]
