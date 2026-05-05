"""Day-level episode keys for epiSTAR retrieval.

For each trading day s, build a regime-context key vector composed of:
  (a) the existing 8-dimensional risk feature vector (VIX, VXN, VVIX,
      VIX term slope, XBI 20d/60d realized vol, 5d VIX change, XBI 5d
      forward absolute return),
  (b) cross-sectional regime diagnostics computed from the panel's
      log_return column (rolling first-principal-component variance share
      over 60 days, average pairwise correlation over 60 days, current-day
      cross-sectional dispersion, skewness, kurtosis, and active ticker
      count as a regularizer feature).

The key is the cosine-similarity input for memory retrieval. It is
intentionally compact; macro feeds (Treasury yields, EPU, GPR) are
deferred until they are pulled.

All cross-sectional diagnostics are causal: day-s diagnostics are computed
from data available on or before day s.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


RISK_COLS = [
    "vix",
    "vxn",
    "vvix",
    "vix_term_slope",
    "xbi_rv_20d",
    "xbi_rv_60d",
    "vix_5d_change",
    "xbi_fwd_abs_ret_5d",
]

CS_DIAG_COLS = [
    "cs_pc1_share_60d",
    "cs_avg_pairwise_corr_60d",
    "cs_dispersion",
    "cs_skewness",
    "cs_kurtosis",
    "cs_active_count",
]

EPISODE_KEY_COLS = RISK_COLS + CS_DIAG_COLS


@dataclass
class EpisodeKeyConfig:
    """Hyperparameters for episode-key construction.

    Attributes:
        risk_features_parquet: path to the existing 8-dim risk feature file.
        cs_window_days: rolling window for PC1 and pairwise correlation.
        active_count_norm: divisor applied to cs_active_count to bring it
            into a comparable scale before standardization.
    """

    risk_features_parquet: Path = Path("data/processed/risk_features.parquet")
    cs_window_days: int = 60
    active_count_norm: float = 250.0


def build_cross_sectional_diagnostics(
    log_returns: np.ndarray,
    mask: np.ndarray,
    cfg: EpisodeKeyConfig,
) -> pd.DataFrame:
    """Compute per-day cross-sectional regime diagnostics.

    Args:
        log_returns: [T, N] daily log returns of the panel.
        mask: [T, N] bool, True where the ticker is active on that day.
        cfg: hyperparameters.

    Returns:
        DataFrame with one row per trading day index t and columns from
        CS_DIAG_COLS. Days where the rolling window is incomplete have NaN
        in the rolling features; the loader forward-fills these.
    """
    t_total, n = log_returns.shape
    out = np.full((t_total, len(CS_DIAG_COLS)), np.nan, dtype=np.float32)
    w = cfg.cs_window_days

    for t in range(t_total):
        active = mask[t]
        if active.sum() < 5:
            continue
        r_t = log_returns[t, active]

        out[t, 2] = float(np.std(r_t, ddof=0))
        out[t, 3] = float(_skewness(r_t))
        out[t, 4] = float(_kurtosis(r_t))
        out[t, 5] = float(active.sum()) / cfg.active_count_norm

        if t < w:
            continue
        # Use the trailing w days where each ticker is required to be active
        # for the whole window. Tickers that drop in mid-window are excluded.
        win_mask = mask[t - w + 1 : t + 1]  # [w, N]
        full_active = win_mask.all(axis=0)
        if full_active.sum() < 10:
            continue
        win_r = log_returns[t - w + 1 : t + 1, full_active]  # [w, n_active]

        # Cross-sectional standardization across tickers, per day, then SVD
        # on the de-meaned matrix to estimate variance shares.
        x = win_r - win_r.mean(axis=0, keepdims=True)
        s = np.std(x, axis=0, ddof=0)
        s = np.where(s < 1e-8, 1e-8, s)
        x = x / s
        # First principal component variance share: largest singular value
        # squared divided by sum of squared singular values.
        try:
            sv = np.linalg.svd(x, compute_uv=False)
            ev = sv ** 2
            out[t, 0] = float(ev[0] / ev.sum()) if ev.sum() > 0 else np.nan
        except np.linalg.LinAlgError:
            out[t, 0] = np.nan

        # Average pairwise correlation: corrcoef across tickers, then mean
        # of off-diagonal entries.
        if full_active.sum() <= 200:
            corr = np.corrcoef(win_r.T)
            np.fill_diagonal(corr, np.nan)
            out[t, 1] = float(np.nanmean(corr))
        else:
            sample = np.random.default_rng(0).choice(
                full_active.sum(), size=200, replace=False
            )
            corr = np.corrcoef(win_r[:, sample].T)
            np.fill_diagonal(corr, np.nan)
            out[t, 1] = float(np.nanmean(corr))

    df = pd.DataFrame(out, columns=CS_DIAG_COLS)
    return df


def _skewness(x: np.ndarray) -> float:
    if x.size < 3:
        return 0.0
    m = x.mean()
    s = x.std(ddof=0)
    if s < 1e-8:
        return 0.0
    return float(((x - m) ** 3).mean() / (s ** 3))


def _kurtosis(x: np.ndarray) -> float:
    if x.size < 4:
        return 0.0
    m = x.mean()
    s = x.std(ddof=0)
    if s < 1e-8:
        return 0.0
    return float(((x - m) ** 4).mean() / (s ** 4) - 3.0)


def build_episode_keys(
    dates: list[pd.Timestamp],
    log_returns: np.ndarray,
    mask: np.ndarray,
    cfg: EpisodeKeyConfig | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Build the per-day episode key matrix.

    Args:
        dates: list of trading days, length T.
        log_returns: [T, N] panel log returns.
        mask: [T, N] active mask.
        cfg: hyperparameters.

    Returns:
        keys: [T, K] float32 raw (unstandardized) episode keys.
        col_names: names of the K columns.
    """
    cfg = cfg or EpisodeKeyConfig()
    risk = pd.read_parquet(cfg.risk_features_parquet)
    risk = risk.copy()
    risk.index = pd.to_datetime(risk.index)
    # Align to panel dates; forward-fill missing risk rows (closed-but-no-data days)
    risk_aligned = risk.reindex(pd.to_datetime(dates)).ffill().bfill()
    risk_arr = risk_aligned[RISK_COLS].to_numpy(dtype=np.float32)

    diag_df = build_cross_sectional_diagnostics(log_returns, mask, cfg)
    diag_df = diag_df.ffill().bfill().fillna(0.0)
    diag_arr = diag_df.to_numpy(dtype=np.float32)

    keys = np.concatenate([risk_arr, diag_arr], axis=1)
    return keys, EPISODE_KEY_COLS


__all__ = [
    "EpisodeKeyConfig",
    "EPISODE_KEY_COLS",
    "RISK_COLS",
    "CS_DIAG_COLS",
    "build_episode_keys",
    "build_cross_sectional_diagnostics",
]
