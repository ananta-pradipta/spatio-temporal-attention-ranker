"""Sector-z-scoring + train-fold standardization for LATTICE.

Per spec section 5.1:
- 4 distress proxies sector-z-scored within fold
- 4 intangible proxies sector-z-scored within fold
- 3 other fundamentals sector-z-scored

The remaining feature blocks (price+vol, stocktwits, catalyst, flags) are
NOT sector-z-scored per spec; they are panel-z-scored at training time using
train-fold mean and std.

Per spec section 2.2 (audit A3): standardization statistics use train fold
only. Scalers are saved per fold to
  experiments/lattice/<fold>/scalers.pkl
and re-loaded at validation/test time. NEVER recompute scalers using val or
test data.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.lattice.data.build_panel import (
    DISTRESS_COLS, INTANGIBLE_COLS, OTHER_FUND_COLS,
    PRICE_VOL_COLS, KLINE_COLS, MOMENTUM_EXTRA_COLS, VOL_PV_EXTRA_COLS,
    CATALYST_COLS, FLAG_COLS, ST_FEATURE_COLS,
    MACRO_FEATURE_COLS,
)


@dataclass
class FoldScaler:
    """Per-fold standardization statistics.

    Attributes:
        fold: integer fold id (1-3).
        sector_zscore_stats: dict mapping (feature, sector) -> (mean, std).
            Used for the 4+4+3 sector-z-scored features.
        panel_zscore_stats: dict mapping feature -> (mean, std).
            Used for price+vol + stocktwits + catalyst + macro features.
        train_idx_count: integer number of train days the scaler was fit on.
    """

    fold: int
    sector_zscore_stats: dict
    panel_zscore_stats: dict
    train_idx_count: int


SECTOR_Z_FEATURES = DISTRESS_COLS + INTANGIBLE_COLS + OTHER_FUND_COLS
# K-line, multi-horizon momentum, and price-volume-extra additions get
# panel-z-scoring (panel-wide stats, not sector-specific): they describe
# universal price-action geometry, not a sector-specific fundamental.
PANEL_Z_FEATURES = (
    PRICE_VOL_COLS + KLINE_COLS + MOMENTUM_EXTRA_COLS + VOL_PV_EXTRA_COLS
    + ST_FEATURE_COLS + CATALYST_COLS
)
NO_Z_FEATURES = FLAG_COLS  # binary flags, no z-score


def fit_fold_scaler(
    panel: pd.DataFrame,
    train_idx: np.ndarray,
    cohorts: pd.DataFrame,
    fold: int,
) -> FoldScaler:
    """Fit per-fold standardization on train days only.

    Args:
        panel: per-(ticker, date) panel features parquet (pre-z-score).
        train_idx: integer indices into the unique sorted dates list.
        cohorts: per-(ticker, date) cohort labels (provides 'sector' column).
        fold: integer fold id.

    Returns:
        FoldScaler with sector and panel z-score stats fit only on train.
    """
    panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)
    dates = sorted(panel["date"].unique())
    train_dates = set(np.asarray(dates)[train_idx])
    train_panel = panel[panel["date"].isin(train_dates)].copy()
    train_panel = train_panel.merge(
        cohorts[["ticker", "date", "sector"]], how="left", on=["ticker", "date"]
    )

    sector_zscore_stats: dict = {}
    for feat in SECTOR_Z_FEATURES:
        if feat not in train_panel.columns:
            continue
        for sec, sub in train_panel.groupby("sector"):
            vals = sub[feat].replace([np.inf, -np.inf], np.nan).dropna()
            if len(vals) < 30:
                # Fall back to global train stats for thin sectors
                continue
            mu = float(vals.mean())
            sd = float(vals.std(ddof=0))
            if sd < 1e-6:
                sd = 1.0
            sector_zscore_stats[(feat, sec)] = (mu, sd)
        # Global fallback for sectors with too-few observations
        global_vals = train_panel[feat].replace([np.inf, -np.inf], np.nan).dropna()
        if len(global_vals) > 0:
            mu_g = float(global_vals.mean())
            sd_g = float(global_vals.std(ddof=0))
            if sd_g < 1e-6:
                sd_g = 1.0
            sector_zscore_stats[(feat, "_global_")] = (mu_g, sd_g)

    panel_zscore_stats: dict = {}
    for feat in PANEL_Z_FEATURES:
        if feat not in train_panel.columns:
            continue
        vals = train_panel[feat].replace([np.inf, -np.inf], np.nan).dropna()
        if len(vals) < 30:
            continue
        mu = float(vals.mean())
        sd = float(vals.std(ddof=0))
        if sd < 1e-6:
            sd = 1.0
        panel_zscore_stats[feat] = (mu, sd)

    return FoldScaler(
        fold=fold,
        sector_zscore_stats=sector_zscore_stats,
        panel_zscore_stats=panel_zscore_stats,
        train_idx_count=len(train_idx),
    )


def apply_fold_scaler(
    panel: pd.DataFrame, cohorts: pd.DataFrame, scaler: FoldScaler,
) -> pd.DataFrame:
    """Apply the saved per-fold scaler to the full panel.

    Args:
        panel: per-(ticker, date) panel features parquet.
        cohorts: per-(ticker, date) cohort labels with 'sector' column.
        scaler: fitted FoldScaler.

    Returns:
        Panel with standardized features in the same column names.
        Sector-z-scored features use within-sector train stats; panel-z
        features use global train stats. Binary flags untouched.
    """
    panel = panel.merge(
        cohorts[["ticker", "date", "sector"]], how="left", on=["ticker", "date"]
    )

    out = panel.copy()
    for feat in SECTOR_Z_FEATURES:
        if feat not in out.columns:
            continue
        new_col = np.zeros(len(out), dtype=np.float32)
        for sec, sub_idx in out.groupby("sector").groups.items():
            stats = scaler.sector_zscore_stats.get((feat, sec))
            if stats is None:
                stats = scaler.sector_zscore_stats.get((feat, "_global_"))
            if stats is None:
                stats = (0.0, 1.0)
            mu, sd = stats
            new_col[sub_idx] = ((out.loc[sub_idx, feat]
                                  .replace([np.inf, -np.inf], np.nan).fillna(mu)
                                  - mu) / sd).clip(lower=-5, upper=5).astype(np.float32)
        out[feat] = new_col

    for feat in PANEL_Z_FEATURES:
        if feat not in out.columns:
            continue
        stats = scaler.panel_zscore_stats.get(feat)
        if stats is None:
            stats = (0.0, 1.0)
        mu, sd = stats
        out[feat] = ((out[feat].replace([np.inf, -np.inf], np.nan).fillna(mu)
                       - mu) / sd).clip(lower=-5, upper=5).astype(np.float32)

    return out.drop(columns=["sector"])


def save_fold_scaler(scaler: FoldScaler, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(scaler, f)


def load_fold_scaler(path: Path) -> FoldScaler:
    with open(path, "rb") as f:
        return pickle.load(f)


__all__ = [
    "FoldScaler", "SECTOR_Z_FEATURES", "PANEL_Z_FEATURES", "NO_Z_FEATURES",
    "fit_fold_scaler", "apply_fold_scaler",
    "save_fold_scaler", "load_fold_scaler",
]
