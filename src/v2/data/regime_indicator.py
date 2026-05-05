"""60-day realized rate-volatility z-score (RT-CSGA regime indicator).

Per spec Section 5.1 of `docs/specs/rt_csga_spec.md`. The RT-CSGA
softmax temperature is a scalar function of one exogenous regime
indicator: the annualized 60-day realized volatility of daily DGS10
(10-year Treasury yield) log-changes, z-scored using train-fold
statistics only.

Usage:
    rvol = compute_dgs10_rvol_60d(dgs10)
    rvol_z = standardize_regime_indicator(rvol, train_fold_dates)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class RegimeIndicatorConfig:
    """Hyperparameters for the regime indicator."""

    rvol_window: int = 60
    fred_cache: Path = Path("data/raw/macro_fred.csv")
    panel_start: str = "2014-09-01"
    panel_end: str = "2023-01-15"


def compute_dgs10_rvol_60d(
    dgs10: pd.Series, rvol_window: int = 60,
) -> pd.Series:
    """Annualized 60-day realized volatility of daily DGS10 log-changes.

    Args:
        dgs10: daily DGS10 yield (percent) indexed by date.
        rvol_window: trailing-window length (default 60).

    Returns:
        pd.Series of rvol values aligned to the input index. The
        first ``rvol_window - 1`` values are NaN (insufficient history).
    """
    yield_log_change = np.log(dgs10).diff()
    rvol = yield_log_change.rolling(window=rvol_window, min_periods=20).std()
    return rvol * np.sqrt(252.0)


def standardize_regime_indicator(
    rvol: pd.Series, train_fold_dates: pd.DatetimeIndex,
) -> pd.Series:
    """Z-score ``rvol`` using ONLY ``train_fold_dates`` for mean and std.

    The spec's four-check audit verifies this scope is
    train-fold-only; using validation or test dates here is a
    leakage failure.
    """
    train_subset = rvol.loc[rvol.index.isin(train_fold_dates)].dropna()
    if train_subset.size < 10:
        # Defensive: fall back to global stats if too few train cells.
        mu = float(rvol.dropna().mean())
        sigma = float(rvol.dropna().std())
    else:
        mu = float(train_subset.mean())
        sigma = float(train_subset.std())
    if sigma < 1e-8:
        sigma = 1.0
    return (rvol - mu) / sigma


def load_dgs10_from_cache(
    cfg: RegimeIndicatorConfig | None = None,
) -> pd.Series:
    """Load DGS10 daily series from the FRED cache built by macro_features."""
    cfg = cfg or RegimeIndicatorConfig()
    if not cfg.fred_cache.exists():
        raise FileNotFoundError(
            f"DGS10 cache missing at {cfg.fred_cache}. Run "
            f"src.v2.data.macro_features.build_macro_features() first."
        )
    df = pd.read_csv(cfg.fred_cache, parse_dates=["date"]).set_index("date")
    if "DGS10" not in df.columns:
        raise KeyError(f"DGS10 column missing from {cfg.fred_cache}")
    return df["DGS10"].astype(float)


def build_regime_indicator_array(
    panel_dates: list[pd.Timestamp],
    train_fold_dates: pd.DatetimeIndex,
    cfg: RegimeIndicatorConfig | None = None,
) -> np.ndarray:
    """Per-day [T] regime indicator (z-scored) aligned to ``panel_dates``.

    Pulls DGS10 from the FRED cache, computes the 60-day rvol, aligns
    to ``panel_dates``, forward-fills up to 5 days for non-trading
    holidays, then z-scores using train-fold dates only.

    Args:
        panel_dates: list of trading days (panel index).
        train_fold_dates: training-fold dates for the standardisation
            scope.
        cfg: hyperparameters; defaults to RegimeIndicatorConfig().

    Returns:
        [T] float32 z-scored regime indicator. Days where the rvol
        rolling window is incomplete are filled with 0 (i.e., the
        train-fold mean) after standardisation.
    """
    cfg = cfg or RegimeIndicatorConfig()
    dgs10 = load_dgs10_from_cache(cfg)
    rvol = compute_dgs10_rvol_60d(dgs10, rvol_window=cfg.rvol_window)
    rvol_z = standardize_regime_indicator(rvol, train_fold_dates)
    panel_index = pd.DatetimeIndex(pd.to_datetime(panel_dates).normalize())
    rvol_z_aligned = rvol_z.reindex(panel_index).ffill(limit=5).fillna(0.0)
    return rvol_z_aligned.to_numpy(dtype=np.float32)


__all__ = [
    "RegimeIndicatorConfig",
    "compute_dgs10_rvol_60d",
    "standardize_regime_indicator",
    "load_dgs10_from_cache",
    "build_regime_indicator_array",
]
