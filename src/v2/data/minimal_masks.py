"""Minimal mask separation for OW-epiSTAR v1.

The existing ``active mask`` from ``panel_enriched.py`` is the row indicator
of the model-ready feature panel; that panel drops rows missing any of
``fwd_return_h`` (requires close[t+5]), ``log_return``, ``log_volume``,
or ``realized_vol_20d`` (requires 20 days of return history). As a
result, the active mask conflates three distinct concepts:

    - tradability (close and volume present at day t)
    - label availability (5-day forward return computable)
    - sufficient history (20-day rolling features computable)

For survivorship-aware modelling of late IPOs we need to separate these.
This module builds:

    - ``tradable_mask[t, i]``      : raw close and volume present at t
    - ``label_mask[t, i]``         : close[t + horizon] present and
                                     tradable_mask[t, i] true
    - ``loss_mask[t, i]``          : tradable_mask & label_mask
    - ``graph_candidate_mask``     : alias of tradable_mask
    - ``history_available[t, i]``  : trailing-window valid-day count for
                                     L=20 and L=60 (separate arrays)

The masks are derived from the raw price panel
(``data/raw/prices_universe.parquet``) so the early-life period of late
IPOs is preserved (the existing active mask erases the first 20 days
because realized_vol_20d is NaN).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class MinimalMaskConfig:
    """Hyperparameters for minimal-mask construction."""

    raw_prices_parquet: Path = Path("data/raw/prices_universe.parquet")
    horizon_days: int = 5
    history_window_short: int = 20
    history_window_long: int = 60


def build_minimal_masks(
    dates: list[pd.Timestamp],
    tickers: list[str],
    cfg: MinimalMaskConfig | None = None,
) -> dict[str, np.ndarray]:
    """Build the minimal mask family aligned to ``dates`` x ``tickers``.

    Args:
        dates: panel trading days (from build_enriched_panel).
        tickers: panel tickers (from build_enriched_panel).
        cfg: configuration; defaults to MinimalMaskConfig().

    Returns:
        Dict with keys:
            tradable_mask:        [T, N] bool
            label_mask:           [T, N] bool
            loss_mask:            [T, N] bool
            graph_candidate_mask: [T, N] bool (alias of tradable_mask)
            history_valid_20d:    [T, N] float, fraction in [0, 1]
            history_valid_60d:    [T, N] float, fraction in [0, 1]

    Notes:
        history_valid_*d counts tradable days in a *trailing* window
        (no future leakage). Tickers with fewer than L observed prior
        days have a partial-window denominator of L (so the ratio is
        small for new tickers, as desired).
    """
    cfg = cfg or MinimalMaskConfig()

    raw = pd.read_parquet(cfg.raw_prices_parquet)
    raw = raw.copy()
    raw["date"] = pd.to_datetime(raw["date"]).dt.normalize()

    panel_dates = pd.DatetimeIndex(pd.to_datetime(dates).normalize())
    raw = raw[raw["date"].isin(panel_dates)]
    raw = raw[raw["ticker"].isin(tickers)]

    ticker_to_idx = {t: i for i, t in enumerate(tickers)}
    date_to_idx = {pd.Timestamp(d): i for i, d in enumerate(panel_dates)}
    t_total, n = len(panel_dates), len(tickers)

    tradable = np.zeros((t_total, n), dtype=bool)
    has_close = raw["close"].notna() & (raw["close"] > 0)
    has_volume = raw["volume"].notna() & (raw["volume"] > 0)
    is_tradable = has_close & has_volume
    sub = raw[is_tradable]
    di = sub["date"].map(date_to_idx).to_numpy()
    ti = sub["ticker"].map(ticker_to_idx).to_numpy()
    valid = ~pd.isna(di) & ~pd.isna(ti)
    di = di[valid].astype(np.int64)
    ti = ti[valid].astype(np.int64)
    tradable[di, ti] = True

    label = np.zeros((t_total, n), dtype=bool)
    h = cfg.horizon_days
    if t_total > h:
        # close[t + h] must exist for ticker i to compute the 5d log return.
        future_tradable = tradable[h:]
        label[: t_total - h] = tradable[: t_total - h] & future_tradable

    loss = tradable & label

    hist20 = _trailing_valid_ratio(tradable, cfg.history_window_short)
    hist60 = _trailing_valid_ratio(tradable, cfg.history_window_long)

    return {
        "tradable_mask": tradable,
        "label_mask": label,
        "loss_mask": loss,
        "graph_candidate_mask": tradable.copy(),
        "history_valid_20d": hist20,
        "history_valid_60d": hist60,
    }


def _trailing_valid_ratio(mask: np.ndarray, window: int) -> np.ndarray:
    """Trailing fraction of days in [t - window + 1, t] where ``mask`` is True."""
    t_total, n = mask.shape
    out = np.zeros((t_total, n), dtype=np.float32)
    if window <= 0:
        return out
    cumulative = np.cumsum(mask.astype(np.int32), axis=0)
    for t in range(t_total):
        lo = t - window + 1
        if lo <= 0:
            count = cumulative[t]
        else:
            count = cumulative[t] - cumulative[lo - 1]
        out[t] = count.astype(np.float32) / float(window)
    return out


def compute_age_from_tradable(tradable_mask: np.ndarray) -> np.ndarray:
    """Cumulative tradable-day count per (day, ticker).

    Per spec: age_trading_days[t, i] is the cumulative number of tradable
    days for ticker i from the first tradable date through and including t.
    First tradable day has age=1; a ticker that has never been tradable
    has age=0 throughout.
    """
    return np.cumsum(tradable_mask.astype(np.int64), axis=0)


__all__ = [
    "MinimalMaskConfig",
    "build_minimal_masks",
    "compute_age_from_tradable",
]
