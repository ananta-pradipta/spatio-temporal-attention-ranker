"""Multi-horizon panel: returns forward-return targets at several horizons.

Wraps `src.mtgn.training.panel.build_panel` to produce a dict of targets
keyed by horizon (in trading days). Used by the multi-horizon training
loop so per-feature-group signal can find its natural-horizon target.

The horizon sweep (2026-04-13) showed:
  h=1   StockTwits IC +0.0138 (best), Price IC -0.0115 (reversal)
  h=5   Price IC +0.0387, StockTwits IC +0.0159 (worse than h=1)

A single 5-day target forces StockTwits features to predict a signal
that has already reversed (Yang et al. 2024: 1-2d sentiment half-life).
Multi-horizon heads let the model route each feature group to its
natural target horizon.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.mtgn.training.panel import (
    FEATURE_COLS,
    PanelConfig,
    build_panel,
    panel_to_tensors,
)


def _compute_multi_horizon_returns(
    prices_parquet: Path, tickers: list[str], horizons: list[int]
) -> dict[int, pd.DataFrame]:
    """Returns a dict {h: DataFrame[ticker, date, fwd_return_h]} per horizon."""
    prices = pd.read_parquet(prices_parquet)
    prices["ticker"] = prices["ticker"].astype(str).str.upper()
    prices = prices[prices["ticker"].isin(set(tickers))]
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)

    out: dict[int, pd.DataFrame] = {}
    for h in horizons:
        frames = []
        for t, sub in prices.groupby("ticker", sort=False):
            sub = sub.copy()
            sub[f"fwd_return_{h}"] = np.log(sub["close"].shift(-h) / sub["close"])
            frames.append(sub[["ticker", "date", f"fwd_return_{h}"]])
        out[h] = pd.concat(frames, ignore_index=True)
    return out


@dataclass
class MultiHorizonPanel:
    """Dense [T, N, F] features and [T, N] targets for multiple horizons."""
    x: np.ndarray               # [T, N, F]
    y_by_h: dict[int, np.ndarray]     # h -> [T, N]
    mask: np.ndarray            # [T, N]   valid (ticker, date) cell
    tickers: list[str]
    dates: list[pd.Timestamp]


def build_multi_horizon_panel(
    cfg: PanelConfig, horizons: list[int]
) -> MultiHorizonPanel:
    """Build a panel once with cfg.horizon_days, then overlay per-horizon targets.

    The underlying `build_panel` drops rows with NaN target for its
    configured horizon. To keep a single shared mask across all
    horizons, we use the MAX horizon for the base panel (ensures any
    valid (ticker, date) has a forward-return at all horizons) and then
    recompute each horizon's target.
    """
    max_h = max(horizons)
    base_cfg = PanelConfig(
        prices_parquet=cfg.prices_parquet,
        stocktwits_features_parquet=cfg.stocktwits_features_parquet,
        volatility_parquet=cfg.volatility_parquet,
        universe_csv=cfg.universe_csv,
        start_date=cfg.start_date,
        end_date=cfg.end_date,
        horizon_days=max_h,
        max_tickers=cfg.max_tickers,
        vol_window=cfg.vol_window,
    )
    panel, tickers, dates = build_panel(base_cfg)
    tensors = panel_to_tensors(panel, tickers, dates)
    x = tensors["x"]
    mask = tensors["mask"]

    # Compute per-horizon targets on the same panel grid
    multi_rets = _compute_multi_horizon_returns(cfg.prices_parquet, tickers, horizons)
    ticker_to_i = {t: i for i, t in enumerate(tickers)}
    date_to_i = {d: i for i, d in enumerate(dates)}
    y_by_h: dict[int, np.ndarray] = {}
    for h, df in multi_rets.items():
        y = np.zeros((len(dates), len(tickers)), dtype=np.float32)
        for _, row in df.iterrows():
            ti = ticker_to_i.get(row["ticker"])
            di = date_to_i.get(pd.Timestamp(row["date"]))
            if ti is None or di is None:
                continue
            y[di, ti] = float(row[f"fwd_return_{h}"])
        y_by_h[h] = y

    return MultiHorizonPanel(x=x, y_by_h=y_by_h, mask=mask, tickers=tickers, dates=dates)
