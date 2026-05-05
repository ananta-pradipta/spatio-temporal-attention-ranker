"""Daily cross-sectional panel loader for MTGN Phase 1 training.

Joins three sources on (ticker, trading-date):

  * prices_universe.parquet     (OHLCV + derived returns)
  * stocktwits_features.parquet (5 StockTwits Phase-1 features)
  * volatility_indices.parquet  (VIX, VXN, VVIX broadcast to all tickers)

Produces:

  * `panel`   :  long-format DataFrame with columns
                 ticker, date, log_return, log_volume, realized_vol,
                 st_volume_24h, st_volume_change_30d, st_bullish_ratio,
                 st_sentiment_dispersion, st_labeled_ratio,
                 VIX, VXN, VVIX, fwd_return_h
  * `tickers` :  list of unique tickers in the panel, sorted
  * `dates`   :  list of unique trading dates in the panel, sorted

Forward return for horizon h is log(close_{t+h}/close_t).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


PRICE_COLS = ["log_return", "log_volume", "realized_vol"]
ST_COLS = [
    "st_volume_24h",
    "st_volume_change_30d",
    "st_bullish_ratio",
    "st_sentiment_dispersion",
    "st_labeled_ratio",
]
# VIX/VXN/VVIX are broadcast identically across tickers on any given day,
# so they have ZERO cross-sectional variance and cannot discriminate
# between tickers in a ranking setup (signal audit 2026-04-13). Removed
# from per-node features. May return later as a global gate scalar.
VOL_COLS: list[str] = []
FEATURE_COLS = PRICE_COLS + ST_COLS + VOL_COLS


@dataclass
class PanelConfig:
    prices_parquet: Path = Path("data/raw/prices_universe.parquet")
    stocktwits_features_parquet: Path = Path("data/processed/stocktwits_features.parquet")
    volatility_parquet: Path = Path("data/raw/volatility_indices.parquet")
    universe_csv: Path = Path("data/raw/biotech_universe_v1.csv")
    start_date: str = "2020-01-01"
    end_date: str = "2022-12-31"   # bounded by StockTwits corpus (2022-12-31)
    horizon_days: int = 5
    max_tickers: int | None = None
    vol_window: int = 20


def _load_active_tickers(universe_csv: Path, limit: int | None) -> list[str]:
    u = pd.read_csv(universe_csv)
    if "status" in u.columns:
        u = u[u["status"] == "active"]
    tickers = sorted(u["ticker"].dropna().astype(str).str.upper().unique().tolist())
    if limit is not None:
        tickers = tickers[:limit]
    return tickers


def build_panel(cfg: PanelConfig | None = None) -> tuple[pd.DataFrame, list[str], list[pd.Timestamp]]:
    cfg = cfg or PanelConfig()

    tickers = _load_active_tickers(cfg.universe_csv, cfg.max_tickers)
    tickers_set = set(tickers)

    # Prices
    prices = pd.read_parquet(cfg.prices_parquet)
    prices["ticker"] = prices["ticker"].astype(str).str.upper()
    prices = prices[prices["ticker"].isin(tickers_set)]
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices[(prices["date"] >= cfg.start_date) & (prices["date"] <= cfg.end_date)]
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Per-ticker derived features
    frames = []
    for t, sub in prices.groupby("ticker", sort=False):
        sub = sub.copy()
        sub["log_return"] = np.log(sub["close"]).diff()
        sub["log_volume"] = np.log(sub["volume"].replace(0, np.nan))
        sub["realized_vol"] = sub["log_return"].rolling(cfg.vol_window, min_periods=5).std()
        sub["fwd_return_h"] = np.log(sub["close"].shift(-cfg.horizon_days) / sub["close"])
        frames.append(sub)
    prices = pd.concat(frames, ignore_index=True)

    # StockTwits features
    st = pd.read_parquet(cfg.stocktwits_features_parquet)
    st["ticker"] = st["ticker"].astype(str).str.upper()
    st["date"] = pd.to_datetime(st["date"])
    st = st[(st["ticker"].isin(tickers_set))
            & (st["date"] >= cfg.start_date)
            & (st["date"] <= cfg.end_date)]

    # Merge
    panel = prices.merge(st, how="left", on=["ticker", "date"])

    # Impute missing StockTwits with zeros / neutral priors.
    panel["st_volume_24h"] = panel["st_volume_24h"].fillna(0.0)
    panel["st_volume_change_30d"] = panel["st_volume_change_30d"].fillna(1.0)
    panel["st_bullish_ratio"] = panel["st_bullish_ratio"].fillna(0.5)
    panel["st_sentiment_dispersion"] = panel["st_sentiment_dispersion"].fillna(0.0)
    panel["st_labeled_ratio"] = panel["st_labeled_ratio"].fillna(0.0)

    # Sanitize any remaining inf / NaN that can appear when a rolling
    # baseline is ~0 (st_volume_change_30d on newly-listed tickers, etc.).
    for c in FEATURE_COLS:
        if c in panel.columns:
            panel[c] = pd.to_numeric(panel[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
    # Log-transform + clip the volume-change ratio to tame heavy tails.
    panel["st_volume_change_30d"] = np.log1p(panel["st_volume_change_30d"].clip(lower=0.0, upper=1000.0))
    panel["st_volume_24h"] = np.log1p(panel["st_volume_24h"].clip(lower=0.0))
    # Final fill for any straggler NaN in numeric feature cols.
    panel[FEATURE_COLS] = panel[FEATURE_COLS].fillna(0.0)
    # Winsorize 1st and 99th percentile on log_return to kill extreme outliers.
    lo = panel["log_return"].quantile(0.005)
    hi = panel["log_return"].quantile(0.995)
    panel["log_return"] = panel["log_return"].clip(lower=lo, upper=hi)

    # Drop rows with missing prediction target or required price features.
    panel = panel.dropna(subset=["log_return", "fwd_return_h", "log_volume", "realized_vol"])
    panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)

    dates = sorted(panel["date"].unique().tolist())
    tickers_present = sorted(panel["ticker"].unique().tolist())
    return panel, tickers_present, dates


def panel_to_tensors(
    panel: pd.DataFrame, tickers: list[str], dates: list[pd.Timestamp]
) -> dict[str, np.ndarray]:
    """Reshape the long panel into dense [T, N, F] arrays for training.

    Missing (ticker, date) cells (ticker not trading that day) are filled
    with zeros and returned as a boolean `mask` array.
    """
    import numpy as np

    ticker_to_idx = {t: i for i, t in enumerate(tickers)}
    date_to_idx = {d: i for i, d in enumerate(dates)}

    T, N, F = len(dates), len(tickers), len(FEATURE_COLS)
    x = np.zeros((T, N, F), dtype=np.float32)
    y = np.zeros((T, N), dtype=np.float32)
    mask = np.zeros((T, N), dtype=bool)

    ti = panel["ticker"].map(ticker_to_idx).to_numpy()
    di = panel["date"].map(date_to_idx).to_numpy()
    x[di, ti] = panel[FEATURE_COLS].to_numpy(dtype=np.float32)
    y[di, ti] = panel["fwd_return_h"].to_numpy(dtype=np.float32)
    mask[di, ti] = True
    return {"x": x, "y": y, "mask": mask, "tickers": tickers, "dates": dates}
