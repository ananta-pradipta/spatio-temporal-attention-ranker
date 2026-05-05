"""Enriched panel for Ridge + LSTM baselines.

Extends src.mtgn.training.panel.build_panel:
  (a) 5-year default window (2018-01-01 to 2022-12-31)
  (b) 9 price-derived features (lag returns + vol + range)
  (c) 5 StockTwits features (unchanged)
  (d) 7 fundamentals forward-filled from quarterly + sector-median imputed
  (e) has_fundamentals flag (~22nd feature)

Total feature dim: 22.

Output: x [T, N, F], y [T, N], mask [T, N], plus tickers + dates.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


PRICE_COLS = [
    "log_return",
    "log_return_5d",
    "log_return_20d",
    "log_volume",
    "log_volume_ratio_20d",
    "realized_vol_20d",
    "realized_vol_60d",
    "high_low_range",
    "close_to_high",
]
ST_COLS = [
    "st_volume_24h",
    "st_volume_change_30d",
    "st_bullish_ratio",
    "st_sentiment_dispersion",
    "st_labeled_ratio",
]
FUND_COLS = [
    "log_market_cap",
    "cash_runway_q",
    "rd_intensity",
    "revenue_growth_yoy",
    "cash_to_mc",
    "shares_outstanding_yoy",
    "total_assets_growth",
]
FLAG_COLS = ["has_fundamentals"]
FEATURE_COLS = PRICE_COLS + ST_COLS + FUND_COLS + FLAG_COLS


@dataclass
class EnrichedPanelConfig:
    prices_parquet: Path = Path("data/raw/prices_universe.parquet")
    stocktwits_features_parquet: Path = Path("data/processed/stocktwits_features.parquet")
    fundamentals_parquet: Path = Path("data/raw/fundamentals_edgar.parquet")
    universe_csv: Path = Path("data/raw/biotech_universe_v1.csv")
    start_date: str = "2018-01-01"
    end_date: str = "2022-12-31"
    horizon_days: int = 5
    max_tickers: int | None = None


def _price_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    frames = []
    for t, sub in df.groupby("ticker", sort=False):
        s = sub.copy()
        close = s["close"]
        vol = s["volume"].replace(0, np.nan)
        s["log_return"] = np.log(close).diff()
        s["log_return_5d"] = np.log(close / close.shift(5))
        s["log_return_20d"] = np.log(close / close.shift(20))
        s["log_volume"] = np.log(vol)
        s["log_volume_ratio_20d"] = np.log(vol / vol.rolling(20, min_periods=5).mean())
        s["realized_vol_20d"] = s["log_return"].rolling(20, min_periods=5).std()
        s["realized_vol_60d"] = s["log_return"].rolling(60, min_periods=10).std()
        if "high" in s.columns and "low" in s.columns:
            s["high_low_range"] = np.log(s["high"] / s["low"]).clip(upper=0.5)
            s["close_to_high"] = (close - s["low"]) / (s["high"] - s["low"]).replace(0, np.nan)
        else:
            s["high_low_range"] = 0.0
            s["close_to_high"] = 0.5
        s["fwd_return_h"] = np.log(close.shift(-5) / close)   # default horizon; config overrides later
        frames.append(s)
    return pd.concat(frames, ignore_index=True)


def _derive_fundamentals(fund: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Compute derived fundamentals from the EDGAR quarterly DataFrame.

    Critical: uses `filed_date` (public-availability) for forward-fill
    indexing, NOT `quarter_end`. This prevents the look-ahead leakage
    where e.g. Q4-2020 figures appear in the panel before ~Feb 2021.

    Historical market cap = close_price * shares_outstanding, using the
    close on the FILING date (not quarter-end) because that's the first
    day we know the shares-outstanding figure.

    `fund` columns: ticker, cik, quarter_end, filed_date, cash, assets,
                    shares, revenue, rd_expense, net_income, op_cf
    """
    fund = fund.sort_values(["ticker", "filed_date", "quarter_end"]).reset_index(drop=True).copy()
    prices_small = prices[["ticker", "date", "close"]].copy()
    prices_small["date"] = pd.to_datetime(prices_small["date"]).dt.normalize()
    fund["filed_date"] = pd.to_datetime(fund["filed_date"]).dt.normalize()
    fund["quarter_end"] = pd.to_datetime(fund["quarter_end"]).dt.normalize()

    # Attach close at the FILING date (filing is a trading day in most cases)
    fund = fund.merge(
        prices_small.rename(columns={"date": "filed_date"}),
        how="left", on=["ticker", "filed_date"],
    )
    # If the filing happens on a non-trading day, use the next trading day's close
    def _next_trading_close(sub_fund, sub_prices):
        sub_prices = sub_prices.sort_values("date")
        out = []
        for _, r in sub_fund.iterrows():
            if pd.notna(r.get("close")):
                out.append(r["close"])
                continue
            nxt = sub_prices[sub_prices["date"] >= r["filed_date"]]
            out.append(nxt.iloc[0]["close"] if len(nxt) > 0 else np.nan)
        sub_fund = sub_fund.copy()
        sub_fund["close"] = out
        return sub_fund

    frames = []
    for t, sub in fund.groupby("ticker", sort=False):
        sub_prices = prices_small[prices_small["ticker"] == t]
        sub = _next_trading_close(sub, sub_prices)
        sub["market_cap"] = sub["close"] * sub["shares"]
        # Use filed_date as the effective "date" for downstream forward-fill
        sub["date"] = sub["filed_date"]
        sub["log_market_cap"] = np.log(sub["market_cap"].replace({0: np.nan}))
        sub["burn_q"] = -sub["op_cf"].clip(upper=0)
        sub["cash_runway_q"] = np.where(
            (sub["burn_q"] > 0) & sub["cash"].notna(),
            sub["cash"] / sub["burn_q"],
            np.nan,
        )
        sub["rd_intensity"] = np.where(
            (sub["market_cap"] > 0) & sub["rd_expense"].notna(),
            sub["rd_expense"] / sub["market_cap"],
            np.nan,
        )
        sub["revenue_growth_yoy"] = sub["revenue"].pct_change(4, fill_method=None)
        sub["cash_to_mc"] = np.where(
            (sub["market_cap"] > 0) & sub["cash"].notna(),
            sub["cash"] / sub["market_cap"],
            np.nan,
        )
        sub["shares_outstanding_yoy"] = sub["shares"].pct_change(4, fill_method=None)
        sub["total_assets_growth"] = sub["assets"].pct_change(4, fill_method=None)
        frames.append(sub[["ticker", "date"] + FUND_COLS])
    return pd.concat(frames, ignore_index=True)


def _forward_fill_fundamentals(fund: pd.DataFrame, dates: list[pd.Timestamp]) -> pd.DataFrame:
    """For each ticker, forward-fill quarterly values to daily grid."""
    rows = []
    trading = pd.DatetimeIndex(sorted(dates))
    for t, sub in fund.groupby("ticker"):
        s = sub.sort_values("date").set_index("date")
        s = s[~s.index.duplicated(keep="last")]
        s = s.reindex(trading, method="ffill")
        s["ticker"] = t
        s["date"] = s.index
        rows.append(s.reset_index(drop=True))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_enriched_panel(cfg: EnrichedPanelConfig | None = None) -> tuple[pd.DataFrame, list[str], list[pd.Timestamp]]:
    cfg = cfg or EnrichedPanelConfig()

    u = pd.read_csv(cfg.universe_csv)
    if "status" in u.columns:
        u = u[u["status"] == "active"]
    tickers = sorted(u["ticker"].dropna().astype(str).str.upper().unique().tolist())
    if cfg.max_tickers is not None:
        tickers = tickers[: cfg.max_tickers]
    tickers_set = set(tickers)

    # Prices + derived price features
    prices = pd.read_parquet(cfg.prices_parquet)
    prices["ticker"] = prices["ticker"].astype(str).str.upper()
    prices = prices[prices["ticker"].isin(tickers_set)]
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices[(prices["date"] >= cfg.start_date) & (prices["date"] <= cfg.end_date)]
    prices = _price_features(prices)
    if cfg.horizon_days != 5:
        prices["fwd_return_h"] = np.log(
            prices.groupby("ticker")["close"].shift(-cfg.horizon_days) / prices["close"]
        )

    # StockTwits
    st = pd.read_parquet(cfg.stocktwits_features_parquet)
    st["ticker"] = st["ticker"].astype(str).str.upper()
    st["date"] = pd.to_datetime(st["date"])
    st = st[st["ticker"].isin(tickers_set) & (st["date"] >= cfg.start_date) & (st["date"] <= cfg.end_date)]

    panel = prices.merge(st, how="left", on=["ticker", "date"])
    panel["st_volume_24h"] = np.log1p(panel["st_volume_24h"].fillna(0.0))
    panel["st_volume_change_30d"] = np.log1p(panel["st_volume_change_30d"].fillna(1.0).clip(lower=0, upper=1000))
    panel["st_bullish_ratio"] = panel["st_bullish_ratio"].fillna(0.5)
    panel["st_sentiment_dispersion"] = panel["st_sentiment_dispersion"].fillna(0.0)
    panel["st_labeled_ratio"] = panel["st_labeled_ratio"].fillna(0.0)

    # Fundamentals: derive, forward-fill, merge
    if cfg.fundamentals_parquet.exists():
        fund_raw = pd.read_parquet(cfg.fundamentals_parquet)
        fund_raw["ticker"] = fund_raw["ticker"].astype(str).str.upper()
        fund_raw = fund_raw[fund_raw["ticker"].isin(tickers_set)]
        fund = _derive_fundamentals(fund_raw, prices)
        panel_dates = sorted(panel["date"].unique().tolist())
        fund_daily = _forward_fill_fundamentals(fund, panel_dates)
        if not fund_daily.empty:
            fund_daily = fund_daily[["ticker", "date"] + FUND_COLS]
            panel = panel.merge(fund_daily, how="left", on=["ticker", "date"])
        else:
            for c in FUND_COLS:
                panel[c] = np.nan
    else:
        for c in FUND_COLS:
            panel[c] = np.nan

    # has_fundamentals flag (1 if any of the 7 fund cols is non-null for this ticker)
    panel["has_fundamentals"] = panel[FUND_COLS].notna().any(axis=1).astype(float)

    # Winsorize extreme fundamental ratios using train-slice percentiles only
    # (train slice is approximately the first 65% of dates; compute bounds from
    # the panel rows where 'date' is in the first 65% of unique dates).
    unique_dates = sorted(panel["date"].unique())
    n_train_dates = int(0.65 * len(unique_dates))
    train_cutoff = unique_dates[n_train_dates - 1] if n_train_dates > 0 else unique_dates[-1]
    train_mask_row = panel["date"] <= train_cutoff
    ratio_cols = [
        "cash_runway_q", "rd_intensity", "revenue_growth_yoy",
        "cash_to_mc", "shares_outstanding_yoy", "total_assets_growth",
    ]
    for c in ratio_cols:
        if c not in panel.columns:
            continue
        panel[c] = panel[c].replace([np.inf, -np.inf], np.nan)
        vals_train = panel.loc[train_mask_row, c].dropna()
        if len(vals_train) > 100:
            lo = float(vals_train.quantile(0.01))
            hi = float(vals_train.quantile(0.99))
            panel[c] = panel[c].clip(lower=lo, upper=hi)

    # Sector-median impute fundamentals: group by date, fill NaN with per-date median
    for c in FUND_COLS:
        panel[c] = panel[c].replace([np.inf, -np.inf], np.nan)
        medians = panel.groupby("date")[c].transform("median")
        panel[c] = panel[c].fillna(medians)
    # If the per-date median is itself NaN (rare, first-day edge), fall back to
    # the train-slice global median of that feature (not global to avoid leak)
    for c in FUND_COLS:
        if panel[c].isna().any():
            train_median = panel.loc[train_mask_row, c].median()
            if pd.isna(train_median):
                train_median = 0.0
            panel[c] = panel[c].fillna(train_median)

    # Drop rows with missing target or core price features
    panel = panel.dropna(subset=["fwd_return_h", "log_return", "log_volume", "realized_vol_20d"])
    panel = panel[FEATURE_COLS_CHECK(panel)]  # keep columns sane

    # Winsorize returns
    lo = panel["log_return"].quantile(0.005)
    hi = panel["log_return"].quantile(0.995)
    panel["log_return"] = panel["log_return"].clip(lower=lo, upper=hi)
    panel["log_return_5d"] = panel["log_return_5d"].clip(lower=-0.8, upper=0.8)
    panel["log_return_20d"] = panel["log_return_20d"].clip(lower=-1.5, upper=1.5)

    # Final NaN sweep
    for c in FEATURE_COLS:
        panel[c] = pd.to_numeric(panel[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

    panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)
    dates = sorted(panel["date"].unique().tolist())
    tickers_present = sorted(panel["ticker"].unique().tolist())
    return panel, tickers_present, dates


def FEATURE_COLS_CHECK(panel: pd.DataFrame) -> list[str]:
    """Return the full panel column list, preserving non-feature columns too."""
    return list(panel.columns)


def panel_to_tensors(panel: pd.DataFrame, tickers: list[str], dates: list[pd.Timestamp]) -> dict:
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
