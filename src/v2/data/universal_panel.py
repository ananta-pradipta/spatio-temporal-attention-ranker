"""Universal-panel builder for the S&P 500 validation (Phase 3 of the universal
validation spec).

Parallel to ``src/mtgn/training/panel_enriched.py`` but with three biotech-
specific feature substitutions, dimension-preserving 1:1:

  Position 16  cash_runway_q   -> altman_z_score
  Position 17  rd_intensity    -> capex_intensity
  (rate-sensitivity slot 8)    rolling_xbi_beta_60d -> rolling_sector_etf_beta_60d

The remaining 19 panel features are preserved bit-for-bit; the 22-d shape and
the on-disk feature ordering are unchanged so the existing model code, IPO
bank keys, and gate inputs all read transparently from the universal panel.

Inputs:
    data/raw/sp500/prices_sp500.parquet
    data/raw/sp500/fundamentals_sp500.parquet
    data/raw/sp500/sp500_constituents_history.parquet  (membership + GICS)
    data/raw/stocktwits/symbols.parquet                (biotech-overlap)
    data/raw/stocktwits_sp500/symbols.parquet/         (S&P 500 new pull)

Outputs:
    panel:  pandas DataFrame with the same 22 FEATURE_COLS as panel_enriched
    tickers, dates lists for downstream tensor build.

Active mask gate (per spec 3e): ticker is active on day t if it has a valid
close, a computable 5-day forward log return, >=20 days continuous prior
history, AND is an S&P 500 constituent on day t per the membership table.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.mtgn.training.panel_enriched import (
    PRICE_COLS, ST_COLS, FUND_COLS, FLAG_COLS, FEATURE_COLS,
)


@dataclass
class UniversalPanelConfig:
    """Hyperparameters for the universal-panel build."""

    prices_parquet: Path = Path("data/raw/sp500/prices_sp500.parquet")
    fundamentals_parquet: Path = Path("data/raw/sp500/fundamentals_sp500.parquet")
    constituents_parquet: Path = Path("data/raw/sp500/sp500_constituents_history.parquet")
    stocktwits_features_parquet: Path = Path("data/processed/stocktwits_features_sp500.parquet")
    start_date: str = "2015-01-09"
    end_date: str = "2022-12-31"
    horizon_days: int = 5


# ----------------------------- Price features -------------------------------

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
        s["fwd_return_h"] = np.log(close.shift(-5) / close)
        frames.append(s)
    return pd.concat(frames, ignore_index=True)


# ------------------------- Universal fundamentals --------------------------

def _derive_fundamentals(fund: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Compute the 7 panel-fundamental features from the universal raw fundamentals.

    Substitutions vs biotech panel:
      Position 2: altman_z_score   replaces  cash_runway_q
      Position 3: capex_intensity  replaces  rd_intensity

    Altman Z-score (Altman 1968, public manufacturers):
        Z = 1.2*(WC/TA) + 1.4*(RE/TA) + 3.3*(EBIT/TA) + 0.6*(MV_E/TL) + 1.0*(Sales/TA)

    Where:
        WC      = working capital = assets_current - liabilities_current
        TA      = assets
        RE      = retained_earnings
        EBIT    = ebit (OperatingIncomeLoss)
        MV_E    = market_cap = close[filed_date] * shares
        TL      = total_liabilities
        Sales   = revenue (annualised from the most recent four quarters)

    Capex intensity:
        capex / revenue, winsorised at panel-build time (downstream).
    """
    fund = fund.sort_values(["ticker", "filed_date", "quarter_end"]).reset_index(drop=True).copy()
    prices_small = prices[["ticker", "date", "close"]].copy()
    prices_small["date"] = pd.to_datetime(prices_small["date"]).dt.normalize()
    fund["filed_date"] = pd.to_datetime(fund["filed_date"]).dt.normalize()
    fund["quarter_end"] = pd.to_datetime(fund["quarter_end"]).dt.normalize()

    fund = fund.merge(
        prices_small.rename(columns={"date": "filed_date"}),
        how="left", on=["ticker", "filed_date"],
    )

    def _next_trading_close(sub_fund, sub_prices):
        sub_prices = sub_prices.sort_values("date")
        out = []
        for _, r in sub_fund.iterrows():
            if pd.notna(r.get("close")):
                out.append(r["close"]); continue
            nxt = sub_prices[sub_prices["date"] >= r["filed_date"]]
            out.append(nxt.iloc[0]["close"] if len(nxt) > 0 else np.nan)
        sub_fund = sub_fund.copy()
        sub_fund["close"] = out
        return sub_fund

    frames = []
    for t, sub in fund.groupby("ticker", sort=False):
        sub_prices = prices_small[prices_small["ticker"] == t]
        sub = _next_trading_close(sub, sub_prices)
        sub = sub.sort_values("quarter_end").reset_index(drop=True)
        sub["market_cap"] = sub["close"] * sub["shares"]
        sub["log_market_cap"] = np.log(sub["market_cap"].replace({0: np.nan}))
        sub["date"] = sub["filed_date"]

        # Altman Z components (clip at zero to prevent division issues)
        ta  = sub["assets"].replace({0: np.nan})
        wc  = sub["assets_current"] - sub["liabilities_current"]
        re_ = sub["retained_earnings"]
        ebit = sub["ebit"]
        mv_e = sub["market_cap"]
        tl  = sub["total_liabilities"].replace({0: np.nan})
        # Annualised revenue = trailing 4-quarter sum, falls back to current quarter * 4
        sub["revenue_ttm"] = sub["revenue"].rolling(4, min_periods=1).sum()

        sub["altman_z_score"] = (
            1.2 * (wc   / ta).clip(lower=-5, upper=5)
            + 1.4 * (re_  / ta).clip(lower=-5, upper=5)
            + 3.3 * (ebit / ta).clip(lower=-5, upper=5)
            + 0.6 * (mv_e / tl).clip(lower=0, upper=100)
            + 1.0 * (sub["revenue_ttm"] / ta).clip(lower=0, upper=10)
        )

        # capex_intensity = capex / revenue (clipped non-negative; capex is reported negative on cash flow but our column stores absolute)
        rev = sub["revenue"].replace({0: np.nan})
        sub["capex_intensity"] = (sub["capex"].abs() / rev).clip(lower=0, upper=2.0)

        # Standard biotech panel features (kept identical)
        sub["revenue_growth_yoy"] = sub["revenue"].pct_change(4, fill_method=None)
        sub["cash_to_mc"] = np.where(
            (sub["market_cap"] > 0) & sub["cash"].notna(),
            sub["cash"] / sub["market_cap"], np.nan,
        )
        sub["shares_outstanding_yoy"] = sub["shares"].pct_change(4, fill_method=None)
        sub["total_assets_growth"] = sub["assets"].pct_change(4, fill_method=None)

        # Output the 7 universal FUND_COLS in the SAME on-disk position as biotech.
        # Position 16 (cash_runway_q) is replaced by altman_z_score; we still
        # emit it under the column name `cash_runway_q` so that downstream
        # FEATURE_COLS keys read transparently. Same for rd_intensity ->
        # capex_intensity.
        sub_out = sub[["ticker", "date"]].copy()
        sub_out["log_market_cap"]        = sub["log_market_cap"]
        sub_out["cash_runway_q"]         = sub["altman_z_score"]
        sub_out["rd_intensity"]          = sub["capex_intensity"]
        sub_out["revenue_growth_yoy"]    = sub["revenue_growth_yoy"]
        sub_out["cash_to_mc"]            = sub["cash_to_mc"]
        sub_out["shares_outstanding_yoy"] = sub["shares_outstanding_yoy"]
        sub_out["total_assets_growth"]   = sub["total_assets_growth"]
        frames.append(sub_out)
    return pd.concat(frames, ignore_index=True)


def _forward_fill_fundamentals(fund: pd.DataFrame, dates: list[pd.Timestamp]) -> pd.DataFrame:
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


# --------------------------- StockTwits features ---------------------------

def _load_stocktwits_features(cfg: UniversalPanelConfig) -> pd.DataFrame:
    """Read pre-aggregated per-(ticker, date) StockTwits features.

    Built once by ``scripts/build_stocktwits_features_sp500.py`` to avoid
    holding the full 46.6M-row corpus in memory at panel-build time.
    """
    if not cfg.stocktwits_features_parquet.exists():
        print(f"[universal-panel] WARNING: {cfg.stocktwits_features_parquet} missing; "
              "StockTwits features will be neutral fallbacks for every ticker.",
              flush=True)
        return pd.DataFrame(columns=["ticker", "date"] + ST_COLS)
    st = pd.read_parquet(cfg.stocktwits_features_parquet)
    st["date"] = pd.to_datetime(st["date"])
    return st


# ------------------------------- Build entry --------------------------------

def build_universal_panel(
    cfg: UniversalPanelConfig | None = None,
) -> tuple[pd.DataFrame, list[str], list[pd.Timestamp]]:
    cfg = cfg or UniversalPanelConfig()

    # 1) Constituent universe + GICS sector
    hist = pd.read_parquet(cfg.constituents_parquet)
    hist["ticker"] = hist["ticker"].astype(str).str.upper()
    hist["start_date"] = pd.to_datetime(hist["start_date"])
    hist["end_date"]   = pd.to_datetime(hist["end_date"])
    universe = sorted(hist["ticker"].unique().tolist())
    sector_map = (hist.drop_duplicates("ticker")
                       .set_index("ticker")["gics_sector"].to_dict())

    # 2) Prices + derived features
    prices = pd.read_parquet(cfg.prices_parquet)
    prices["ticker"] = prices["ticker"].astype(str).str.upper()
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices[(prices["date"] >= cfg.start_date) & (prices["date"] <= cfg.end_date)]
    prices = _price_features(prices)
    if cfg.horizon_days != 5:
        prices["fwd_return_h"] = np.log(
            prices.groupby("ticker")["close"].shift(-cfg.horizon_days) / prices["close"]
        )

    # 3) StockTwits per-(ticker, date) features
    st = _load_stocktwits_features(cfg)
    panel = prices.merge(st, how="left", on=["ticker", "date"])
    panel["st_volume_24h"]          = np.log1p(panel["st_volume_24h"].fillna(0.0))
    panel["st_volume_change_30d"]   = np.log1p(panel["st_volume_change_30d"].fillna(1.0).clip(lower=0, upper=1000))
    panel["st_bullish_ratio"]       = panel["st_bullish_ratio"].fillna(0.5)
    panel["st_sentiment_dispersion"] = panel["st_sentiment_dispersion"].fillna(0.0)
    panel["st_labeled_ratio"]       = panel["st_labeled_ratio"].fillna(0.0)

    # 4) Fundamentals: derive, forward-fill, merge, sector-median impute
    fund_raw = pd.read_parquet(cfg.fundamentals_parquet)
    fund_raw["ticker"] = fund_raw["ticker"].astype(str).str.upper()
    fund = _derive_fundamentals(fund_raw, prices)
    panel_dates = sorted(panel["date"].unique().tolist())
    fund_daily = _forward_fill_fundamentals(fund, panel_dates)
    if not fund_daily.empty:
        fund_daily = fund_daily[["ticker", "date"] + FUND_COLS]
        panel = panel.merge(fund_daily, how="left", on=["ticker", "date"])
    else:
        for c in FUND_COLS:
            panel[c] = np.nan

    # has_fundamentals flag
    panel["has_fundamentals"] = panel[FUND_COLS].notna().any(axis=1).astype(float)

    # Winsorise extreme ratios using TRAIN-WINDOW only (first 65% of dates)
    unique_dates = sorted(panel["date"].unique())
    n_train_dates = int(0.65 * len(unique_dates))
    train_cutoff = unique_dates[n_train_dates - 1] if n_train_dates > 0 else unique_dates[-1]
    train_mask_row = panel["date"] <= train_cutoff
    ratio_cols = [
        "cash_runway_q", "rd_intensity", "revenue_growth_yoy",
        "cash_to_mc", "shares_outstanding_yoy", "total_assets_growth",
    ]
    for c in ratio_cols:
        panel[c] = panel[c].replace([np.inf, -np.inf], np.nan)
        vals_train = panel.loc[train_mask_row, c].dropna()
        if len(vals_train) > 100:
            lo = float(vals_train.quantile(0.01))
            hi = float(vals_train.quantile(0.99))
            panel[c] = panel[c].clip(lower=lo, upper=hi)

    # GICS sector-median impute fundamentals (instead of the biotech industry)
    panel["gics_sector"] = panel["ticker"].map(sector_map)
    for c in FUND_COLS:
        panel[c] = panel[c].replace([np.inf, -np.inf], np.nan)
        med = panel.groupby(["date", "gics_sector"])[c].transform("median")
        panel[c] = panel[c].fillna(med)
    # Fall back to per-date global median if sector-median is itself NaN
    for c in FUND_COLS:
        if panel[c].isna().any():
            med = panel.groupby("date")[c].transform("median")
            panel[c] = panel[c].fillna(med)
    # Final fallback: train-window global median
    for c in FUND_COLS:
        if panel[c].isna().any():
            train_med = panel.loc[train_mask_row, c].median()
            if pd.isna(train_med):
                train_med = 0.0
            panel[c] = panel[c].fillna(train_med)

    # 5) Drop rows missing target/core price features
    panel = panel.dropna(subset=["fwd_return_h", "log_return", "log_volume", "realized_vol_20d"])

    # 6) Membership-mask gate: a (ticker, date) row is kept iff at least one of
    # the ticker's [start_date, end_date] intervals covers it. Built per-ticker
    # via numpy interval coverage to avoid an O(N) df.apply.
    panel = panel.reset_index(drop=True)
    intervals = hist[["ticker", "start_date", "end_date"]].copy()
    iv_by_ticker: dict[str, np.ndarray] = {}
    for tk, sub in intervals.groupby("ticker"):
        iv_by_ticker[tk] = np.stack([
            sub["start_date"].astype("int64").to_numpy(),
            sub["end_date"].astype("int64").to_numpy(),
        ])
    panel_dt_int = panel["date"].astype("int64").to_numpy()
    keep = np.zeros(len(panel), dtype=bool)
    for tk, sub in panel.groupby("ticker", sort=False):
        ivs = iv_by_ticker.get(tk)
        if ivs is None or ivs.shape[1] == 0:
            continue
        idx = sub.index.to_numpy()
        d = panel_dt_int[idx]
        # Broadcast: for each row's date d, check if ANY interval covers
        in_iv = (ivs[0][None, :] <= d[:, None]) & (d[:, None] <= ivs[1][None, :])
        keep[idx] = in_iv.any(axis=1)
    panel = panel[keep].copy()

    # 7) Winsorise returns
    lo = panel["log_return"].quantile(0.005)
    hi = panel["log_return"].quantile(0.995)
    panel["log_return"] = panel["log_return"].clip(lower=lo, upper=hi)
    panel["log_return_5d"] = panel["log_return_5d"].clip(lower=-0.8, upper=0.8)
    panel["log_return_20d"] = panel["log_return_20d"].clip(lower=-1.5, upper=1.5)

    # 8) Final NaN sweep
    for c in FEATURE_COLS:
        panel[c] = pd.to_numeric(panel[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

    panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)
    dates = sorted(panel["date"].unique().tolist())
    tickers_present = sorted(panel["ticker"].unique().tolist())
    return panel, tickers_present, dates


def universal_panel_to_tensors(panel: pd.DataFrame, tickers: list[str],
                                dates: list[pd.Timestamp]) -> dict:
    """Same shape contract as panel_enriched.panel_to_tensors."""
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


__all__ = [
    "UniversalPanelConfig",
    "build_universal_panel",
    "universal_panel_to_tensors",
]
