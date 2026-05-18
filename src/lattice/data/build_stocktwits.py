"""LATTICE Phase 1 StockTwits features.

Per spec Section 5.1 (StockTwits generic, 5 features):

    st_volume_24h_log         log1p of message count per (ticker, day)
    st_volume_abnormal_z60d   today's log volume minus 60d rolling mean,
                              divided by 60d rolling std (per-ticker)
    st_sentiment_dispersion   weighted within-day cross-user variance of
                              bullish/bearish tags (NaN-filled with 0.0
                              when n_messages < 5)
    st_labeled_ratio          fraction of messages with explicit bull/bear
                              tags (NaN-filled with prior-30-day ticker mean)
    st_bullish_ratio_demeaned today's bullish ratio minus the ticker-60d
                              rolling mean and minus the sector-day mean

This builder reuses the v2 universal-validation pre-aggregated parquet
(data/processed/stocktwits_features_sp500.parquet) as input source,
which already has st_volume_24h, st_volume_change_30d, st_bullish_ratio,
st_sentiment_dispersion, st_labeled_ratio per (ticker, date).

Output: data/lattice/processed/stocktwits_features.parquet
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def build_stocktwits_features(
    raw_st_path: Path = Path("data/processed/stocktwits_features_sp500.parquet"),
    sp500_constituents_path: Path = Path("data/lattice/raw/sp500_constituents_pit.parquet"),
    out_path: Path = Path("data/lattice/processed/stocktwits_features.parquet"),
) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    st = pd.read_parquet(raw_st_path)
    st["ticker"] = st["ticker"].astype(str).str.upper()
    st["date"] = pd.to_datetime(st["date"]).dt.normalize()
    print(f"[lattice stocktwits] loaded {len(st):,} (ticker, date) rows", flush=True)

    # Sector map for sector-day demeaning
    hist = pd.read_parquet(sp500_constituents_path)
    sector_map = (hist.drop_duplicates("ticker")
                       .set_index("ticker")["gics_sector"].to_dict())
    st["sector"] = st["ticker"].map(sector_map)

    st = st.sort_values(["ticker", "date"]).reset_index(drop=True)

    # 1. log volume
    st["st_volume_24h_log"] = np.log1p(st["st_volume_24h"])

    # 2. abnormal volume z-score (60d per-ticker)
    grp = st.groupby("ticker", sort=False)["st_volume_24h_log"]
    rolling_mean = grp.transform(lambda x: x.rolling(60, min_periods=10).mean())
    rolling_std = grp.transform(lambda x: x.rolling(60, min_periods=10).std())
    st["st_volume_abnormal_z60d"] = ((st["st_volume_24h_log"] - rolling_mean)
                                       / rolling_std.clip(lower=0.01)).clip(lower=-5, upper=5)
    st["st_volume_abnormal_z60d"] = st["st_volume_abnormal_z60d"].fillna(0.0)

    # 3. dispersion: keep v2's value but NaN-fill where n < 5
    st["st_sentiment_dispersion"] = np.where(
        st["st_volume_24h"] >= 5, st["st_sentiment_dispersion"], 0.0,
    )

    # 4. labeled ratio: NaN-fill with ticker-30d prior mean
    grp_lab = st.groupby("ticker", sort=False)["st_labeled_ratio"]
    prior_mean = grp_lab.transform(lambda x: x.rolling(30, min_periods=1).mean().shift(1))
    st["st_labeled_ratio_filled"] = st["st_labeled_ratio"].fillna(prior_mean).fillna(0.0)

    # 5. bullish ratio demeaned (ticker-60d rolling mean + sector-day mean)
    grp_bull = st.groupby("ticker", sort=False)["st_bullish_ratio"]
    bull_60d_mean = grp_bull.transform(lambda x: x.rolling(60, min_periods=5).mean())
    sector_day_mean = (st.groupby(["sector", "date"])["st_bullish_ratio"]
                          .transform("mean"))
    st["st_bullish_ratio_demeaned"] = (
        st["st_bullish_ratio"] - bull_60d_mean.fillna(0.5) - (sector_day_mean - 0.5)
    ).fillna(0.0)

    out = st[["ticker", "date",
              "st_volume_24h_log", "st_volume_abnormal_z60d",
              "st_sentiment_dispersion", "st_labeled_ratio_filled",
              "st_bullish_ratio_demeaned"]].rename(
        columns={"st_labeled_ratio_filled": "st_labeled_ratio"})

    out.to_parquet(out_path, index=False)
    print(f"[lattice stocktwits] wrote {out_path}: {len(out):,} rows, "
          f"{out.ticker.nunique()} tickers", flush=True)
    return {
        "rows": len(out),
        "tickers": out.ticker.nunique(),
        "dates": out.date.nunique(),
    }


__all__ = ["build_stocktwits_features"]
