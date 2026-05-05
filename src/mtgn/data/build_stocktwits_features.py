"""Aggregate the biotech-filtered StockTwits corpus into per-stock per-day features.

Input:  data/raw/stocktwits/symbols.parquet  (denormalized per-(message, symbol);
        one row per ticker mentioned in each message. Columns include symbol,
        created_at (date), sentiment, user_id.)

Output: data/processed/stocktwits_features.parquet (long format)
        Columns: ticker, date, st_volume_24h, st_volume_change_30d,
                 st_bullish_ratio, st_sentiment_dispersion, st_labeled_ratio

Feature definitions follow drafts/memorizing-tgn-social-signal-data-sources.md
Section 2.1 (Phase 1 Option A, follower-weighted and account-age-based variants
deferred to Phase 2 because the public StockTwits S3 corpus has anonymous
user_ids with no follower count and no account creation date).

Feature formulas (daily):
    st_volume_24h         = number of ticker mentions on the day
    st_volume_change_30d  = st_volume_24h / rolling_mean(st_volume_24h, 30, end=t-1)
    st_bullish_ratio      = (sentiment == 1).sum() / sentiment.notna().sum()
    st_sentiment_dispersion = std of sentiment over labeled posts on the day
    st_labeled_ratio      = sentiment.notna().sum() / st_volume_24h

Important:
    - The 30-day rolling baseline uses window END at t-1 to avoid look-ahead.
    - Days with zero mentions are OMITTED (sparse long-format output);
      downstream data loader reindexes to the full trading calendar and
      fills as zero.
    - sentiment in the public corpus is float64 with values in {1.0, -1.0, NaN}.

Usage:
    python3 -m src.mtgn.data.build_stocktwits_features \\
        --in data/raw/stocktwits/symbols.parquet \\
        --out data/processed/stocktwits_features.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def build_features(symbols_parquet: Path) -> pd.DataFrame:
    df = pd.read_parquet(
        symbols_parquet,
        columns=["symbol", "created_at", "sentiment"],
    )

    # Normalize types
    df["ticker"] = df["symbol"].astype(str).str.upper()
    raw_date = pd.to_datetime(df["created_at"])

    # Roll weekend posts forward to the next trading day (Monday) so
    # sentiment accumulated Fri-close through Sun-night is credited to
    # Monday's cross-section rather than silently dropped by the later
    # trading-date join. Holidays are not calendar-aware here; the
    # post-trading-day pipeline (panel.build_panel) further filters to
    # actual trading dates so any remaining calendar-day-but-not-trading
    # posts will simply be dropped at that stage.
    wd = raw_date.dt.weekday
    roll = pd.to_timedelta((7 - wd) % 7 * (wd >= 5), unit="D")   # Sat/Sun -> Mon
    df["date"] = (raw_date + roll).dt.date
    df = df.drop(columns=["symbol", "created_at"])

    grp = df.groupby(["ticker", "date"], sort=True)

    # Daily aggregates in a single pass
    daily = grp.agg(
        st_volume_24h=("sentiment", "size"),
        labeled_count=("sentiment", "count"),
        bullish_count=("sentiment", lambda s: int((s == 1.0).sum())),
        st_sentiment_dispersion=("sentiment", lambda s: float(s.std(ddof=0)) if s.count() > 1 else 0.0),
    ).reset_index()

    daily["st_bullish_ratio"] = np.where(
        daily["labeled_count"] > 0,
        daily["bullish_count"] / daily["labeled_count"],
        np.nan,
    )
    daily["st_labeled_ratio"] = np.where(
        daily["st_volume_24h"] > 0,
        daily["labeled_count"] / daily["st_volume_24h"],
        0.0,
    )

    # 30-day rolling baseline per ticker, shifted by 1 day to avoid leakage.
    daily = daily.sort_values(["ticker", "date"]).reset_index(drop=True)
    daily["date"] = pd.to_datetime(daily["date"])
    baselines: list[pd.Series] = []
    for t, sub in daily.groupby("ticker"):
        # reindex to daily calendar within the ticker's observed range so rolling is over actual days, not row positions
        full_idx = pd.date_range(sub["date"].min(), sub["date"].max(), freq="D")
        s = sub.set_index("date")["st_volume_24h"].reindex(full_idx, fill_value=0)
        base = s.rolling(window=30, min_periods=5).mean().shift(1)
        baseline_on_obs = base.reindex(sub["date"]).values
        baselines.append(pd.Series(baseline_on_obs, index=sub.index))
    daily["volume_30d_baseline"] = pd.concat(baselines).sort_index()
    daily["st_volume_change_30d"] = daily["st_volume_24h"] / daily["volume_30d_baseline"]

    # Clean up intermediate columns for the output
    out = daily[[
        "ticker",
        "date",
        "st_volume_24h",
        "st_volume_change_30d",
        "st_bullish_ratio",
        "st_sentiment_dispersion",
        "st_labeled_ratio",
    ]].copy()

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--in",
        dest="inp",
        type=Path,
        default=Path("data/raw/stocktwits/symbols.parquet"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/processed/stocktwits_features.parquet"),
    )
    args = parser.parse_args()

    print(f"Reading {args.inp}")
    features = build_features(args.inp)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(args.out, index=False)
    print(f"Wrote {args.out}")
    print(f"Rows: {len(features):,}")
    print(f"Unique tickers: {features['ticker'].nunique()}")
    print(f"Date range: {features['date'].min()} to {features['date'].max()}")
    print()
    print("Sample:")
    print(features.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
