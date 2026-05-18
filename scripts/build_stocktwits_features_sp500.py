"""Pre-aggregate StockTwits messages into per-(ticker, date) features.

Streams the combined biotech + S&P 500 corpora partition-by-partition,
aggregates to daily granularity, and writes a small parquet that the
universal panel builder can read without OOMing.

Output schema matches biotech `stocktwits_features.parquet`:
  ticker, date,
  st_volume_24h, st_volume_change_30d, st_bullish_ratio,
  st_sentiment_dispersion, st_labeled_ratio
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

WINDOW = ("2014-12-01", "2023-01-15")
OUT_PATH = Path("data/processed/stocktwits_features_sp500.parquet")


def _aggregate_partition(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a partition's per-message rows to per-(ticker, date) counts
    and sentiment stats."""
    df = df.rename(columns={"symbol": "ticker"})
    df["created_at"] = pd.to_datetime(df["created_at"])
    df = df[(df.created_at >= WINDOW[0]) & (df.created_at <= WINDOW[1])]
    df["date"] = df.created_at.dt.normalize()
    grp = df.groupby(["ticker", "date"])
    agg = grp.agg(
        n=("sentiment", "size"),
        s_sum=("sentiment", lambda x: x.fillna(0.5).sum()),
        s_sqsum=("sentiment", lambda x: (x.fillna(0.5) ** 2).sum()),
        n_labeled=("sentiment", "count"),
    ).reset_index()
    return agg


def main() -> None:
    biotech = Path("data/raw/stocktwits/symbols.parquet")
    sp500 = Path("data/raw/stocktwits_sp500/symbols.parquet")

    aggs: list[pd.DataFrame] = []
    if biotech.exists():
        print(f"Aggregating biotech corpus ({biotech})...", flush=True)
        df = pd.read_parquet(biotech, columns=["symbol", "created_at", "sentiment"])
        a = _aggregate_partition(df)
        aggs.append(a)
        print(f"  {len(df):,} msgs -> {len(a):,} (ticker, date) rows", flush=True)
        del df

    if sp500.exists():
        parts = sorted(sp500.glob("*.parquet"))
        print(f"Aggregating SP500 corpus ({len(parts)} shards)...", flush=True)
        for i, p in enumerate(parts, 1):
            f = pq.ParquetFile(str(p))
            df = f.read(columns=["symbol", "created_at", "sentiment"]).to_pandas()
            a = _aggregate_partition(df)
            aggs.append(a)
            if i % 20 == 0:
                print(f"  [{i}/{len(parts)}] cum-rows: {sum(len(x) for x in aggs):,}", flush=True)
            del df
        print(f"  total SP500 partitions done", flush=True)

    # Combine partial aggregations -> sum n / s_sum / s_sqsum / n_labeled, then derive features
    print("Reducing across partitions...", flush=True)
    combined = pd.concat(aggs, ignore_index=True)
    del aggs
    final = combined.groupby(["ticker", "date"], as_index=False).agg(
        n=("n", "sum"),
        s_sum=("s_sum", "sum"),
        s_sqsum=("s_sqsum", "sum"),
        n_labeled=("n_labeled", "sum"),
    )
    print(f"Final (ticker, date) rows: {len(final):,}", flush=True)

    # Derive features
    final["st_volume_24h"]    = final["n"]
    final["st_bullish_ratio"] = final["s_sum"] / final["n"].clip(lower=1)
    var = (final["s_sqsum"] / final["n"].clip(lower=1)) - final["st_bullish_ratio"] ** 2
    final["st_sentiment_dispersion"] = np.sqrt(var.clip(lower=0))
    final["st_labeled_ratio"] = final["n_labeled"] / final["n"].clip(lower=1)

    # 30-day rolling volume change per ticker
    final = final.sort_values(["ticker", "date"]).reset_index(drop=True)
    rolling_mean = (
        final.groupby("ticker")["st_volume_24h"]
        .transform(lambda s: s.rolling(30, min_periods=1).mean())
    )
    final["st_volume_change_30d"] = final["st_volume_24h"] / rolling_mean.clip(lower=1)

    # Output schema
    out = final[["ticker", "date", "st_volume_24h", "st_volume_change_30d",
                 "st_bullish_ratio", "st_sentiment_dispersion", "st_labeled_ratio"]]
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH}: {len(out):,} rows, {out.ticker.nunique()} tickers", flush=True)


if __name__ == "__main__":
    main()
