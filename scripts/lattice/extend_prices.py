"""Extend prices_sp500.parquet forward to 2025-12-31 (or current date).

Reads the existing prices_sp500.parquet (2015-01 to 2023-01) plus the
EXTENDED constituents history. For each ticker that has a membership
interval after 2023-01-15, pulls additional yfinance OHLCV bars from
the max-existing-date+1 to today.

Writes the merged result to a new parquet:
  data/raw/sp500/prices_sp500_extended.parquet

Usage:
  cd $HOME/phd-research
  source ~/phd-research-gpu-env/bin/activate
  python3 scripts/lattice/extend_prices.py
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

PRICES_PATH = Path("data/raw/sp500/prices_sp500.parquet")
CONSTITUENTS_EXTENDED = Path("data/raw/sp500/sp500_constituents_history_extended.parquet")
OUT_PATH = Path("data/raw/sp500/prices_sp500_extended.parquet")


def fetch_yfinance(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    import yfinance as yf
    try:
        df = yf.download(
            ticker, start=start, end=end,
            auto_adjust=False, progress=False, threads=False,
        )
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]
        df["ticker"] = ticker
        df = df.rename(columns={"adj close": "adj_close"})
        return df[["ticker", "date", "open", "high", "low", "close", "adj_close", "volume"]]
    except Exception as e:
        print(f"  yfinance error for {ticker}: {str(e)[:80]}", flush=True)
        return None


def main() -> None:
    existing = pd.read_parquet(PRICES_PATH)
    existing["date"] = pd.to_datetime(existing["date"])
    existing_max_per_ticker = (
        existing.groupby("ticker")["date"].max().to_dict()
    )
    print(f"[ext] existing prices: {len(existing):,} rows, "
          f"{existing['ticker'].nunique()} tickers, "
          f"max date {existing['date'].max().date()}", flush=True)

    cons = pd.read_parquet(CONSTITUENTS_EXTENDED)
    cons_tickers = sorted(set(cons["ticker"].astype(str).str.strip().tolist()))
    print(f"[ext] extended constituents: {len(cons_tickers)} unique tickers",
          flush=True)

    today = pd.Timestamp.today().normalize()
    end_str = today.strftime("%Y-%m-%d")

    new_panels = [existing]
    n_pulled = 0
    n_missing = 0
    for i, t in enumerate(cons_tickers, 1):
        start = existing_max_per_ticker.get(t, pd.Timestamp("2015-01-01"))
        # Pull starting the day after the existing max.
        pull_start = (start + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        if pull_start >= end_str:
            continue
        df = fetch_yfinance(t, pull_start, end_str)
        if df is None or df.empty:
            n_missing += 1
            if n_missing < 20:
                print(f"  [{i}/{len(cons_tickers)}] {t}: no new data from {pull_start}", flush=True)
            continue
        new_panels.append(df)
        n_pulled += 1
        if i % 50 == 0:
            print(f"  [{i}/{len(cons_tickers)}] pulled={n_pulled} missing={n_missing}",
                  flush=True)
        time.sleep(0.05)

    merged = pd.concat(new_panels, ignore_index=True)
    merged["date"] = pd.to_datetime(merged["date"])
    # Drop duplicates: a ticker-date pair may exist if a later pull starts
    # at an overlapping boundary.
    merged = (
        merged.drop_duplicates(subset=["ticker", "date"])
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(OUT_PATH)
    print(f"\n[ext] WROTE {OUT_PATH}: {len(merged):,} rows, "
          f"{merged['ticker'].nunique()} tickers, "
          f"date range {merged['date'].min().date()} to {merged['date'].max().date()}",
          flush=True)
    print(f"[ext] tickers with new data: {n_pulled}", flush=True)
    print(f"[ext] tickers with no new data: {n_missing}", flush=True)


if __name__ == "__main__":
    main()
