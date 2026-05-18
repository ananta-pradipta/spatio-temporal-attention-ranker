"""Phase 5b C: build the earnings calendar for the LATTICE S&P 500 panel.

Per spec section 6.1 with ambiguity-3 resolution: pull earnings dates via
``yfinance.Ticker(symbol).get_earnings_dates(limit=100)`` for every
ever-S&P-500 ticker (n=685). Restrict to events in
[2014-12-01, 2023-01-31] (the panel window plus 30-day buffers). Save the
union to ``data/lattice/raw/earnings_calendar.parquet``. EDGAR 8-K Item
2.02 fallback is built only if the cell-level coverage drops below 95% of
panel cells (i.e., > 5% of (date, ticker) panel cells lack a "next earnings
event within the next 90 trading days" classification).

Usage::

    python3 -m scripts.lattice.build_earnings_calendar
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


WINDOW_START = pd.Timestamp("2014-12-01")
WINDOW_END = pd.Timestamp("2023-01-31")
LIMIT = 100
SLEEP_BETWEEN = 0.4


def pull_one(symbol: str, limit: int = LIMIT) -> pd.DataFrame:
    """Pull earnings dates for a single ticker.

    Returns an empty frame if yfinance has no record or rate-limits us.
    """
    import yfinance as yf
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.get_earnings_dates(limit=limit)
    except Exception as exc:
        print(f"[earnings pull] {symbol} FAILED: {type(exc).__name__}: {exc}", flush=True)
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.reset_index().rename(columns={"Earnings Date": "event_date"})
    df["ticker"] = symbol
    df["event_type"] = "earnings"
    df["source"] = "yfinance"
    df["pulled_at"] = datetime.now(timezone.utc)
    return df[["ticker", "event_date", "event_type", "source", "pulled_at"]]


def filter_window(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows with event_date in [WINDOW_START, WINDOW_END]."""
    if df.empty:
        return df
    dates = pd.to_datetime(df["event_date"], utc=True)
    naive = dates.dt.tz_localize(None) if dates.dt.tz is not None else dates
    df = df.copy()
    df["event_date"] = naive
    return df[(naive >= WINDOW_START) & (naive <= WINDOW_END)]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--constituents", type=str,
                    default="data/lattice/raw/sp500_constituents_pit.parquet")
    p.add_argument("--out", type=str,
                    default="data/lattice/raw/earnings_calendar.parquet")
    p.add_argument("--cache", type=str,
                    default="data/lattice/raw/earnings_calendar_progress.parquet",
                    help="Per-ticker partial cache; resumable.")
    p.add_argument("--limit", type=int, default=LIMIT)
    p.add_argument("--sleep", type=float, default=SLEEP_BETWEEN)
    p.add_argument("--start-from", type=str, default=None,
                    help="Resume from a specific ticker (alphabetical).")
    args = p.parse_args()

    constituents = pd.read_parquet(args.constituents)
    tickers = sorted(constituents["ticker"].dropna().unique().tolist())
    print(f"[earnings pull] found {len(tickers)} ever-tickers", flush=True)

    cache_path = Path(args.cache)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        cache_df = pd.read_parquet(cache_path)
        done_tickers = set(cache_df["ticker"].unique())
        print(f"[earnings pull] resuming; {len(done_tickers)} tickers already cached",
              flush=True)
    else:
        cache_df = pd.DataFrame(
            columns=["ticker", "event_date", "event_type", "source", "pulled_at"],
        )
        done_tickers = set()

    if args.start_from:
        tickers = [t for t in tickers if t >= args.start_from]

    n_pulled = 0
    no_dates_tickers: list[str] = []
    for i, symbol in enumerate(tickers):
        if symbol in done_tickers:
            continue
        df = pull_one(symbol, limit=args.limit)
        if df.empty:
            no_dates_tickers.append(symbol)
        else:
            df = filter_window(df)
            cache_df = pd.concat([cache_df, df], ignore_index=True)
            n_pulled += 1
            if (n_pulled % 25) == 0:
                cache_df.to_parquet(cache_path, index=False)
                print(f"[earnings pull] {i + 1}/{len(tickers)} processed; "
                      f"{n_pulled} new rows added; cache flushed", flush=True)
        time.sleep(args.sleep)

    cache_df.to_parquet(cache_path, index=False)
    cache_df.to_parquet(args.out, index=False)
    print(f"[earnings pull] done: {len(cache_df)} total rows, "
          f"{cache_df['ticker'].nunique()} tickers with at least one event",
          flush=True)
    print(f"[earnings pull] {len(no_dates_tickers)} tickers had no events; "
          f"first 20: {no_dates_tickers[:20]}", flush=True)


if __name__ == "__main__":
    main()
