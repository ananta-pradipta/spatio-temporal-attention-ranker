"""Catalyst calendar for MTGN evaluation diagnostics.

Phase 1 scope for the catalyst-window subset diagnostic: build a per-
(ticker, date) boolean mask that flags days within a window around a
confirmed catalyst event. Phase 1 uses ONLY earnings dates from
yfinance as the catalyst source because they are free, reliable, and
cover all 244 biotech tickers without requiring DrugBank-mediated
ticker-to-drug resolution. FDA PDUFA and clinical trial readouts
augment this in Phase 2.

The memo's pre-declared prediction is: MTGN's gain over vanilla TGN is
largest on catalyst-window days and smallest on calm periods. Even an
earnings-only catalyst signal is enough to test the direction of that
effect.

Output: data/processed/catalyst_days.parquet
    columns: ticker, date, is_catalyst_day
    is_catalyst_day = True iff date is within +/- `window` trading days
    of any earnings event for the ticker.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm


def fetch_earnings_dates(tickers: list[str]) -> pd.DataFrame:
    """Return DataFrame with columns ticker, earnings_date."""
    rows: list[dict] = []
    for t in tqdm(tickers, ncols=80):
        try:
            yt = yf.Ticker(t)
            df = yt.earnings_dates
        except Exception:
            continue
        if df is None or df.empty:
            continue
        df = df.reset_index()
        col = next((c for c in df.columns if "earnings" in str(c).lower() and "date" in str(c).lower()), None)
        if col is None:
            # yfinance usually names it "Earnings Date"; fallback to first datetime col
            col = next((c for c in df.columns if np.issubdtype(df[c].dtype, np.datetime64)), None)
        if col is None:
            continue
        for dt in pd.to_datetime(df[col]).dt.date.dropna().unique():
            rows.append({"ticker": t, "earnings_date": pd.Timestamp(dt)})
    return pd.DataFrame(rows)


def build_catalyst_mask(
    earnings: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
    window_days: int = 5,
) -> pd.DataFrame:
    """Expand each earnings date into a +/- window of trading days."""
    out_rows: list[dict] = []
    tdates_sorted = trading_dates.sort_values()
    for t, sub in earnings.groupby("ticker"):
        marked: set[pd.Timestamp] = set()
        for ev in pd.to_datetime(sub["earnings_date"]).unique():
            # Trading-day window around the event
            idx = tdates_sorted.searchsorted(ev)
            lo = max(0, idx - window_days)
            hi = min(len(tdates_sorted) - 1, idx + window_days)
            for d in tdates_sorted[lo : hi + 1]:
                marked.add(d)
        for d in marked:
            out_rows.append({"ticker": t, "date": d, "is_catalyst_day": True})
    return pd.DataFrame(out_rows).drop_duplicates(["ticker", "date"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--universe", type=Path, default=Path("data/raw/biotech_universe_v1.csv")
    )
    parser.add_argument(
        "--prices", type=Path, default=Path("data/raw/prices_universe.parquet")
    )
    parser.add_argument("--window-days", type=int, default=5)
    parser.add_argument(
        "--out", type=Path, default=Path("data/processed/catalyst_days.parquet")
    )
    args = parser.parse_args()

    u = pd.read_csv(args.universe)
    if "status" in u.columns:
        u = u[u["status"] == "active"]
    tickers = sorted(u["ticker"].dropna().astype(str).str.upper().unique().tolist())
    print(f"Fetching earnings for {len(tickers)} tickers")

    earnings = fetch_earnings_dates(tickers)
    print(f"Retrieved {len(earnings)} earnings-date rows covering {earnings['ticker'].nunique()} tickers")

    prices = pd.read_parquet(args.prices, columns=["date"])
    trading_dates = pd.DatetimeIndex(sorted(pd.to_datetime(prices["date"]).unique()))

    mask = build_catalyst_mask(earnings, trading_dates, window_days=args.window_days)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    mask.to_parquet(args.out, index=False)
    print(f"Wrote {args.out}: {len(mask):,} rows, window=+/-{args.window_days} trading days")


if __name__ == "__main__":
    main()
