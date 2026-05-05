"""Merge earnings + clinical trials + FDA into a unified catalyst calendar.

Output: data/processed/catalyst_days_full.parquet
    columns: ticker, date, event_type, is_catalyst_day
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--earnings", type=Path, default=Path("data/processed/catalyst_days.parquet"))
    parser.add_argument("--trials", type=Path, default=Path("data/processed/catalyst_trials.parquet"))
    parser.add_argument("--fda", type=Path, default=Path("data/processed/catalyst_fda.parquet"))
    parser.add_argument("--prices", type=Path, default=Path("data/raw/prices_universe.parquet"))
    parser.add_argument("--window-days", type=int, default=5)
    parser.add_argument("--out", type=Path, default=Path("data/processed/catalyst_days_full.parquet"))
    args = parser.parse_args()

    frames: list[pd.DataFrame] = []

    if args.earnings.exists():
        e = pd.read_parquet(args.earnings)
        e["date"] = pd.to_datetime(e["date"])
        e["event_type"] = "earnings"
        frames.append(e[["ticker", "date", "event_type"]])

    if args.trials.exists():
        t = pd.read_parquet(args.trials)
        t["date"] = pd.to_datetime(t["date"])
        if "event_type" not in t.columns:
            t["event_type"] = "trial_readout"
        frames.append(t[["ticker", "date", "event_type"]])

    if args.fda.exists():
        f = pd.read_parquet(args.fda)
        f["date"] = pd.to_datetime(f["date"])
        if "event_type" not in f.columns:
            f["event_type"] = "FDA_action"
        frames.append(f[["ticker", "date", "event_type"]])

    events = pd.concat(frames, ignore_index=True)
    events["ticker"] = events["ticker"].astype(str).str.upper()
    events = events.dropna(subset=["date"])

    # Expand each event into +/- window_days trading-day range
    prices = pd.read_parquet(args.prices, columns=["date"])
    trading = pd.DatetimeIndex(sorted(pd.to_datetime(prices["date"]).unique()))

    out_rows: list[dict] = []
    for (ticker, event_type), sub in events.groupby(["ticker", "event_type"]):
        marked: dict[pd.Timestamp, str] = {}
        for ev in pd.to_datetime(sub["date"]).unique():
            idx = trading.searchsorted(ev)
            lo = max(0, idx - args.window_days)
            hi = min(len(trading) - 1, idx + args.window_days)
            for d in trading[lo : hi + 1]:
                marked[d] = event_type
        for d, et in marked.items():
            out_rows.append({"ticker": ticker, "date": d, "event_type": et, "is_catalyst_day": True})

    out = pd.DataFrame(out_rows)
    # If the same cell is flagged by multiple event types, keep one with priority
    priority = {"FDA_action": 0, "trial_readout": 1, "M_and_A": 2, "partnership": 3, "earnings": 4}
    out["__pri"] = out["event_type"].map(priority).fillna(99)
    out = out.sort_values(["ticker", "date", "__pri"]).drop_duplicates(["ticker", "date"], keep="first")
    out = out.drop(columns="__pri")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)

    n_ticker_days_total = len(prices) * events["ticker"].nunique()  # upper bound
    print(f"Wrote {args.out}: {len(out):,} flagged (ticker, date) cells")
    print(f"  tickers flagged: {out['ticker'].nunique()}")
    print(f"  event-type distribution:")
    print(out["event_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
