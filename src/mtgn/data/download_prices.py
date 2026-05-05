"""Download daily OHLCV for the full MTGN biotech universe from Yahoo Finance.

Reads tickers from `data/raw/biotech_universe_v1.csv` (or v2 active subset),
pulls adjusted OHLCV via `yfinance.download`, and writes a single long-format
parquet at `data/raw/prices_universe.parquet` with columns:

    ticker, date, open, high, low, close, adj_close, volume

Tickers with no returned data (delisted, never traded, yfinance miss) are
logged to `data/raw/prices_universe_missing.txt` and skipped.

Usage:
    python3 -m src.mtgn.data.download_prices \\
        --universe data/raw/biotech_universe_v1.csv \\
        --start 2019-01-01 --end 2025-04-12 \\
        --out data/raw/prices_universe.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf
from tqdm import tqdm


def load_tickers(universe_csv: Path) -> list[str]:
    df = pd.read_csv(universe_csv)
    if "status" in df.columns:
        df = df[df["status"] == "active"]
    tickers = (
        df["ticker"].dropna().astype(str).str.strip().str.upper().drop_duplicates().tolist()
    )
    if not tickers:
        raise ValueError(f"No tickers loaded from {universe_csv}")
    return tickers


def fetch_one(ticker: str, start: str, end: str) -> pd.DataFrame:
    raw = yf.download(
        ticker,
        start=start,
        end=end,
        progress=False,
        auto_adjust=False,
        actions=False,
    )
    if raw.empty:
        return pd.DataFrame()
    # yfinance may return MultiIndex columns for single-ticker requests.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )
    raw = raw.reset_index().rename(columns={"Date": "date"})
    raw["ticker"] = ticker
    cols = ["ticker", "date", "open", "high", "low", "close", "adj_close", "volume"]
    for c in cols:
        if c not in raw.columns:
            raw[c] = None
    return raw[cols]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", type=Path, default=Path("data/raw/biotech_universe_v1.csv"))
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2025-04-12")
    parser.add_argument("--out", type=Path, default=Path("data/raw/prices_universe.parquet"))
    args = parser.parse_args()

    tickers = load_tickers(args.universe)
    print(f"Fetching OHLCV for {len(tickers)} tickers from {args.start} to {args.end}...")

    frames: list[pd.DataFrame] = []
    missing: list[str] = []
    for t in tqdm(tickers, ncols=80):
        try:
            df = fetch_one(t, args.start, args.end)
        except Exception as e:
            missing.append(f"{t}\tERROR\t{type(e).__name__}: {e}")
            continue
        if df.empty:
            missing.append(f"{t}\tEMPTY")
            continue
        frames.append(df)

    if not frames:
        raise RuntimeError("No ticker returned data")
    all_df = pd.concat(frames, ignore_index=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    all_df.to_parquet(args.out, index=False)

    miss_path = args.out.with_name(args.out.stem + "_missing.txt")
    miss_path.write_text("\n".join(missing) + ("\n" if missing else ""))

    print()
    print(f"Wrote {args.out}: {len(all_df):,} rows across {all_df['ticker'].nunique()} tickers.")
    print(f"Missing / empty: {len(missing)} tickers (see {miss_path})")
    print(f"Date coverage: {all_df['date'].min()} to {all_df['date'].max()}")


if __name__ == "__main__":
    main()
