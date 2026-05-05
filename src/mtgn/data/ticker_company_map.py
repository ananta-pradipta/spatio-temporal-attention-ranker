"""Build a ticker -> company-name map for the biotech universe.

Used downstream to query ClinicalTrials.gov, openFDA, and EDGAR 8-K
by sponsor / company name.

Output: data/processed/ticker_company.parquet
    columns: ticker, long_name, short_name, industry, sector
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf
from tqdm import tqdm


def fetch_one(ticker: str) -> dict:
    try:
        info = yf.Ticker(ticker).get_info()
    except Exception as e:
        return {"ticker": ticker, "error": f"{type(e).__name__}: {e}"}
    return {
        "ticker": ticker,
        "long_name": info.get("longName"),
        "short_name": info.get("shortName"),
        "industry": info.get("industry"),
        "sector": info.get("sector"),
        "country": info.get("country"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--universe", type=Path, default=Path("data/raw/biotech_universe_v1.csv")
    )
    parser.add_argument(
        "--out", type=Path, default=Path("data/processed/ticker_company.parquet")
    )
    args = parser.parse_args()

    u = pd.read_csv(args.universe)
    if "status" in u.columns:
        u = u[u["status"] == "active"]
    tickers = sorted(u["ticker"].dropna().astype(str).str.upper().unique().tolist())
    print(f"Resolving {len(tickers)} tickers via yfinance get_info()")

    rows = []
    for t in tqdm(tickers, ncols=80):
        rows.append(fetch_one(t))

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)

    n_err = int(df.get("error", pd.Series(dtype=object)).notna().sum())
    n_resolved = int(df["long_name"].notna().sum())
    print(f"Wrote {args.out}: {n_resolved} resolved, {n_err} errors, {len(df)} total")


if __name__ == "__main__":
    main()
