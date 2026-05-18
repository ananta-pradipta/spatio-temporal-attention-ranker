"""Phase 1c: pull EDGAR XBRL fundamentals for the 7 usable delisted tickers.

CIKs were resolved manually (mix of company_tickers.json hits and EFTS
name-search) since several catalogued tickers have been recycled and the
naive ticker-to-CIK lookup returns a different live company.

Output: `data/raw/fundamentals_edgar_delisted.parquet` with the same
schema as `fundamentals_edgar.parquet` (ticker, cik, quarter_end,
filed_date, cash, assets, shares, revenue, rd_expense, net_income, op_cf).

Usage:
  python3 scripts/pull_delisted_fundamentals.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.mtgn.data.fundamentals_edgar import CONCEPTS, fetch_ticker  # noqa: E402


# Hand-verified CIKs for the 7 delisted tickers with usable price data.
# Sources: SCMP/SHPG/STAB from EDGAR ticker-CIK map; BIVV/JUNO/TSRO/NEXT from
# EFTS company-name search (recycled tickers had wrong CIK in company_tickers.json).
CIK_MAP = {
    "SHPG": 936402,    # Shire plc (Takeda acquired 2019)
    "TSRO": 1491576,   # Tesaro (GSK acquired 2019-01)
    "JUNO": 1594864,   # Juno Therapeutics (Celgene acquired 2018)
    "SCMP": 1365216,   # Sucampo (Mallinckrodt acquired 2018)
    "NEXT": 1138776,   # Aevi Genomic Medicine
    "BIVV": 1681689,   # Bioverativ (Sanofi acquired 2018)
    "STAB": 1318641,   # Statera Biopharma / Cytocom (Chapter 11 2022)
}


def main() -> None:
    out_path = Path("data/raw/fundamentals_edgar_delisted.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    for ticker, cik in CIK_MAP.items():
        print(f"[{ticker}] CIK {cik:010d}")
        try:
            df = fetch_ticker(ticker, cik)
        except Exception as e:
            print(f"  error: {e}")
            continue
        if df.empty:
            print("  no fundamentals returned")
            continue
        # Rename to match fundamentals_quarterly schema column names
        rename = {
            "shares": "shares_outstanding",
            "rd_expense": "rd_expense",
            "op_cf": "operating_cashflow",
            "assets": "total_assets",
        }
        df = df.rename(columns=rename)
        n_q = len(df)
        q_first = df.quarter_end.min()
        q_last = df.quarter_end.max()
        print(f"  {n_q} quarters  ({q_first.date()} -> {q_last.date()})")
        frames.append(df)

    if not frames:
        print("\nNo fundamentals fetched. Exiting.")
        return

    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(out_path, index=False)
    print(f"\nWrote {out_path}: {len(out):,} rows, {out.ticker.nunique()} tickers")
    print(f"  quarter_end range: {out.quarter_end.min().date()} to {out.quarter_end.max().date()}")
    print(f"  filed_date range:  {out.filed_date.min().date()} to {out.filed_date.max().date()}")
    print("\nNon-null rate per feature:")
    feature_cols = [c for c in out.columns if c not in {"ticker", "cik", "quarter_end", "filed_date"}]
    for c in feature_cols:
        print(f"  {c:22s}  {out[c].notna().mean() * 100:5.1f}%")

    print("\nPer-ticker quarter count:")
    counts = out.groupby("ticker").quarter_end.agg(["count", "min", "max"])
    counts.columns = ["n_quarters", "first_q", "last_q"]
    print(counts.sort_values("n_quarters", ascending=False).to_string())


if __name__ == "__main__":
    main()
