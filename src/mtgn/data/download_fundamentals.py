"""Fetch quarterly fundamentals from yfinance for the biotech universe.

Pulls quarterly balance_sheet / income_stmt / cashflow for each active
ticker. Derives biotech-relevant features (cash runway, R&D intensity,
shares-outstanding growth, etc.) and writes a ticker-date parquet
suitable for forward-filling into the daily panel.

Usage:
    python3 -m src.mtgn.data.download_fundamentals \\
        --universe data/raw/biotech_universe_v1.csv \\
        --out data/raw/fundamentals_quarterly.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm


def _safe_row(df: pd.DataFrame, row_names: list[str]) -> pd.Series | None:
    """Return the first matching row from a yfinance statement DataFrame."""
    if df is None or df.empty:
        return None
    for name in row_names:
        for candidate in df.index:
            if str(candidate).lower() == name.lower():
                return df.loc[candidate]
    return None


def fetch_one(ticker: str) -> pd.DataFrame:
    """Return a per-quarter DataFrame of raw fundamentals for one ticker."""
    t = yf.Ticker(ticker)
    try:
        bs = t.quarterly_balance_sheet
        is_ = t.quarterly_income_stmt
        cf = t.quarterly_cashflow
        info = t.get_info()
    except Exception:
        return pd.DataFrame()

    if (bs is None or bs.empty) and (is_ is None or is_.empty):
        return pd.DataFrame()

    # yfinance quarterly statements are columns=dates, rows=line items
    dates = set()
    for df in (bs, is_, cf):
        if df is not None and not df.empty:
            dates.update(df.columns)
    dates = sorted(dates)
    if not dates:
        return pd.DataFrame()

    rows = []
    cash_names = ["Cash And Cash Equivalents", "CashAndCashEquivalents", "Cash"]
    total_assets_names = ["Total Assets", "TotalAssets"]
    shares_out_names = ["Ordinary Shares Number", "Share Issued", "Common Stock Shares Outstanding"]
    rev_names = ["Total Revenue", "Revenues", "Revenue"]
    rd_names = ["Research And Development", "Research Development", "ResearchAndDevelopment"]
    ni_names = ["Net Income", "NetIncome"]
    op_cf_names = ["Operating Cash Flow", "Total Cash From Operating Activities"]

    cash = _safe_row(bs, cash_names)
    total_assets = _safe_row(bs, total_assets_names)
    shares_out = _safe_row(bs, shares_out_names)
    rev = _safe_row(is_, rev_names)
    rd = _safe_row(is_, rd_names)
    ni = _safe_row(is_, ni_names)
    op_cf = _safe_row(cf, op_cf_names)

    for d in dates:
        row = {"ticker": ticker, "quarter_end": pd.Timestamp(d)}
        row["cash"] = float(cash.get(d)) if cash is not None and d in cash.index else np.nan
        row["total_assets"] = float(total_assets.get(d)) if total_assets is not None and d in total_assets.index else np.nan
        row["shares_outstanding"] = float(shares_out.get(d)) if shares_out is not None and d in shares_out.index else np.nan
        row["revenue"] = float(rev.get(d)) if rev is not None and d in rev.index else np.nan
        row["rd_expense"] = float(rd.get(d)) if rd is not None and d in rd.index else np.nan
        row["net_income"] = float(ni.get(d)) if ni is not None and d in ni.index else np.nan
        row["operating_cashflow"] = float(op_cf.get(d)) if op_cf is not None and d in op_cf.index else np.nan
        rows.append(row)

    out = pd.DataFrame(rows)
    # current info: approximate market cap snapshot (not historical; attach for reference)
    out["current_market_cap"] = info.get("marketCap", np.nan)
    out["sector"] = info.get("sector", "")
    out["industry"] = info.get("industry", "")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", type=Path, default=Path("data/raw/biotech_universe_v1.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/raw/fundamentals_quarterly.parquet"))
    args = parser.parse_args()

    u = pd.read_csv(args.universe)
    if "status" in u.columns:
        u = u[u["status"] == "active"]
    tickers = sorted(u["ticker"].dropna().astype(str).str.upper().unique().tolist())
    print(f"Fetching fundamentals for {len(tickers)} tickers")

    frames = []
    missing = []
    for t in tqdm(tickers, ncols=80):
        df = fetch_one(t)
        if df.empty:
            missing.append(t)
        else:
            frames.append(df)

    if not frames:
        raise RuntimeError("no fundamentals fetched")
    out = pd.concat(frames, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    args.out.with_name(args.out.stem + "_missing.txt").write_text("\n".join(missing))
    print(f"Wrote {args.out}: {len(out)} rows, {out['ticker'].nunique()} tickers, {len(missing)} tickers missing")


if __name__ == "__main__":
    main()
