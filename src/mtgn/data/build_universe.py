"""Assemble the MTGN biotech ticker universe from free sources.

Phase 1 strategy (while WRDS / CRSP is unavailable):
  1. Download current holdings from SPDR XBI and iShares IBB.
  2. Union with the 38-ticker pilot seed.
  3. Flag each ticker with its source ETF(s) and first-seen date (today
     for survivors; user to back-fill historical entries for delistings).
  4. Write data/raw/biotech_universe_v1.csv.

The result is the "current-membership" universe. Historical delisting
supplementation via SEC EDGAR is a separate TODO (scripts/edgar_delistings.py,
not implemented yet).

Usage:
    python3 -m src.mtgn.data.build_universe
"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import requests


XBI_URL = (
    "https://www.ssga.com/us/en/individual/library-content/products/"
    "fund-data/etfs/us/holdings-daily-us-en-xbi.xlsx"
)

# iShares holdings endpoint for IBB. fileType=csv returns CSV-ish with a metadata prefix.
IBB_URL = (
    "https://www.ishares.com/us/products/239699/ishares-nasdaq-biotechnology-etf/"
    "1467271812596.ajax?fileType=csv&fileName=IBB_holdings&dataType=fund"
)

UA = {"User-Agent": "Mozilla/5.0 (research) Python-requests"}


def _find_header_row(lines: list[str], token: str) -> int:
    for i, ln in enumerate(lines):
        if token.lower() in ln.lower():
            return i
    raise RuntimeError(f"Header token '{token}' not found in CSV")


def fetch_xbi_holdings() -> pd.DataFrame:
    """SSGA publishes XBI as an xlsx; the first few rows are metadata."""
    r = requests.get(XBI_URL, headers=UA, timeout=30)
    r.raise_for_status()
    # openpyxl can read from bytes
    all_sheets = pd.read_excel(io.BytesIO(r.content), sheet_name=None, header=None, engine="openpyxl")
    frames = []
    for name, df in all_sheets.items():
        # find the header row containing "Ticker"
        for i, row in df.iterrows():
            if any(isinstance(v, str) and v.strip().lower() == "ticker" for v in row.values):
                body = df.iloc[i + 1 :].copy()
                body.columns = [str(v).strip() if isinstance(v, str) else v for v in df.iloc[i].values]
                body = body.dropna(axis=1, how="all")
                frames.append(body)
                break
    if not frames:
        raise RuntimeError("XBI: no Ticker header found in any sheet")
    xbi = pd.concat(frames, ignore_index=True)
    # Normalize column name for ticker
    tcol = next(c for c in xbi.columns if isinstance(c, str) and c.strip().lower() == "ticker")
    xbi = xbi.rename(columns={tcol: "ticker"})
    xbi["ticker"] = xbi["ticker"].astype(str).str.strip().str.upper()
    xbi = xbi[xbi["ticker"].str.len().between(1, 6)]
    xbi["source_etf"] = "XBI"
    return xbi[["ticker", "source_etf"]].drop_duplicates("ticker")


def fetch_ibb_holdings() -> pd.DataFrame:
    """iShares returns a CSV with ~9 rows of metadata before the actual header."""
    r = requests.get(IBB_URL, headers=UA, timeout=30)
    r.raise_for_status()
    text = r.content.decode("utf-8-sig", errors="replace")
    lines = text.splitlines()
    header_idx = _find_header_row(lines, "Ticker")
    body = "\n".join(lines[header_idx:])
    df = pd.read_csv(io.StringIO(body))
    # IBB CSV uses "Ticker" column
    tcol = next(c for c in df.columns if c.strip().lower() == "ticker")
    df = df.rename(columns={tcol: "ticker"})
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df = df[df["ticker"].str.len().between(1, 6)]
    df["source_etf"] = "IBB"
    return df[["ticker", "source_etf"]].drop_duplicates("ticker")


def main() -> None:
    print("Fetching XBI holdings...")
    try:
        xbi = fetch_xbi_holdings()
        print(f"  XBI: {len(xbi)} tickers")
    except Exception as e:
        print(f"  XBI FAIL: {type(e).__name__}: {e}")
        xbi = pd.DataFrame(columns=["ticker", "source_etf"])

    print("Fetching IBB holdings...")
    try:
        ibb = fetch_ibb_holdings()
        print(f"  IBB: {len(ibb)} tickers")
    except Exception as e:
        print(f"  IBB FAIL: {type(e).__name__}: {e}")
        ibb = pd.DataFrame(columns=["ticker", "source_etf"])

    seed_path = Path("src/mtgn/data/xbi_proxy_tickers.txt")
    seed = pd.DataFrame(
        {"ticker": [t.strip().upper() for t in seed_path.read_text().splitlines() if t.strip()]}
    )
    seed["source_etf"] = "seed38"

    all_tickers = pd.concat([xbi, ibb, seed], ignore_index=True)
    agg = (
        all_tickers.groupby("ticker")["source_etf"]
        .apply(lambda s: ",".join(sorted(set(s))))
        .reset_index()
    )
    agg["status"] = "active"
    agg["first_seen"] = pd.Timestamp.utcnow().date().isoformat()
    agg["last_seen"] = agg["first_seen"]
    agg = agg.sort_values("ticker").reset_index(drop=True)

    out = Path("data/raw/biotech_universe_v1.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(out, index=False)

    print()
    print(f"Wrote {out}")
    print(f"Total unique tickers: {len(agg)}")
    src = agg["source_etf"]
    in_xbi = src.str.contains("XBI")
    in_ibb = src.str.contains("IBB")
    in_seed = src.str.contains("seed38")
    print(f"  in XBI (any):        {in_xbi.sum()}")
    print(f"  in IBB (any):        {in_ibb.sum()}")
    print(f"  in XBI ∩ IBB:        {(in_xbi & in_ibb).sum()}")
    print(f"  in seed38 (any):     {in_seed.sum()}")
    print(f"  XBI only (no IBB):   {(in_xbi & ~in_ibb).sum()}")
    print(f"  IBB only (no XBI):   {(in_ibb & ~in_xbi).sum()}")
    print()
    print("Next step: supplement delisted biotech tickers in the training")
    print("window via SEC EDGAR Form 25 / 8-K search. Not automated yet.")


if __name__ == "__main__":
    main()
