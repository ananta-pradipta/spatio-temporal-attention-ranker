"""Fetch historical quarterly fundamentals from SEC EDGAR XBRL.

Uses `data.sec.gov/api/xbrl/companyconcept/...` endpoint. Covers
2009-present for most US-listed filers. Quarterly data from 10-Q and
10-K filings.

Fundamentals pulled (US-GAAP concept names):
  - Cash                        -> CashAndCashEquivalentsAtCarryingValue
  - Total assets                -> Assets
  - Shares outstanding          -> CommonStockSharesOutstanding (fallback: Issued)
  - Revenue                     -> Revenues / RevenueFromContractWithCustomerExcludingAssessedTax
  - R&D expense                 -> ResearchAndDevelopmentExpense
  - Net income                  -> NetIncomeLoss
  - Operating cash flow         -> NetCashProvidedByUsedInOperatingActivities

Usage:
    python3 -m src.mtgn.data.fundamentals_edgar \\
        --universe data/raw/biotech_universe_v1.csv \\
        --out data/raw/fundamentals_edgar.parquet
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm


UA = {
    "User-Agent": "NJIT MTGN Research (adp232@njit.edu)",
    "Accept-Encoding": "gzip, deflate",
    "Host": "data.sec.gov",
}
SEC_COMPANY_TICKERS = "https://www.sec.gov/files/company_tickers.json"
RATE_LIMIT_SEC = 0.12   # ~8 req/s; SEC allows 10.

# Concept fallbacks (try first available)
CONCEPTS = {
    "cash":         ["CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents", "Cash"],
    "assets":       ["Assets"],
    "shares":       ["CommonStockSharesOutstanding", "CommonStockSharesIssued", "EntityCommonStockSharesOutstanding"],
    "revenue":      ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"],
    "rd_expense":   ["ResearchAndDevelopmentExpense", "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost"],
    "net_income":   ["NetIncomeLoss"],
    "op_cf":        ["NetCashProvidedByUsedInOperatingActivities"],
}


def load_ticker_cik_map() -> dict[str, int]:
    headers = {"User-Agent": UA["User-Agent"]}
    r = requests.get(SEC_COMPANY_TICKERS, headers=headers, timeout=30)
    r.raise_for_status()
    data = r.json()
    # Keyed by row index; each entry has cik_str, ticker, title
    mapping: dict[str, int] = {}
    for _, row in data.items():
        t = str(row.get("ticker", "")).upper()
        cik = int(row.get("cik_str"))
        if t:
            mapping[t] = cik
    return mapping


def fetch_concept(cik: int, concept: str) -> list[dict]:
    url = f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik:010d}/us-gaap/{concept}.json"
    try:
        r = requests.get(url, headers=UA, timeout=30)
    except Exception:
        return []
    time.sleep(RATE_LIMIT_SEC)
    if r.status_code != 200:
        return []
    d = r.json()
    rows: list[dict] = []
    for currency, vals in (d.get("units") or {}).items():
        for f in vals:
            form = f.get("form", "")
            if not (form.startswith("10-Q") or form.startswith("10-K")):
                continue
            rows.append({
                "quarter_end": pd.Timestamp(f["end"]),         # period end (reference)
                "filed_date":  pd.Timestamp(f.get("filed") or f["end"]),  # PUBLIC-FROM date
                "val": float(f["val"]) if f["val"] is not None else None,
                "form": form,
                "fy": f.get("fy"),
                "fp": f.get("fp"),
                "currency": currency,
            })
    return rows


def fetch_ticker(ticker: str, cik: int) -> pd.DataFrame:
    """Return per-filing quarterly fundamentals with filing date preserved.

    Output schema:
        ticker, cik, quarter_end, filed_date, <each CONCEPT key>

    One row per unique (ticker, filed_date) combination. The filed_date
    is the "known-from" public availability; downstream forward-fill
    must be indexed on filed_date, NOT quarter_end, to avoid look-ahead.
    """
    per_feat: dict[str, pd.DataFrame] = {}
    for feat, concept_list in CONCEPTS.items():
        for concept in concept_list:
            data = fetch_concept(cik, concept)
            if not data:
                continue
            df = pd.DataFrame(data)
            # Prefer the ORIGINAL (earliest) filing for each quarter_end — amendments
            # (10-K/A, 10-Q/A) land months or years later and would inflate filing
            # lag, but the original 10-K/10-Q is what was publicly available first.
            df = df.sort_values(["quarter_end", "filed_date"])
            df = df.drop_duplicates(["quarter_end"], keep="first")
            per_feat[feat] = df[["quarter_end", "filed_date", "val"]].rename(columns={"val": feat})
            break
    if not per_feat:
        return pd.DataFrame()

    # Merge on quarter_end (filing timestamps can differ slightly across concepts
    # for the same quarter; use the MAX filed_date across concepts for that quarter
    # to ensure we don't use any value before it was truly available).
    merged = None
    filed_date_map: dict[pd.Timestamp, pd.Timestamp] = {}
    for feat, df in per_feat.items():
        for _, r in df.iterrows():
            q = r["quarter_end"]; fd = r["filed_date"]
            filed_date_map[q] = max(filed_date_map.get(q, fd), fd)
        small = df[["quarter_end", feat]]
        merged = small if merged is None else merged.merge(small, how="outer", on="quarter_end")
    merged = merged.sort_values("quarter_end").reset_index(drop=True)
    merged["filed_date"] = merged["quarter_end"].map(filed_date_map)
    merged["ticker"] = ticker
    merged["cik"] = cik
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", type=Path, default=Path("data/raw/biotech_universe_v1.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/raw/fundamentals_edgar.parquet"))
    args = parser.parse_args()

    u = pd.read_csv(args.universe)
    if "status" in u.columns:
        u = u[u["status"] == "active"]
    tickers = sorted(u["ticker"].dropna().astype(str).str.upper().unique().tolist())

    print(f"Loading SEC ticker->CIK map...")
    tc_map = load_ticker_cik_map()
    hits = [(t, tc_map[t]) for t in tickers if t in tc_map]
    misses = [t for t in tickers if t not in tc_map]
    print(f"  resolved: {len(hits)} / {len(tickers)}  (missing: {len(misses)})")

    frames = []
    for ticker, cik in tqdm(hits, ncols=80):
        try:
            df = fetch_ticker(ticker, cik)
        except Exception:
            continue
        if not df.empty:
            frames.append(df)

    if not frames:
        raise RuntimeError("No fundamentals fetched")
    out = pd.concat(frames, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)

    miss_path = args.out.with_name(args.out.stem + "_missing.txt")
    miss_path.write_text("\n".join(misses))
    print(f"\nWrote {args.out}: {len(out):,} rows, {out['ticker'].nunique()} tickers")
    print(f"  quarter_end range: {out['quarter_end'].min()} to {out['quarter_end'].max()}")
    print(f"  filed_date range:  {out['filed_date'].min()} to {out['filed_date'].max()}")
    print(f"  non-null rate per feature:")
    for c in CONCEPTS:
        if c in out.columns:
            print(f"    {c:12s}  {out[c].notna().mean() * 100:5.1f}%")


if __name__ == "__main__":
    main()
