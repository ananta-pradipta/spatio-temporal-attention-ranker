"""Phase 1 delta pull: extend EDGAR fundamentals with LATTICE-specific
intangible and distress concepts.

LATTICE's 4-feature intangible-intensity panel needs:
  rd_to_sales       (have rd_expense and revenue from v2)
  sga_to_sales      (NEED SGAExpense)
  gross_profitability  (gross_profit / total_assets; NEED GrossProfit)
  capex_to_sales    (have capex and revenue from v2)

LATTICE's 4-feature distress-proxy panel needs:
  interest_coverage (NEED InterestExpense)
  net_debt_to_ebitda (have ebit and total_liabilities from v2; NEED Cash if not yet)
  fcf_yield         (have op_cf and capex from v2; market cap computed at panel build)
  current_ratio     (have assets_current and liabilities_current from v2)

Plus LATTICE's StockTwits abnormal-attention features need
AdvertisingExpense (NEED for SG&A decomposition; many firms don't break it
out so it'll be NaN-heavy but worth pulling).

This script pulls 5 new XBRL concepts (SGAExpense, GrossProfit,
InterestExpense, AdvertisingExpense, plus a re-pull of EBIT to verify the
v2 mapping) and writes a parquet that merges with the v2-symlinked
fundamentals_edgar_sp500.parquet at panel-build time.

Output: data/lattice/raw/fundamentals_edgar_sp500_extra.parquet
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm


UA = {
    "User-Agent": "PhD Research adp232@njit.edu",
    "Accept-Encoding": "gzip, deflate",
    "Host": "data.sec.gov",
}
RATE_LIMIT_SEC = 0.12

EXTRA_CONCEPTS = {
    "sga_expense":        ["SellingGeneralAndAdministrativeExpense"],
    "gross_profit":       ["GrossProfit"],
    "interest_expense":   ["InterestExpense", "InterestExpenseDebt"],
    "advertising_expense": ["AdvertisingExpense", "MarketingExpense"],
}


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
                "quarter_end": pd.Timestamp(f["end"]),
                "filed_date":  pd.Timestamp(f.get("filed") or f["end"]),
                "val": float(f["val"]) if f["val"] is not None else None,
            })
    return rows


def fetch_ticker(ticker: str, cik: int) -> pd.DataFrame:
    per_feat: dict[str, pd.DataFrame] = {}
    for feat, concept_list in EXTRA_CONCEPTS.items():
        for concept in concept_list:
            data = fetch_concept(cik, concept)
            if not data:
                continue
            df = pd.DataFrame(data)
            df = df.sort_values(["quarter_end", "filed_date"])
            df = df.drop_duplicates(["quarter_end"], keep="first")
            per_feat[feat] = df[["quarter_end", "filed_date", "val"]].rename(columns={"val": feat})
            break
    if not per_feat:
        return pd.DataFrame()
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
    hist = pd.read_parquet("data/raw/sp500/sp500_constituents_history.parquet")
    sub = hist.drop_duplicates("ticker")
    sub = sub[sub.cik.notna()]
    pairs = list(zip(sub.ticker.tolist(), sub.cik.astype(int).tolist()))
    print(f"Pulling extra fundamentals for {len(pairs)} tickers...")

    frames = []
    for ticker, cik in tqdm(pairs, ncols=80):
        try:
            df = fetch_ticker(ticker, cik)
        except Exception as e:
            print(f"  error for {ticker} (CIK {cik}): {str(e)[:80]}")
            continue
        if not df.empty:
            frames.append(df)

    if not frames:
        print("No fundamentals fetched.")
        return

    out = pd.concat(frames, ignore_index=True)
    out_path = Path("data/lattice/raw/fundamentals_edgar_sp500_extra.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"\nWrote {out_path}: {len(out):,} rows, {out.ticker.nunique()} tickers")
    print("Non-null rate per feature:")
    feature_cols = [c for c in out.columns
                    if c not in ("ticker", "cik", "quarter_end", "filed_date")]
    for c in feature_cols:
        print(f"  {c:25s}  {out[c].notna().mean()*100:5.1f}%")


if __name__ == "__main__":
    main()
