"""Phase 2b: pull SEC EDGAR XBRL fundamentals for the 685 S&P 500 historical
constituents, including the line items required for the Altman Z-score.

Extends src.mtgn.data.fundamentals_edgar with these additional concepts:
    working_capital     = AssetsCurrent - LiabilitiesCurrent
                          (computed from concept totals; both required)
    retained_earnings  -> RetainedEarningsAccumulatedDeficit (fallback:
                          RetainedEarningsUnappropriated)
    ebit               -> OperatingIncomeLoss (fallback: IncomeLossFromContinuingOperationsBeforeInterestExpense)
    total_liabilities  -> Liabilities
    capex              -> PaymentsToAcquirePropertyPlantAndEquipment

Output: data/raw/sp500/fundamentals_sp500.parquet
Schema: ticker, cik, quarter_end, filed_date,
        cash, assets, shares, revenue, rd_expense, net_income, op_cf,
        assets_current, liabilities_current, retained_earnings, ebit,
        total_liabilities, capex

Usage:
  PYTHONPATH=/home/apradipta/phd-research python3 scripts/pull_sp500_fundamentals.py
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

# Universal-panel concepts (match the original biotech pipeline order)
CONCEPTS = {
    "cash":         ["CashAndCashEquivalentsAtCarryingValue",
                     "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
                     "Cash"],
    "assets":       ["Assets"],
    "shares":       ["CommonStockSharesOutstanding",
                     "CommonStockSharesIssued",
                     "EntityCommonStockSharesOutstanding"],
    "revenue":      ["Revenues",
                     "RevenueFromContractWithCustomerExcludingAssessedTax",
                     "SalesRevenueNet"],
    "rd_expense":   ["ResearchAndDevelopmentExpense",
                     "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost"],
    "net_income":   ["NetIncomeLoss"],
    "op_cf":        ["NetCashProvidedByUsedInOperatingActivities"],
    # Altman Z line items (new):
    "assets_current":      ["AssetsCurrent"],
    "liabilities_current": ["LiabilitiesCurrent"],
    "retained_earnings":   ["RetainedEarningsAccumulatedDeficit",
                            "RetainedEarningsUnappropriated"],
    "ebit":                ["OperatingIncomeLoss",
                            "IncomeLossFromContinuingOperationsBeforeInterestExpenseInterestIncomeIncomeTaxesExtraordinaryItemsNoncontrollingInterestsNet"],
    "total_liabilities":   ["Liabilities"],
    "capex":               ["PaymentsToAcquirePropertyPlantAndEquipment"],
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
                "form": form,
            })
    return rows


def fetch_ticker(ticker: str, cik: int) -> pd.DataFrame:
    per_feat: dict[str, pd.DataFrame] = {}
    for feat, concept_list in CONCEPTS.items():
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
    print(f"Pulling EDGAR fundamentals for {len(pairs)} tickers (CIK known)...")

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
    out_path = Path("data/raw/sp500/fundamentals_sp500.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"\nWrote {out_path}: {len(out):,} rows, {out.ticker.nunique()} tickers")
    print(f"  quarter_end range: {out.quarter_end.min()} to {out.quarter_end.max()}")
    print(f"  non-null per feature:")
    feature_cols = [c for c in out.columns if c not in ("ticker", "cik", "quarter_end", "filed_date")]
    for c in feature_cols:
        if c in out.columns:
            print(f"    {c:22s}  {out[c].notna().mean()*100:5.1f}%")


if __name__ == "__main__":
    main()
