"""Fetch FDA drug-approval dates from openFDA for the biotech universe.

Source: https://api.fda.gov/drug/drugsfda.json
Queries by sponsor_name for each company in data/processed/ticker_company.parquet
and extracts submission / approval dates.

Output: data/processed/catalyst_fda.parquet
    columns: ticker, date, event_type ('FDA_action'), application_number,
             submission_type, sponsor, product_brand_names
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm


OPENFDA = "https://api.fda.gov/drug/drugsfda.json"


def _sponsor_variants(name: str) -> list[str]:
    """Generate upper-cased variants openFDA indexes under."""
    import re
    if not name:
        return []
    raw = name.upper().strip().rstrip(".")
    # Drop common corporate suffixes iteratively; keep both forms
    stripped = re.sub(
        r"\b(INC|INCORPORATED|CORP|CORPORATION|LTD|LIMITED|LLC|PLC|CO|COMPANY|"
        r"PHARMACEUTICALS|PHARMACEUTICAL|PHARMA|THERAPEUTICS|BIO|BIOSCIENCES|"
        r"HOLDINGS|GROUP|AG|SA|GMBH|NV|SE)\b\.?",
        "",
        raw,
    ).strip()
    stripped = re.sub(r"\s+", " ", stripped)
    variants = [raw, f"{stripped} INC", stripped]
    # dedup while preserving order
    seen = set()
    out = []
    for v in variants:
        if v and v not in seen and len(v) >= 3:
            out.append(v)
            seen.add(v)
    return out


def fetch_sponsor(sponsor: str, limit: int = 100) -> list[dict]:
    """Return openFDA records for any sponsor_name variant."""
    results: list[dict] = []
    for variant in _sponsor_variants(sponsor):
        params = {"search": f'sponsor_name:"{variant}"', "limit": limit}
        try:
            r = requests.get(OPENFDA, params=params, timeout=30)
        except Exception:
            continue
        if r.status_code != 200:
            continue
        recs = r.json().get("results", [])
        if recs:
            results.extend(recs)
            break   # first matching variant wins
    return results


def parse_record(rec: dict, ticker: str) -> list[dict]:
    out: list[dict] = []
    app_no = rec.get("application_number")
    sponsor = rec.get("sponsor_name")
    products = rec.get("products", []) or []
    brand_names = ";".join(
        sorted({p.get("brand_name", "") for p in products if p.get("brand_name")})
    )
    for sub in (rec.get("submissions", []) or []):
        status_date = sub.get("submission_status_date")
        if not status_date:
            continue
        try:
            dt = pd.to_datetime(status_date, format="%Y%m%d", errors="coerce")
        except Exception:
            continue
        if pd.isna(dt):
            continue
        out.append(
            {
                "ticker": ticker,
                "date": dt,
                "event_type": "FDA_action",
                "application_number": app_no,
                "submission_type": sub.get("submission_type"),
                "submission_status": sub.get("submission_status"),
                "sponsor": sponsor,
                "product_brand_names": brand_names,
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ticker-company", type=Path,
        default=Path("data/processed/ticker_company.parquet"),
    )
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2025-04-12")
    parser.add_argument(
        "--out", type=Path, default=Path("data/processed/catalyst_fda.parquet")
    )
    args = parser.parse_args()

    tc = pd.read_parquet(args.ticker_company)
    tc = tc[tc["long_name"].notna()].reset_index(drop=True)
    print(f"Querying openFDA for {len(tc)} tickers ...")

    rows: list[dict] = []
    for _, r in tqdm(tc.iterrows(), total=len(tc), ncols=80):
        ticker = r["ticker"]
        long_name = r.get("long_name") or ""
        if not long_name or len(long_name) < 3:
            continue
        recs = fetch_sponsor(long_name)
        seen_apps: set[str] = set()
        for rec in recs:
            app_no = rec.get("application_number")
            if app_no in seen_apps:
                continue
            seen_apps.add(app_no)
            rows.extend(parse_record(rec, ticker))
        time.sleep(0.05)

    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df = df[(df["date"] >= args.start) & (df["date"] <= args.end)]
        df = df.drop_duplicates(subset=["ticker", "date", "application_number", "submission_type"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"Wrote {args.out}: {len(df):,} FDA-action events, {df['ticker'].nunique()} tickers covered")


if __name__ == "__main__":
    main()
