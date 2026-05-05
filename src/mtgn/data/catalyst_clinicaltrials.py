"""Fetch clinical-trial primary-completion dates from ClinicalTrials.gov for
the biotech universe.

Uses the CT.gov API v2 (https://clinicaltrials.gov/api/v2/studies). Searches
by sponsor / lead-sponsor name derived from the ticker -> company map.
Pulls trials with primary-completion date in a configurable window and
records (ticker, date, event_type='trial_readout') rows.

Fan-out note: ~275 tickers, several hundred trials per active biotech,
pagination 100/page. Budget: ~2-5 min at default pageSize. No API key.

Output: data/processed/catalyst_trials.parquet
    columns: ticker, date, event_type, nct_id, phase, sponsor, study_title
"""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
from tqdm import tqdm


CT_API = "https://clinicaltrials.gov/api/v2/studies"

UA = {"User-Agent": "NJIT MTGN Research (adp232@njit.edu)"}


def _sponsor_queries(company_name: str) -> list[str]:
    """Generate a small set of sponsor-name aliases to query.

    CT.gov `sponsor` search is exact-substring. A company listed as
    'Amgen Inc.' in yfinance might appear as 'Amgen' or 'Amgen Inc' at
    CT.gov. We try a few natural variants.
    """
    if not company_name or not isinstance(company_name, str):
        return []
    name = company_name.strip()
    name_no_suffix = re.sub(r"\s+(Inc|Inc\.|Ltd|Ltd\.|LLC|Corp|Corp\.|Company|Co|Co\.|plc|PLC|Pharmaceuticals|Pharma|Pharmaceutical|Therapeutics|Bio|Biosciences|Holdings)\.?$", "", name).strip()
    candidates = {name, name_no_suffix}
    # also split on commas and take the leading chunk
    if "," in name:
        candidates.add(name.split(",")[0].strip())
    return sorted({c for c in candidates if c and len(c) >= 3})


def fetch_studies_for_sponsor(
    sponsor: str, start_date: str, end_date: str, page_size: int = 100, sleep: float = 0.1
) -> list[dict]:
    """Return raw study dicts from CT.gov matching sponsor and date window."""
    rows: list[dict] = []
    next_token: str | None = None
    for _ in range(20):
        params = {
            "query.lead": f'"{sponsor}"',
            "filter.advanced": (
                f"AREA[PrimaryCompletionDate]RANGE[{start_date},{end_date}]"
            ),
            "fields": "protocolSection.identificationModule,"
                      "protocolSection.sponsorCollaboratorsModule,"
                      "protocolSection.designModule,"
                      "protocolSection.statusModule",
            "pageSize": page_size,
        }
        if next_token:
            params["pageToken"] = next_token
        r = requests.get(CT_API, params=params, headers=UA, timeout=30)
        if r.status_code != 200:
            break
        data = r.json()
        rows.extend(data.get("studies", []))
        next_token = data.get("nextPageToken")
        if not next_token:
            break
        time.sleep(sleep)
    return rows


def parse_study(study: dict) -> dict | None:
    ps = study.get("protocolSection", {})
    ident = ps.get("identificationModule", {})
    design = ps.get("designModule", {})
    status = ps.get("statusModule", {})
    spons = ps.get("sponsorCollaboratorsModule", {}).get("leadSponsor", {})
    pcd = status.get("primaryCompletionDateStruct", {}).get("date")
    if not pcd:
        return None
    return {
        "nct_id": ident.get("nctId"),
        "study_title": ident.get("briefTitle"),
        "phase": ",".join(design.get("phases", []) or []),
        "primary_completion_date": pcd,
        "sponsor": spons.get("name"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ticker-company", type=Path,
        default=Path("data/processed/ticker_company.parquet"),
    )
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2025-04-12")
    parser.add_argument(
        "--out", type=Path, default=Path("data/processed/catalyst_trials.parquet")
    )
    args = parser.parse_args()

    tc = pd.read_parquet(args.ticker_company)
    tc = tc[tc["long_name"].notna()].reset_index(drop=True)
    print(f"Querying CT.gov for {len(tc)} tickers ...")

    rows: list[dict] = []
    for _, r in tqdm(tc.iterrows(), total=len(tc), ncols=80):
        ticker = r["ticker"]
        for sponsor in _sponsor_queries(r.get("long_name") or r.get("short_name", "")):
            try:
                studies = fetch_studies_for_sponsor(sponsor, args.start, args.end)
            except Exception:
                continue
            for study in studies:
                parsed = parse_study(study)
                if parsed is None:
                    continue
                pcd = pd.to_datetime(parsed["primary_completion_date"], errors="coerce")
                if pd.isna(pcd) or not (pd.Timestamp(args.start) <= pcd <= pd.Timestamp(args.end)):
                    continue
                rows.append(
                    {
                        "ticker": ticker,
                        "date": pcd,
                        "event_type": "trial_readout",
                        "nct_id": parsed["nct_id"],
                        "phase": parsed["phase"],
                        "sponsor": parsed["sponsor"],
                        "study_title": parsed["study_title"],
                    }
                )
            # No need to try more aliases if we already got hits
            if rows and rows[-1]["ticker"] == ticker:
                break

    df = pd.DataFrame(rows).drop_duplicates(subset=["ticker", "date", "nct_id"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"Wrote {args.out}: {len(df):,} trial-readout events, {df['ticker'].nunique()} tickers covered")


if __name__ == "__main__":
    main()
