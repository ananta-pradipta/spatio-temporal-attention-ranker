"""Pull biotech delistings from SEC EDGAR to mitigate survivorship bias.

Form 25 (Notification of Removal From Listing) is the official filing that
marks a security's delisting. This script queries EDGAR's full-text search
for Form 25 filings in the training window, filters to biotech SIC codes,
extracts tickers and delisting dates, and merges with the current-membership
biotech universe to produce `data/raw/biotech_universe_v2.csv` with both
active and delisted tickers.

Biotech SIC codes used:
    2833 Medicinal Chemicals and Botanical Products
    2834 Pharmaceutical Preparations
    2835 In Vitro and In Vivo Diagnostic Substances
    2836 Biological Products (Except Diagnostic Substances)
    8731 Commercial Physical and Biological Research

EDGAR full-text search endpoint: https://efts.sec.gov/LATEST/search-index
EDGAR company submissions endpoint: https://data.sec.gov/submissions/CIK<cik>.json

Both require a descriptive User-Agent per SEC fair-access policy.

Usage:
    python3 -m src.mtgn.data.edgar_delistings \\
        --start 2020-01-01 --end 2025-04-12
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests


BIOTECH_SIC = ("2833", "2834", "2835", "2836", "8731")

EDGAR_SEARCH = "https://efts.sec.gov/LATEST/search-index"
EDGAR_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik:010d}.json"

UA = {
    # SEC's fair-access policy requires a meaningful UA with a real contact email.
    "User-Agent": "NJIT MTGN Research (adp232@njit.edu)",
    "Accept-Encoding": "gzip, deflate",
}

RATE_LIMIT_PER_SEC = 5   # SEC guideline: 10 req/sec cap; stay well below.


def _throttle() -> None:
    time.sleep(1.0 / RATE_LIMIT_PER_SEC)


def _get_json(url: str, params: dict, max_retries: int = 5) -> dict:
    """GET with exponential backoff on 5xx and connection errors."""
    delay = 2.0
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            r = requests.get(url, params=params, headers=UA, timeout=45)
            if r.status_code >= 500:
                raise requests.HTTPError(f"{r.status_code} for {r.url}")
            r.raise_for_status()
            _throttle()
            return r.json()
        except (requests.RequestException, ValueError) as e:
            last_err = e
            time.sleep(delay)
            delay = min(delay * 2, 30.0)
    raise RuntimeError(f"EDGAR request failed after {max_retries} retries: {last_err}")


def _search_form25_window(start_date: str, end_date: str) -> Iterable[dict]:
    """Paginate Form 25 filings within a single narrow date window.

    efts has a practical pagination ceiling; chunk date ranges into
    short windows (e.g. a quarter) to stay below it.
    """
    frm = 0
    while True:
        params = {
            "q": "",
            "forms": "25,25-NSE",
            "dateRange": "custom",
            "startdt": start_date,
            "enddt": end_date,
            "from": frm,
        }
        data = _get_json(EDGAR_SEARCH, params)
        hits = data.get("hits", {}).get("hits", [])
        if not hits:
            break
        for h in hits:
            yield h
        total = data.get("hits", {}).get("total", {}).get("value", 0)
        frm += len(hits)
        if frm >= total:
            break
        if frm >= 9900:   # efts cap is 10,000; narrow the window instead
            print(f"  WARN: hit efts cap in {start_date} to {end_date}; narrow the window")
            break


def search_form25(start_date: str, end_date: str) -> Iterable[dict]:
    """Yield Form 25 filings across the full window by chunking into quarters."""
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    # quarter-by-quarter windows
    cur = start
    while cur <= end:
        q_end = (cur + pd.offsets.QuarterEnd(0)).normalize()
        if q_end > end:
            q_end = end
        print(f"  window: {cur.date()} to {q_end.date()}")
        for hit in _search_form25_window(cur.date().isoformat(), q_end.date().isoformat()):
            yield hit
        cur = q_end + pd.Timedelta(days=1)


def get_company_info(cik: int) -> dict:
    """Fetch company submissions JSON for SIC code + ticker lookup."""
    r = requests.get(EDGAR_SUBMISSIONS.format(cik=cik), headers=UA, timeout=30)
    _throttle()
    if r.status_code == 404:
        return {}
    r.raise_for_status()
    return r.json()


def extract_delistings(
    start_date: str,
    end_date: str,
    sic_allowlist: tuple[str, ...] = BIOTECH_SIC,
    max_filings: int | None = None,
) -> pd.DataFrame:
    """Return a DataFrame of (ticker, cik, sic, delisting_date) for biotech delistings."""
    rows: list[dict] = []
    seen_ciks: dict[int, dict] = {}
    for i, hit in enumerate(search_form25(start_date, end_date)):
        if max_filings is not None and i >= max_filings:
            break
        source = hit.get("_source", {})
        ciks = source.get("ciks", [])
        filed = source.get("file_date")
        for cik_str in ciks:
            try:
                cik = int(cik_str)
            except ValueError:
                continue
            if cik not in seen_ciks:
                try:
                    seen_ciks[cik] = get_company_info(cik)
                except requests.HTTPError:
                    seen_ciks[cik] = {}
            info = seen_ciks[cik]
            sic = str(info.get("sic", ""))
            if sic not in sic_allowlist:
                continue
            # `tickers` is usually a list of exchange tickers (could be empty for delisted).
            tickers = info.get("tickers") or []
            for t in (tickers or [None]):
                rows.append(
                    {
                        "ticker": t,
                        "cik": cik,
                        "sic": sic,
                        "delisting_filed": filed,
                        "company": info.get("name"),
                        "former_names": ";".join(
                            fn.get("name", "") for fn in info.get("formerNames", []) or []
                        ),
                    }
                )
    return pd.DataFrame(rows).drop_duplicates()


def merge_with_universe(delistings: pd.DataFrame, universe_csv: Path) -> pd.DataFrame:
    """Produce biotech_universe_v2: v1 is trusted as current-membership truth.

    Form 25 filings are not always company-wide delistings; they are also
    filed for retiring specific share classes, ADR conversions, and similar
    transactions. If a ticker appears in v1 (derived from current XBI / IBB
    holdings), trust v1: the company is actively trading, do not mark it
    delisted. Only tickers NOT in v1 are added as new `delisted` rows.
    """
    v1 = pd.read_csv(universe_csv)
    v1["status"] = v1["status"].fillna("active")
    v1_tickers = set(v1["ticker"].dropna().astype(str).str.upper())

    new_rows = []
    seen_new: set[str] = set()
    for _, r in delistings.iterrows():
        t = r.get("ticker")
        if pd.isna(t) or not t:
            continue
        t = str(t).upper()
        if t in v1_tickers:
            # Active ETF member: keep v1 status. Do not touch.
            continue
        if t in seen_new:
            continue
        seen_new.add(t)
        new_rows.append(
            {
                "ticker": t,
                "source_etf": "edgar_delisting",
                "status": "delisted",
                "first_seen": None,
                "last_seen": r["delisting_filed"],
                "cik": r.get("cik"),
                "sic": r.get("sic"),
                "company": r.get("company"),
            }
        )
    v2 = pd.concat([v1, pd.DataFrame(new_rows)], ignore_index=True)
    return v2.drop_duplicates("ticker").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2025-04-12")
    parser.add_argument("--universe", type=Path, default=Path("data/raw/biotech_universe_v1.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/raw/biotech_universe_v2.csv"))
    parser.add_argument("--max-filings", type=int, default=None, help="Cap for dry runs")
    args = parser.parse_args()

    print(f"Querying EDGAR Form 25 filings from {args.start} to {args.end}...")
    delistings = extract_delistings(
        args.start, args.end, max_filings=args.max_filings
    )
    print(f"Biotech delistings found: {len(delistings)}")

    delist_path = args.out.with_name("biotech_delistings.csv")
    delistings.to_csv(delist_path, index=False)
    print(f"Wrote {delist_path}")

    v2 = merge_with_universe(delistings, args.universe)
    v2.to_csv(args.out, index=False)
    n_active = (v2["status"] == "active").sum()
    n_delisted = (v2["status"] == "delisted").sum()
    print(f"Wrote {args.out}: {len(v2)} tickers ({n_active} active, {n_delisted} delisted).")


if __name__ == "__main__":
    main()
