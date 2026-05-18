"""Validate the 51-ticker hand-curated delisting list against SEC EDGAR.

Inputs:
  Hardcoded list of 51 (ticker, year_estimate, reason, deal_partner) tuples
  drawn from agent memory `project_biotech_delistings_research.md`
  (see `--print-source` for the raw memory entry).

Pipeline per ticker:
  1. Resolve ticker -> CIK via SEC's company_tickers.json (handles current
     listings) and a fallback search through the EDGAR submissions API
     (handles historical / delisted tickers).
  2. Pull EDGAR submissions feed for the CIK.
  3. Find the latest filing of any 10-K, 10-Q, 8-K, 25, or 25-NSE form.
  4. Find the latest 8-K specifically (often closer to deal close date for M&A).
  5. Find any Form 25 / 25-NSE filing (canonical delisting marker).
  6. Apply Shumway-Warther defaults for terminal return:
       - performance / Chapter 11: -0.55
       - voluntary / reverse-merger: 0.0
       - M&A: 0.0 placeholder; user can hand-edit deal-specific values

Output: CSV at data/delisting_log_v1.csv with columns:
  ticker, delisting_date, reason, imputed_terminal_return, source_note,
  cik, last_form_25_date, last_8k_date, last_filing_date, deal_partner

Two CSVs are written to make caveats explicit:
  - data/delisting_log_v1.csv     -- minimal schema for the imputation infra
  - data/processed/delisting_log_v1_validation.csv -- diagnostic columns

Usage:
  python3 scripts/validate_delistings_edgar.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import requests


HEADERS = {
    "User-Agent": "PhD Research adp232@njit.edu",
    "Accept-Encoding": "gzip, deflate",
}

# 51 unique tickers from agent memory (2026-05-02 compilation).
# Some appear in both M&A and performance lists; deduplicated by ticker.
# Format: (ticker, approx_year, reason, deal_partner_or_event)
CATALOGUE = [
    # 2015 M&A
    ("CBST", 2015, "MnA", "Merck"),
    ("HSP",  2015, "MnA", "Pfizer"),
    ("PCYC", 2015, "MnA", "AbbVie"),
    ("RCPT", 2015, "MnA", "Celgene"),
    ("GEVA", 2015, "MnA", "Alexion (Synageva)"),
    # 2016 M&A + performance
    ("BIND", 2016, "performance", "Chapter 11"),
    ("MDVN", 2016, "MnA", "Pfizer"),
    ("TBRA", 2016, "MnA", "Allergan (CVR)"),
    ("AFFX", 2016, "MnA", "ThermoFisher"),
    ("ANAC", 2016, "MnA", "Pfizer"),
    # 2017 M&A
    ("ARIA", 2017, "MnA", "Takeda (ARIAD)"),
    ("KITE", 2017, "MnA", "Gilead"),
    # 2017 performance
    ("GALE", 2017, "voluntary", "reverse merger to SLS"),
    # 2018 M&A
    ("SCMP", 2018, "MnA", "Mallinckrodt"),
    ("BIVV", 2018, "MnA", "Sanofi (CVR)"),
    ("JUNO", 2018, "MnA", "Celgene (recycled ticker)"),
    ("TSRO", 2018, "MnA", "GSK (delisted Jan 2019)"),
    # 2018 performance
    ("OREX", 2018, "performance", "Chapter 11"),
    ("ARLZ", 2018, "performance", "Chapter 11"),
    # 2019 M&A
    ("SHPG", 2019, "MnA", "Takeda (Shire ADR)"),
    ("LOXO", 2019, "MnA", "Lilly"),
    ("CELG", 2019, "MnA", "BMS (CVR)"),
    ("ARRY", 2019, "MnA", "Pfizer"),
    # 2019 performance
    ("AKAO", 2019, "performance", "Chapter 11 (Achaogen)"),
    ("INSY", 2019, "performance", "Chapter 11 (Insys/opioid)"),
    ("NUVA", 2019, "performance", "Chapter 11 (Nuvectra)"),
    ("NEXT", 2019, "voluntary", "reverse merger Aevi"),
    # 2020 M&A
    ("MDCO", 2020, "MnA", "Novartis"),
    ("BOLD", 2020, "MnA", "Astellas (Audentes)"),
    ("AGN",  2020, "MnA", "AbbVie (Allergan)"),
    ("FTSV", 2020, "MnA", "Gilead"),
    ("IMMU", 2020, "MnA", "Gilead"),
    ("PRNB", 2020, "MnA", "Sanofi"),
    ("MYOK", 2020, "MnA", "BMS"),
    ("AIMT", 2020, "MnA", "Nestle (Aimmune)"),
    # 2020 performance
    ("ZFGN", 2020, "voluntary", "reverse merger to Larimar/LRMR"),
    # 2021 M&A
    ("GWPH", 2021, "MnA", "Jazz"),
    ("FPRX", 2021, "MnA", "Amgen (Five Prime)"),
    ("TBIO", 2021, "MnA", "Sanofi (Translate)"),
    ("KDMN", 2021, "MnA", "Sanofi"),
    ("TRIL", 2021, "MnA", "Pfizer"),
    ("XLRN", 2021, "MnA", "Merck (Acceleron)"),
    ("DRNA", 2021, "MnA", "NovoNordisk (Dicerna)"),
    # 2022 M&A
    ("ARNA", 2022, "MnA", "Pfizer"),
    ("BHVN", 2022, "MnA", "Pfizer (CGRP biz, complex spinoff)"),
    ("SRRA", 2022, "MnA", "GSK (Sierra)"),
    ("CCXI", 2022, "MnA", "Amgen"),
    ("FMTX", 2022, "MnA", "NovoNordisk (Forma)"),
    ("ZGNX", 2022, "MnA", "UCB (CVR)"),
    ("RDUS", 2022, "MnA", "take-private"),
    # 2022 performance
    ("4DPH", 2022, "performance", "Chapter 11 (4D Pharma)"),
    ("STAB", 2022, "performance", "Chapter 11 (Statera/Cytocom)"),
]

DEFAULT_TERMINAL_RETURN = {
    "performance": -0.55,
    "MnA": 0.0,
    "voluntary": 0.0,
}


def get_ticker_cik_map() -> dict[str, int]:
    """Pull the SEC's authoritative ticker -> CIK mapping (current listings only)."""
    url = "https://www.sec.gov/files/company_tickers.json"
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    data = r.json()
    mapping = {}
    for entry in data.values():
        mapping[entry["ticker"].upper()] = int(entry["cik_str"])
    return mapping


def search_cik_via_efts(ticker: str) -> int | None:
    """Fallback CIK search via EDGAR full-text search (for delisted tickers)."""
    time.sleep(0.15)
    url = "https://efts.sec.gov/LATEST/search-index"
    params = {"q": ticker, "forms": "10-K,8-K", "dateRange": "custom",
              "startdt": "2014-01-01", "enddt": "2023-12-31"}
    try:
        r = requests.get(url, headers=HEADERS, params=params, timeout=30)
        r.raise_for_status()
        hits = r.json().get("hits", {}).get("hits", [])
        for h in hits:
            ciks = h.get("_source", {}).get("ciks", [])
            if ciks:
                return int(ciks[0])
    except Exception:
        return None
    return None


def get_submissions(cik: int) -> dict | None:
    """Pull EDGAR submissions feed for a CIK."""
    time.sleep(0.15)
    url = f"https://data.sec.gov/submissions/CIK{cik:010d}.json"
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def latest_filing_dates(submissions: dict) -> dict[str, str]:
    """Return the latest filing date per (form, all) bucket."""
    if not submissions:
        return {}
    rec = submissions.get("filings", {}).get("recent", {})
    forms = rec.get("form", [])
    dates = rec.get("filingDate", [])
    out: dict[str, str] = {}
    for form, date in zip(forms, dates):
        if form not in out or date > out[form]:
            out[form] = date
    out["any"] = max(dates) if dates else ""
    return out


def main() -> None:
    out_dir = Path("data")
    proc_dir = Path("data/processed")
    proc_dir.mkdir(parents=True, exist_ok=True)

    print(f"Resolving CIKs for {len(CATALOGUE)} tickers...")
    ticker_map = get_ticker_cik_map()
    print(f"  current SEC ticker map: {len(ticker_map)} tickers")

    rows = []
    for i, (tkr, year, reason, partner) in enumerate(CATALOGUE):
        print(f"[{i+1}/{len(CATALOGUE)}] {tkr:<6}  {year}  {reason:<11}  {partner}")
        cik = ticker_map.get(tkr.upper())
        if cik is None:
            cik = search_cik_via_efts(tkr)
        subs = get_submissions(cik) if cik else None
        latest = latest_filing_dates(subs) if subs else {}

        # Best guess for delisting_date in priority order:
        # 1. Form 25 / 25-NSE filing date (canonical)
        # 2. Latest 8-K (often deal-close announcement)
        # 3. Latest 10-K or 10-Q
        # 4. Memory's approximate year + 06-30 (mid-year placeholder)
        form_25 = latest.get("25") or latest.get("25-NSE")
        last_8k = latest.get("8-K")
        last_10k = latest.get("10-K") or latest.get("10-Q")
        any_date = latest.get("any", "")

        if form_25:
            delisting_date = form_25
            source_note = f"EDGAR Form 25 ({partner})"
        elif last_8k and last_8k.startswith(str(year)):
            delisting_date = last_8k
            source_note = f"EDGAR last 8-K ({partner})"
        elif last_10k and last_10k.startswith(str(year)):
            delisting_date = last_10k
            source_note = f"EDGAR last 10-K/Q ({partner})"
        elif any_date:
            delisting_date = any_date
            source_note = f"EDGAR last filing ({partner})"
        else:
            delisting_date = f"{year}-06-30"
            source_note = f"memory year only ({partner})"

        rows.append({
            "ticker": tkr,
            "delisting_date": delisting_date,
            "reason": reason,
            "imputed_terminal_return": DEFAULT_TERMINAL_RETURN[reason],
            "source_note": source_note,
            "cik": cik,
            "last_form_25_date": form_25 or "",
            "last_8k_date": last_8k or "",
            "last_10k_date": last_10k or "",
            "last_any_date": any_date,
            "memory_year": year,
            "deal_partner": partner,
        })

    df = pd.DataFrame(rows)

    # Write the minimal CSV consumed by apply_delisting_imputation
    minimal = df[["ticker", "delisting_date", "reason",
                  "imputed_terminal_return", "source_note"]]
    minimal.to_csv(out_dir / "delisting_log_v1.csv", index=False)
    print(f"\nWrote {out_dir / 'delisting_log_v1.csv'}: {len(minimal)} rows")

    # Write the diagnostic CSV with EDGAR provenance columns
    df.to_csv(proc_dir / "delisting_log_v1_validation.csv", index=False)
    print(f"Wrote {proc_dir / 'delisting_log_v1_validation.csv'}: {len(df)} rows")

    # Report
    print("\n=== Validation summary ===")
    print(f"Resolved CIKs:        {df.cik.notna().sum()}/{len(df)}")
    print(f"Form 25 filed:        {(df.last_form_25_date != '').sum()}")
    print(f"8-K date matches yr:  {df.apply(lambda r: r.last_8k_date.startswith(str(r.memory_year)), axis=1).sum()}")
    print(f"Source-note breakdown:")
    print(df.source_note.str.split(' \\(').str[0].value_counts().to_dict())

    print("\nFirst 10 rows of the structured CSV:")
    print(minimal.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
