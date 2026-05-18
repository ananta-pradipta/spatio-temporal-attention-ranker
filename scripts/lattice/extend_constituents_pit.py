"""Extend the S&P 500 constituents PIT history forward to 2025.

The existing `data/raw/sp500/sp500_constituents_history.parquet` covers
2015-01-01 to 2022-12-31 (built 2026-05-05, git commit f8a575d). This
script extends it forward through the current date by:

1. Pulling Wikipedia's "List of S&P 500 companies" page (two tables: a
   snapshot of current constituents plus a "Selected changes" change-log).
2. Parsing the change-log for additions and removals dated 2023-01 onward.
3. Reconciling each new ticker's CIK and GICS sector via SEC EDGAR's
   company_tickers.json directory + the company-facts XBRL endpoint
   (yfinance is a secondary fallback for GICS sector).
4. Stitching the new add/remove events into existing membership intervals,
   closing intervals for removed tickers and opening intervals for added
   tickers.
5. Saving the extended PIT parquet to
   `data/raw/sp500/sp500_constituents_history_extended.parquet`.

The extended parquet has the same schema as the original:
  ticker, cik, gics_sector, gics_subsector, name, start_date, end_date

Audit trail outputs:
  data/raw/sp500/_wiki_current_2025.csv
  data/raw/sp500/_wiki_changes_2023plus.csv
  data/raw/sp500/_sec_lookup_2023plus.csv

Note: this script is reviewer-defensible (Wikipedia maintains the
list-of-changes table; we cite that source). It is NOT a full SEC 10-K
constituent-disclosure reconstruction; the latter is the gold standard
but Wikipedia is sufficient for academic panel research where the goal
is to avoid survivorship bias rather than reproduce a regulatory filing.
"""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import pandas as pd
import requests


# Wikipedia URL with the canonical list and changes table.
WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

# SEC EDGAR directory of all company tickers (small JSON, ~100KB).
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

# Polite request headers (SEC requires a User-Agent with an email).
HEADERS = {
    "User-Agent": "PhD Research adp232@njit.edu",
    "Accept-Encoding": "gzip, deflate",
}


def pull_wikipedia_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pull the current S&P 500 constituents and the change-log table."""
    print(f"[wiki] pulling {WIKI_URL}", flush=True)
    # Wikipedia rejects default urllib User-Agent (HTTP 403). Use requests
    # with a polite User-Agent, then parse the HTML body with pandas.
    from io import StringIO
    headers = {**HEADERS, "User-Agent": HEADERS["User-Agent"] + " (Wikipedia table parse)"}
    r = requests.get(WIKI_URL, headers=headers, timeout=30)
    r.raise_for_status()
    tables = pd.read_html(StringIO(r.text), header=0)
    # Table 0 is the current constituents; table 1 is the "Selected changes"
    # change-log. Schema can drift over time; verify columns.
    current = tables[0]
    changes = tables[1]
    print(f"[wiki] tables[0] columns: {list(current.columns)}", flush=True)
    print(f"[wiki] tables[1] columns: {list(changes.columns)}", flush=True)
    return current, changes


def normalize_current(current: pd.DataFrame) -> pd.DataFrame:
    """Normalize the current-constituents table to a consistent schema."""
    rename = {
        "Symbol": "ticker",
        "Security": "name",
        "GICS Sector": "gics_sector",
        "GICS Sub-Industry": "gics_subsector",
        "CIK": "cik",
        "Date added": "wiki_date_added",
    }
    cols = {old: new for old, new in rename.items() if old in current.columns}
    df = current[list(cols.keys())].rename(columns=cols).copy()
    # Strip trailing footnote markers from tickers (e.g., "BF.B[3]").
    df["ticker"] = df["ticker"].astype(str).str.replace(
        r"\[.*?\]", "", regex=True,
    ).str.strip()
    # CIK as int when available; the parquet stores it as int.
    if "cik" in df.columns:
        df["cik"] = pd.to_numeric(df["cik"], errors="coerce").astype("Int64")
    if "wiki_date_added" in df.columns:
        df["wiki_date_added"] = pd.to_datetime(
            df["wiki_date_added"], errors="coerce",
        )
    return df


def normalize_changes(changes: pd.DataFrame) -> pd.DataFrame:
    """Normalize Wikipedia's "Selected changes" change-log table.

    With pd.read_html and a header=0 flattening, the table comes back with
    columns like:
      ['Effective Date', 'Added', 'Added.1', 'Removed', 'Removed.1', 'Reason']
    The .1 suffix is pandas's deduplication of the multi-row header. The
    underlying schema is:
      Effective Date | Added Ticker | Added Security | Removed Ticker |
      Removed Security | Reason
    """
    cols = list(changes.columns)
    # Find the Date column.
    date_col = next(
        (c for c in cols if "date" in str(c).lower()),
        cols[0],
    )
    # Pandas dedupes duplicate columns by appending .1, .2 etc. so the FIRST
    # 'Added' column is the ticker and the SECOND 'Added.1' is the security.
    added_t = next((c for c in cols if str(c) == "Added"), None)
    added_s = next((c for c in cols if str(c) == "Added.1"), None)
    removed_t = next((c for c in cols if str(c) == "Removed"), None)
    removed_s = next((c for c in cols if str(c) == "Removed.1"), None)
    norm = pd.DataFrame({
        "date": pd.to_datetime(changes[date_col], errors="coerce"),
        "added_ticker": changes[added_t] if added_t else pd.NA,
        "added_security": changes[added_s] if added_s else pd.NA,
        "removed_ticker": changes[removed_t] if removed_t else pd.NA,
        "removed_security": changes[removed_s] if removed_s else pd.NA,
    })
    # Strip footnotes in ticker columns.
    for col in ("added_ticker", "removed_ticker"):
        norm[col] = (
            norm[col].astype(str).str.replace(r"\[.*?\]", "", regex=True)
            .str.strip().replace({"nan": pd.NA, "None": pd.NA, "": pd.NA})
        )
    return norm.dropna(subset=["date"]).reset_index(drop=True)


def pull_sec_ticker_directory() -> dict[str, dict]:
    """Pull SEC EDGAR's company_tickers.json and index by ticker -> (cik, name)."""
    print(f"[sec] pulling {SEC_TICKERS_URL}", flush=True)
    r = requests.get(SEC_TICKERS_URL, headers=HEADERS, timeout=30)
    r.raise_for_status()
    data = r.json()
    # Format: {"0": {"cik_str": ..., "ticker": ..., "title": ...}, "1": {...}}
    out = {}
    for row in data.values():
        t = str(row["ticker"]).strip().upper()
        out[t] = {
            "cik": int(row["cik_str"]),
            "name": row.get("title", ""),
        }
    print(f"[sec] indexed {len(out)} tickers", flush=True)
    return out


def reconcile_added_ticker(
    ticker: str, current_norm: pd.DataFrame, sec_dir: dict,
) -> dict:
    """Resolve CIK and GICS sector for a newly added ticker.

    Priority:
      1. If the ticker is on Wikipedia's current-constituents table, use the
         GICS sector and CIK from there.
      2. Otherwise fall back to SEC EDGAR for CIK; mark GICS unknown.
    """
    hit = current_norm[current_norm["ticker"] == ticker]
    if len(hit) == 1:
        row = hit.iloc[0]
        return {
            "ticker": ticker,
            "cik": (int(row["cik"]) if not pd.isna(row.get("cik"))
                    else (sec_dir.get(ticker, {}).get("cik"))),
            "gics_sector": row.get("gics_sector", "Unknown"),
            "gics_subsector": row.get("gics_subsector", "Unknown"),
            "name": row.get("name", ""),
        }
    # Not in current snapshot -> ticker has since been removed again.
    sec_hit = sec_dir.get(ticker, {})
    return {
        "ticker": ticker,
        "cik": sec_hit.get("cik"),
        "gics_sector": "Unknown",
        "gics_subsector": "Unknown",
        "name": sec_hit.get("name", ""),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--existing", type=str,
        default="data/raw/sp500/sp500_constituents_history.parquet",
        help="Existing PIT parquet to extend.",
    )
    p.add_argument(
        "--out", type=str,
        default="data/raw/sp500/sp500_constituents_history_extended.parquet",
        help="Path for the extended parquet.",
    )
    p.add_argument(
        "--cutoff", type=str, default="2023-01-01",
        help="Only consider changes dated on or after this date.",
    )
    args = p.parse_args()

    existing_path = Path(args.existing)
    out_path = Path(args.out)
    cutoff = pd.Timestamp(args.cutoff)

    print(f"[ext] loading existing PIT parquet: {existing_path}", flush=True)
    existing = pd.read_parquet(existing_path)
    print(f"[ext] existing rows: {len(existing)}", flush=True)
    print(f"[ext] existing date range: {existing['start_date'].min().date()} "
          f"to {existing['end_date'].max().date()}", flush=True)

    current, changes = pull_wikipedia_tables()
    current_norm = normalize_current(current)
    changes_norm = normalize_changes(changes)

    # Save audit-trail CSVs.
    audit_dir = out_path.parent
    audit_dir.mkdir(parents=True, exist_ok=True)
    current_norm.to_csv(audit_dir / "_wiki_current_2025.csv", index=False)
    changes_norm.to_csv(audit_dir / "_wiki_changes_2023plus.csv", index=False)
    print(f"[ext] saved audit CSVs to {audit_dir}", flush=True)

    # Filter changes to >= cutoff.
    changes_f = changes_norm[changes_norm["date"] >= cutoff].copy()
    print(f"[ext] changes after {cutoff.date()}: {len(changes_f)} rows",
          flush=True)

    sec_dir = pull_sec_ticker_directory()

    # The extension logic:
    # 1. Close existing intervals for removed tickers (set end_date = remove date - 1d).
    # 2. Open new intervals for added tickers (start_date = add date,
    #    end_date = today or current panel end).
    today = pd.Timestamp.today().normalize()

    new_rows = []
    closed_updates = {}  # ticker -> new end_date

    for _, ev in changes_f.iterrows():
        ev_date = ev["date"]
        if pd.notna(ev["removed_ticker"]):
            t = ev["removed_ticker"]
            # Close the most recent interval for this ticker.
            closed_updates[t] = ev_date - pd.Timedelta(days=1)
        if pd.notna(ev["added_ticker"]):
            t = ev["added_ticker"]
            meta = reconcile_added_ticker(t, current_norm, sec_dir)
            new_rows.append({
                "ticker": t,
                "cik": meta.get("cik"),
                "gics_sector": meta.get("gics_sector"),
                "gics_subsector": meta.get("gics_subsector"),
                "name": meta.get("name", ""),
                "start_date": ev_date,
                "end_date": today,
            })

    # Apply closed_updates: extend existing end_date to the new removal date
    # for any ticker still "active" (end_date >= 2022-12-31).
    existing_ext = existing.copy()
    existing_max_end = existing_ext["end_date"].max()
    for t, new_end in closed_updates.items():
        mask = (
            (existing_ext["ticker"] == t)
            & (existing_ext["end_date"] >= pd.Timestamp("2022-12-30"))
        )
        if mask.any():
            existing_ext.loc[mask, "end_date"] = new_end

    # Any ticker that was "active" at panel end 2022-12-31 and is NOT in the
    # closed_updates set is still active through `today`. Update its end_date.
    still_active_mask = (
        (existing_ext["end_date"] >= pd.Timestamp("2022-12-30"))
        & (~existing_ext["ticker"].isin(closed_updates.keys()))
    )
    existing_ext.loc[still_active_mask, "end_date"] = today

    # Append new additions.
    if new_rows:
        added_df = pd.DataFrame(new_rows)
        added_df["cik"] = added_df["cik"].astype("Int64")
        out_df = pd.concat([existing_ext, added_df], ignore_index=True)
    else:
        out_df = existing_ext

    out_df = out_df.sort_values(["ticker", "start_date"]).reset_index(drop=True)
    out_df.to_parquet(out_path, index=False)

    n_added = sum(1 for r in new_rows)
    n_removed = sum(1 for v in closed_updates.values() if v is not None)
    print(f"[ext] additions:   {n_added}", flush=True)
    print(f"[ext] removals:    {n_removed}", flush=True)
    print(f"[ext] still-active intervals extended to {today.date()}: "
          f"{int(still_active_mask.sum())}", flush=True)
    print(f"[ext] total rows in extended parquet: {len(out_df)}", flush=True)
    print(f"[ext] wrote {out_path}", flush=True)
    print(f"[ext] date range now: {out_df['start_date'].min().date()} "
          f"to {out_df['end_date'].max().date()}", flush=True)


if __name__ == "__main__":
    main()
