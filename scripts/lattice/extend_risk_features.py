"""Extend risk_features parquet to cover 2023 through 2025 using FRED VIXCLS.

The existing risk_features parquets stop at 2022-12-30, which means VIX
is NaN-filled for all F4 (test 2024) and F5 (test 2025) cells in the
trainer. FRED VIXCLS is already cached on disk through 2026-05 in
data/raw/macro_fred_full.csv -- this script reads that and extends:

  data/processed/risk_features.parquet         (biotech path)
  data/processed/risk_features_sp500.parquet   (universal path)

The downstream macro_state.parquet rebuild (build_macro_state) then picks
up real VIX values for 2023-2025 and the IPO-key VIX slot stops being NaN.

Secondary columns (vxn, vvix, vix_term_slope, xbi_*) are left as-is for
the extension period; xbi_* is biotech-specific and not load-bearing on
the universal panel, vxn/vvix are NaN'd, vix_term_slope falls back to 0
via the existing build_macro.py guard.

Run from repo root:
    python -m scripts.lattice.extend_risk_features
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def extend_one(parquet_path: Path, fred_csv: Path, end_date: str) -> None:
    """Append FRED-sourced VIX rows to a risk_features parquet."""
    risk = pd.read_parquet(parquet_path)
    risk = risk.copy()
    risk.index = pd.to_datetime(risk.index)
    risk_end = risk.index.max()
    print(f"  existing rows: {len(risk)}, end={risk_end.date()}", flush=True)

    fred = pd.read_csv(fred_csv, parse_dates=["date"]).set_index("date")
    fred = fred[["VIXCLS"]].dropna()
    end_ts = pd.Timestamp(end_date)
    fred = fred[(fred.index > risk_end) & (fred.index <= end_ts)]
    if fred.empty:
        print(f"  no new dates to append (FRED VIX coverage exhausted)", flush=True)
        return

    new = pd.DataFrame(index=fred.index, columns=risk.columns, dtype="float64")
    new["vix"] = fred["VIXCLS"].astype("float64")
    new.index.name = "date"

    combined = pd.concat([risk, new]).sort_index()
    combined = combined[~combined.index.duplicated(keep="first")]
    print(f"  new rows appended: {len(new)}; total: {len(combined)}, "
          f"end={combined.index.max().date()}", flush=True)
    combined.to_parquet(parquet_path)
    print(f"  wrote {parquet_path}", flush=True)


def main() -> None:
    fred_csv = Path("data/raw/macro_fred_full.csv")
    targets = [
        Path("data/processed/risk_features.parquet"),
        Path("data/processed/risk_features_sp500.parquet"),
    ]
    end_date = "2025-12-31"
    for p in targets:
        if not p.exists():
            print(f"SKIP {p}: missing", flush=True)
            continue
        print(f"\nExtending {p}", flush=True)
        extend_one(p, fred_csv, end_date)


if __name__ == "__main__":
    main()
