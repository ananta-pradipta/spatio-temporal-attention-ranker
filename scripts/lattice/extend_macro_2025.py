"""Extend macro raw data forward to 2025-12-31 (or current date).

Re-pulls FRED series and ETF returns for the full range 2014-09-01 to
current-date, overwriting the existing macro_fred_full.csv,
macro_fred_extra.csv, sector_etfs.parquet, and macro_etfs_extra.parquet.

The 2026-05-11 dataset extension (Scenario A) per
docs/dataset_extension_feasibility.md requires macro coverage through
2025-12 to support new F4 (test 2023 H2 to 2024 H1) and F5 (test 2024 H2
to 2025 H1) walk-forward folds.

This script is a forward extension of scripts/lattice/extend_macro.py
(which hardcoded end="2023-01-15"). Total wall time ~15 minutes,
network-bound.

Outputs (all in-place updates):
  data/raw/macro_fred_full.csv         (existing FRED series; full re-pull)
  data/lattice/raw/macro_fred_extra.csv (T10YIE, DTWEXBGS extras; full re-pull)
  data/raw/sp500/sector_etfs.parquet    (SPY, QQQ, XL* sector ETFs)
  data/lattice/raw/macro_etfs_extra.parquet (IWM, HYG, TLT, GLD broad ETFs)
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")


# All FRED series in the production macro state.
FRED_FULL_SERIES = ["DGS3MO", "DGS2", "DGS10", "BAA10Y", "VIXCLS", "VIXMCLS"]
FRED_EXTRA_SERIES = ["T10YIE", "DTWEXBGS"]

# Sector and broad ETFs.
SECTOR_ETFS = ["SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLY",
               "XLP", "XLU", "XLRE", "XLC", "XLB", "XLI"]
BROAD_ETFS_EXTRA = ["IWM", "HYG", "TLT", "GLD"]

START = "2014-09-01"
END = pd.Timestamp.today().normalize().strftime("%Y-%m-%d")


def fetch_fred(series: list[str]) -> pd.DataFrame:
    from pandas_datareader import data as web
    parts = []
    for s in series:
        try:
            x = web.DataReader(s, "fred", START, END)
        except Exception as exc:
            print(f"  FRED {s}: error {exc}", flush=True)
            continue
        x.columns = [s]
        parts.append(x)
        print(f"  FRED {s}: {len(x)} rows, "
              f"{x.index.min().date()} -> {x.index.max().date()}",
              flush=True)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, axis=1).sort_index()
    out.index.name = "date"
    return out


def fetch_etfs(tickers: list[str]) -> pd.DataFrame:
    import yfinance as yf
    frames = []
    for t in tickers:
        try:
            df = yf.download(t, start=START, end=END,
                             auto_adjust=False, progress=False, threads=False)
            if df is None or df.empty:
                print(f"  yfinance {t}: empty", flush=True)
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            df = df.reset_index()
            df.columns = [c.lower() for c in df.columns]
            df = df.rename(columns={"adj close": "adj_close"})
            df["ticker"] = t
            df = df[["ticker", "date", "open", "high", "low", "close",
                     "adj_close", "volume"]]
            frames.append(df)
            print(f"  {t}: {len(df)} rows {df.date.min().date()} -> {df.date.max().date()}",
                  flush=True)
        except Exception as e:
            print(f"  yfinance {t}: error {str(e)[:80]}", flush=True)
        time.sleep(0.05)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main() -> None:
    print(f"Pulling macro for {START} to {END}", flush=True)
    print("\n=== FRED full ===", flush=True)
    fred_full = fetch_fred(FRED_FULL_SERIES)
    out_fred_full = Path("data/raw/macro_fred_full.csv")
    out_fred_full.parent.mkdir(parents=True, exist_ok=True)
    fred_full.to_csv(out_fred_full)
    print(f"Wrote {out_fred_full}: shape={fred_full.shape}", flush=True)

    print("\n=== FRED extra (T10YIE, DTWEXBGS) ===", flush=True)
    fred_extra = fetch_fred(FRED_EXTRA_SERIES)
    out_fred_extra = Path("data/lattice/raw/macro_fred_extra.csv")
    out_fred_extra.parent.mkdir(parents=True, exist_ok=True)
    fred_extra.to_csv(out_fred_extra)
    print(f"Wrote {out_fred_extra}: shape={fred_extra.shape}", flush=True)

    print("\n=== Sector + index ETFs ===", flush=True)
    sec_etfs = fetch_etfs(SECTOR_ETFS)
    out_sec_etfs = Path("data/raw/sp500/sector_etfs.parquet")
    out_sec_etfs.parent.mkdir(parents=True, exist_ok=True)
    sec_etfs.to_parquet(out_sec_etfs)
    print(f"Wrote {out_sec_etfs}: {len(sec_etfs):,} rows, "
          f"{sec_etfs['ticker'].nunique()} tickers", flush=True)

    print("\n=== Broad ETFs (IWM, HYG, TLT, GLD) ===", flush=True)
    broad_etfs = fetch_etfs(BROAD_ETFS_EXTRA)
    out_broad = Path("data/lattice/raw/macro_etfs_extra.parquet")
    out_broad.parent.mkdir(parents=True, exist_ok=True)
    broad_etfs.to_parquet(out_broad)
    print(f"Wrote {out_broad}: {len(broad_etfs):,} rows, "
          f"{broad_etfs['ticker'].nunique()} tickers", flush=True)


if __name__ == "__main__":
    main()
