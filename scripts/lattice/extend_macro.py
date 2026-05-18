"""Phase 1 delta pull: extra macro factors needed by LATTICE beyond v2.

LATTICE's 24-d macro state needs:
  - FRED:    DGS3MO, DGS2, DGS10, BAA10Y (already in v2's macro_fred_full.csv)
             T10YIE (10-year breakeven), DTWEXBGS (USD index)  [NEW]
  - VIX, VVIX (already in v2's risk_features.parquet)
  - MOVE proxy: rolling 20-day stdev of DGS10 changes (computed from FRED, no pull needed)
  - SPY, QQQ, plus 11 GICS-aligned sector ETFs (already in v2's sector_etfs.parquet)
  - IWM, HYG, TLT, GLD broad ETFs  [NEW]

This script pulls only the new series and writes them to:
  data/lattice/raw/macro_fred_extra.csv
  data/lattice/raw/macro_etfs_extra.parquet

Phase 1 panel-build assembly (separate script) will merge these with the
symlinked v2 files into the final macro state vector.
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")


def fetch_fred_series(series_ids: list[str], start: str, end: str) -> pd.DataFrame:
    from pandas_datareader import data as web
    parts = []
    for s in series_ids:
        try:
            x = web.DataReader(s, "fred", start, end)
        except Exception as exc:
            print(f"  FRED {s}: error {exc}")
            continue
        x.columns = [s]
        parts.append(x)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, axis=1).sort_index()
    out.index.name = "date"
    return out


def fetch_etfs(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    import yfinance as yf
    frames = []
    for t in tickers:
        try:
            df = yf.download(t, start=start, end=end,
                             auto_adjust=False, progress=False, threads=False)
            if df is None or df.empty:
                print(f"  yfinance {t}: empty")
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
            print(f"  {t}: {len(df)} rows {df.date.min().date()} -> {df.date.max().date()}")
        except Exception as e:
            print(f"  yfinance {t}: error {str(e)[:80]}")
        time.sleep(0.05)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main() -> None:
    out_dir = Path("data/lattice/raw")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Pulling new FRED series (T10YIE, DTWEXBGS)...")
    fred = fetch_fred_series(["T10YIE", "DTWEXBGS"],
                              start="2014-09-01", end="2023-01-15")
    fred_path = out_dir / "macro_fred_extra.csv"
    fred.to_csv(fred_path)
    print(f"Wrote {fred_path}: shape={fred.shape}, "
          f"range {fred.index.min()} -> {fred.index.max()}")

    print("\nPulling new broad ETFs (IWM, HYG, TLT, GLD)...")
    etfs = fetch_etfs(["IWM", "HYG", "TLT", "GLD"],
                       start="2014-09-01", end="2023-01-15")
    etf_path = out_dir / "macro_etfs_extra.parquet"
    etfs.to_parquet(etf_path)
    print(f"Wrote {etf_path}: {len(etfs):,} rows, {etfs.ticker.nunique()} ETFs")


if __name__ == "__main__":
    main()
