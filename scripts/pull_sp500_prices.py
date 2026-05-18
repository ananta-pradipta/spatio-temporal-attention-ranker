"""Phase 2a: pull yfinance OHLCV for the 685 S&P 500 historical-constituent tickers.

Schema matches `prices_universe.parquet`: ticker, date, open, high, low, close,
adj_close, volume.

Output: data/raw/sp500/prices_sp500.parquet
Failure log: data/raw/sp500/prices_missing.txt with reason.

Usage:
  PYTHONPATH=/home/apradipta/phd-research python3 scripts/pull_sp500_prices.py
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")


def fetch_yfinance(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    import yfinance as yf
    try:
        df = yf.download(
            ticker, start=start, end=end,
            auto_adjust=False, progress=False, threads=False,
        )
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]
        df["ticker"] = ticker
        df = df.rename(columns={"adj close": "adj_close"})
        return df[["ticker", "date", "open", "high", "low", "close", "adj_close", "volume"]]
    except Exception as e:
        print(f"  yfinance error for {ticker}: {str(e)[:80]}")
        return None


def main() -> None:
    out_dir = Path("data/raw/sp500")
    out_dir.mkdir(parents=True, exist_ok=True)

    hist = pd.read_parquet(out_dir / "sp500_constituents_history.parquet")
    tickers = sorted(hist.ticker.unique().tolist())
    print(f"Pulling OHLCV for {len(tickers)} tickers, 2015-01-01 to 2022-12-31...")

    panels: list[pd.DataFrame] = []
    missing: list[tuple[str, str]] = []
    for i, t in enumerate(tickers, 1):
        df = fetch_yfinance(t, "2015-01-01", "2023-01-15")
        if df is None or df.empty:
            missing.append((t, "no data"))
        elif len(df) < 60:
            missing.append((t, f"only {len(df)} rows"))
            panels.append(df)
        else:
            panels.append(df)
        if i % 50 == 0:
            print(f"  [{i}/{len(tickers)}]  with-data={len(panels)}  missing={len(missing)}")
        time.sleep(0.05)

    if panels:
        panel = pd.concat(panels, ignore_index=True)
        panel["date"] = pd.to_datetime(panel["date"])
        panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
        panel.to_parquet(out_dir / "prices_sp500.parquet")
        print(f"\nWrote {out_dir / 'prices_sp500.parquet'}: {len(panel):,} rows, "
              f"{panel.ticker.nunique()} tickers")

    if missing:
        with open(out_dir / "prices_missing.txt", "w") as f:
            for t, reason in missing:
                f.write(f"{t}\t{reason}\n")
        print(f"Wrote {out_dir / 'prices_missing.txt'}: {len(missing)} tickers")


if __name__ == "__main__":
    main()
