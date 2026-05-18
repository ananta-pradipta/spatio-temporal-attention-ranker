"""Phase 1b: pull daily OHLCV for the 52 catalogued delisted tickers.

Pipeline per ticker:
  1. yfinance with auto_adjust=True, date-bounded to (2015-01-01, delisting_date+5d).
     Date bounding handles recycled-ticker cases where yfinance returns the
     new company's data after the original delisted.
  2. If yfinance returns < 60 trading days OR no rows, fall back to Stooq
     via pandas-datareader (alternative source).
  3. Log every fetch to `data/raw/delisted_source_log.csv` with columns
     ticker, source, first_date, last_date, n_rows.
  4. Tickers with no data from either source go to
     `data/raw/delisted_missing_yfinance.txt` with a brief reason.
  5. Save consolidated price panel to `data/raw/prices_delisted.parquet`
     with the same schema as `prices_universe.parquet`:
       columns: ticker, date, open, high, low, close, volume, adj_close

Usage:
  python3 scripts/pull_delisted_prices.py
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import warnings
warnings.filterwarnings("ignore")


def fetch_yfinance(ticker: str, end_date: str) -> pd.DataFrame | None:
    """Date-bounded yfinance pull (handles recycled tickers)."""
    import yfinance as yf
    try:
        df = yf.download(
            ticker,
            start="2015-01-01",
            end=end_date,
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        if df is None or df.empty:
            return None
        # Flatten multi-index columns if yfinance returns them
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]
        df["ticker"] = ticker
        # Standardise column names to match prices_universe.parquet
        rename = {"adj close": "adj_close"}
        df = df.rename(columns=rename)
        return df[["ticker", "date", "open", "high", "low", "close", "adj_close", "volume"]]
    except Exception as e:
        print(f"    yfinance error for {ticker}: {e}")
        return None


def fetch_stooq(ticker: str, end_date: str) -> pd.DataFrame | None:
    """Stooq fallback via pandas-datareader."""
    try:
        from pandas_datareader import data as pdr
        df = pdr.DataReader(
            ticker, "stooq",
            start="2015-01-01",
            end=end_date,
        )
        if df is None or df.empty:
            return None
        df = df.reset_index().sort_values("Date")
        df.columns = [c.lower() for c in df.columns]
        df["ticker"] = ticker
        # Stooq doesn't have adj_close; use close
        if "adj_close" not in df.columns:
            df["adj_close"] = df["close"]
        return df[["ticker", "date", "open", "high", "low", "close", "adj_close", "volume"]]
    except Exception as e:
        print(f"    stooq error for {ticker}: {e}")
        return None


def main() -> None:
    out_dir = Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)

    catalogue = pd.read_csv("data/delisting_log_v1.csv")
    catalogue["delisting_date"] = pd.to_datetime(catalogue.delisting_date)
    catalogue = catalogue.sort_values("delisting_date")
    print(f"Pulling prices for {len(catalogue)} catalogued tickers...")

    all_panels: list[pd.DataFrame] = []
    log_rows: list[dict] = []
    missing: list[tuple[str, str]] = []

    for i, row in enumerate(catalogue.itertuples(), 1):
        t = row.ticker
        end_buffer = (row.delisting_date + pd.Timedelta(days=5)).strftime("%Y-%m-%d")
        print(f"[{i}/{len(catalogue)}] {t:<6}  end={end_buffer}  ({row.reason})")
        df = fetch_yfinance(t, end_buffer)
        source = "yfinance"
        if df is None or len(df) < 60:
            yf_n = 0 if df is None else len(df)
            print(f"    yfinance returned {yf_n} rows; trying stooq")
            df_alt = fetch_stooq(t, end_buffer)
            if df_alt is not None and len(df_alt) > yf_n:
                df = df_alt
                source = "stooq"
                print(f"    stooq: {len(df)} rows")
            elif df is not None:
                print(f"    keeping yfinance ({yf_n} rows, below 60-day threshold)")
            else:
                df = df_alt
                source = "stooq" if df is not None else "none"
        if df is None or df.empty:
            missing.append((t, "no data from yfinance or stooq"))
            log_rows.append({"ticker": t, "source": "none", "first_date": "", "last_date": "", "n_rows": 0})
            continue
        all_panels.append(df)
        log_rows.append({
            "ticker": t,
            "source": source,
            "first_date": pd.to_datetime(df.date.min()).date(),
            "last_date": pd.to_datetime(df.date.max()).date(),
            "n_rows": len(df),
        })
        time.sleep(0.2)  # be nice to the data sources

    # Consolidate
    if all_panels:
        panel = pd.concat(all_panels, ignore_index=True)
        panel["date"] = pd.to_datetime(panel["date"])
        panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
        panel.to_parquet(out_dir / "prices_delisted.parquet")
        print(f"\nWrote {out_dir / 'prices_delisted.parquet'}: "
              f"{len(panel)} rows across {panel.ticker.nunique()} tickers")
    else:
        print("\nNo price data collected.")

    # Log
    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(out_dir / "delisted_source_log.csv", index=False)
    print(f"Wrote {out_dir / 'delisted_source_log.csv'}: {len(log_df)} rows")

    # Missing
    if missing:
        with open(out_dir / "delisted_missing_yfinance.txt", "w") as f:
            for t, reason in missing:
                f.write(f"{t}\t{reason}\n")
        print(f"Wrote {out_dir / 'delisted_missing_yfinance.txt'}: {len(missing)} tickers")
    print("\n=== Coverage summary ===")
    print(f"Total tickers:        {len(catalogue)}")
    print(f"With data:            {len(log_df[log_df.source != 'none'])}")
    print(f"  via yfinance:       {(log_df.source == 'yfinance').sum()}")
    print(f"  via stooq:          {(log_df.source == 'stooq').sum()}")
    print(f"Missing:              {len(missing)}")
    if not log_df.empty:
        print("\nPer-ticker coverage (sorted by n_rows):")
        ranked = log_df[log_df.source != 'none'].sort_values("n_rows", ascending=False)
        print(ranked.to_string(index=False))


if __name__ == "__main__":
    main()
