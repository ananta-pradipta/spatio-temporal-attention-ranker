"""Download biotech stock price data from Yahoo Finance.

Usage:
    python scripts/data/download_prices.py --config configs/pilot_biotech.yaml

Downloads daily OHLCV data for the configured biotech universe and saves
one CSV per ticker plus a combined close-price panel.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yaml
import yfinance as yf


def load_config(config_path: str) -> dict:
    """Load YAML configuration file.

    Args:
        config_path: Path to the YAML config.

    Returns:
        Parsed configuration dictionary.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def download_ticker(
    ticker: str,
    start_date: str,
    end_date: str,
) -> Optional[pd.DataFrame]:
    """Download OHLCV data for a single ticker.

    Args:
        ticker: Stock ticker symbol.
        start_date: Start date string (YYYY-MM-DD).
        end_date: End date string (YYYY-MM-DD).

    Returns:
        DataFrame with OHLCV columns, or None if download fails.
    """
    try:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if df.empty:
            print(f"  WARN: No data for {ticker}, skipping.")
            return None
        # yfinance sometimes returns MultiIndex columns; flatten
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index.name = "Date"
        return df
    except Exception as e:
        print(f"  FAIL: {ticker} -- {e}")
        return None


def download_all(
    tickers: List[str],
    start_date: str,
    end_date: str,
    raw_dir: str,
) -> Dict[str, pd.DataFrame]:
    """Download price data for all tickers and save individual CSVs.

    Args:
        tickers: List of ticker symbols.
        start_date: Start date string.
        end_date: End date string.
        raw_dir: Directory to save raw CSVs.

    Returns:
        Dictionary mapping ticker to its DataFrame.
    """
    os.makedirs(raw_dir, exist_ok=True)
    results: Dict[str, pd.DataFrame] = {}

    print(f"Downloading {len(tickers)} tickers from {start_date} to {end_date}...")
    for ticker in tickers:
        df = download_ticker(ticker, start_date, end_date)
        if df is not None:
            csv_path = os.path.join(raw_dir, f"{ticker}.csv")
            df.to_csv(csv_path)
            results[ticker] = df
            print(f"  OK: {ticker} -- {len(df)} rows")

    print(f"\nDownloaded {len(results)}/{len(tickers)} tickers successfully.")
    return results


def build_close_panel(
    data: Dict[str, pd.DataFrame],
    raw_dir: str,
) -> pd.DataFrame:
    """Build a panel of adjusted close prices across all tickers.

    Args:
        data: Dictionary mapping ticker to DataFrame.
        raw_dir: Directory to save the panel CSV.

    Returns:
        DataFrame with dates as index and tickers as columns.
    """
    close_dict = {}
    for ticker, df in data.items():
        if "Adj Close" in df.columns:
            close_dict[ticker] = df["Adj Close"]
        elif "Close" in df.columns:
            close_dict[ticker] = df["Close"]

    panel = pd.DataFrame(close_dict)
    panel.index.name = "Date"
    panel.to_csv(os.path.join(raw_dir, "close_panel.csv"))
    print(f"Close panel saved: {panel.shape[0]} days x {panel.shape[1]} tickers")
    return panel


def build_volume_panel(
    data: Dict[str, pd.DataFrame],
    raw_dir: str,
) -> pd.DataFrame:
    """Build a panel of trading volumes across all tickers.

    Args:
        data: Dictionary mapping ticker to DataFrame.
        raw_dir: Directory to save the panel CSV.

    Returns:
        DataFrame with dates as index and tickers as columns.
    """
    vol_dict = {}
    for ticker, df in data.items():
        if "Volume" in df.columns:
            vol_dict[ticker] = df["Volume"]

    panel = pd.DataFrame(vol_dict)
    panel.index.name = "Date"
    panel.to_csv(os.path.join(raw_dir, "volume_panel.csv"))
    print(f"Volume panel saved: {panel.shape[0]} days x {panel.shape[1]} tickers")
    return panel


def main() -> None:
    """Main entry point for the download script."""
    parser = argparse.ArgumentParser(description="Download biotech stock prices.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pilot_biotech.yaml",
        help="Path to YAML config file.",
    )
    args = parser.parse_args()

    # Resolve paths relative to project root
    project_root = Path(__file__).resolve().parent.parent.parent
    config_path = project_root / args.config
    cfg = load_config(str(config_path))

    data_cfg = cfg["data"]
    raw_dir = str(project_root / data_cfg["raw_dir"])

    data = download_all(
        tickers=data_cfg["tickers"],
        start_date=data_cfg["start_date"],
        end_date=data_cfg["end_date"],
        raw_dir=raw_dir,
    )

    if not data:
        print("FAIL: No data downloaded. Check network and ticker list.")
        sys.exit(1)

    build_close_panel(data, raw_dir)
    build_volume_panel(data, raw_dir)
    print("\nData download complete.")


if __name__ == "__main__":
    main()
