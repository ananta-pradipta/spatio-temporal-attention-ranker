"""Fetch VIX, VXN, VVIX daily close series from Yahoo Finance.

These three volatility indices are broadcast to every node at every timestep
as global volatility-regime features in MTGN. See preliminaries.md Section
2.5.5 and research-questions.md RQ1 (Phase 1).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import yfinance as yf


DEFAULT_SYMBOLS = ("^VIX", "^VXN", "^VVIX")


def fetch_volatility_indices(
    start_date: str,
    end_date: str,
    symbols: Iterable[str] = DEFAULT_SYMBOLS,
) -> pd.DataFrame:
    """Download daily close for each volatility index and return a wide DataFrame.

    Args:
        start_date: ISO date string, e.g. "2020-01-01".
        end_date: ISO date string, e.g. "2024-12-31".
        symbols: Yahoo tickers. Defaults to VIX, VXN, VVIX.

    Returns:
        DataFrame indexed by trading date with one column per index
        (e.g. VIX, VXN, VVIX). Columns use the bare name, not the caret prefix.
    """
    frames: list[pd.Series] = []
    for sym in symbols:
        raw = yf.download(
            sym,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=False,
        )
        if raw.empty:
            raise RuntimeError(f"No data returned for {sym}")
        close = raw["Close"]
        if isinstance(close, pd.DataFrame):
            # yfinance returns MultiIndex columns; collapse to a single column Series.
            close = close.iloc[:, 0]
        close.name = sym.lstrip("^")
        frames.append(close)
    out = pd.concat(frames, axis=1).sort_index()
    out.index.name = "date"
    return out


def save_volatility_indices(
    df: pd.DataFrame,
    path: Path = Path("data/raw/volatility_indices.parquet"),
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/raw/volatility_indices.parquet"),
    )
    args = parser.parse_args()

    df = fetch_volatility_indices(args.start, args.end)
    path = save_volatility_indices(df, args.out)
    print(f"Saved {len(df)} rows to {path}. Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
