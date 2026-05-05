"""Download a cashtag-filtered subset of the StockTwits 2008 to 2022 dataset.

Source: s3://stocktwits-nyu/dataset/v1/data/csv (public, no credentials).
Reference: Li, X., Al Ansari, N., and Kaufman, A. (2025). StockTwits:
Comprehensive records of a financial social media platform from 2008 to 2022.
Journal of Quantitative Description: Digital Media, 5.
https://doi.org/10.51685/jqd.2025.020

Key insight from smoke test (docs/stocktwits_smoke_test.md): the `symbols/`
category is a denormalized per-(message, symbol) view with a `symbol`
column as direct filter key. One message mentioning two tickers appears
as two rows. This makes the biotech-universe filter trivial and avoids
parsing the stringified `symbol_list`.

Strategy:
    1. Stream `symbols/*.csv` via dask, filter on `symbol.isin(universe)`,
       write a consolidated parquet under data/raw/stocktwits/symbols.parquet.
    2. Optionally join `msg_info/*.csv` on `message_id` to pull in `length`
       and `important_words` fields for DD-analog filtering.
    3. `feature_wo_messages/` is only needed when intraday timestamps
       matter (Phase 3); skipped by default.

Usage:
    python3 -m src.mtgn.data.download_stocktwits \\
        --tickers-file src/mtgn/data/xbi_proxy_tickers.txt \\
        --categories symbols msg_info \\
        --out-dir data/raw/stocktwits
"""
from __future__ import annotations

import argparse
from pathlib import Path

import dask.dataframe as dd


S3_BASE = "s3://stocktwits-nyu/dataset/v1/data/csv"
STORAGE_OPTIONS = {"anon": True}

# Schemas observed in the 2026-04-12 smoke test. See docs/stocktwits_smoke_test.md.
CATEGORIES = {
    "symbols": {
        "pattern": f"{S3_BASE}/symbols/*.csv",
        "ticker_column": "symbol",   # direct filter key (uppercase)
    },
    "msg_info": {
        "pattern": f"{S3_BASE}/msg_info/*.csv",
        "ticker_column": None,        # no ticker column; join via message_id after filter
    },
    "feature_wo_messages": {
        "pattern": f"{S3_BASE}/feature_wo_messages/*.csv",
        "ticker_column": "symbol_list_match",  # stringified list, custom filter
    },
}


def load_tickers(tickers_file: Path) -> set[str]:
    tickers = {line.strip().upper() for line in tickers_file.read_text().splitlines() if line.strip()}
    if not tickers:
        raise ValueError(f"No tickers loaded from {tickers_file}")
    return tickers


def download_symbols(tickers: set[str], out_dir: Path) -> Path:
    """Stream symbols/*.csv and filter on symbol column. Writes parquet."""
    ddf = dd.read_csv(
        CATEGORIES["symbols"]["pattern"],
        storage_options=STORAGE_OPTIONS,
        dtype={
            "message_id": "int64",
            "user_id": "int64",
            "created_at": "string",
            "sentiment": "float64",
            "symbol_list": "string",
            "sym_number": "int64",
            "symbol": "string",
        },
    )
    ddf = ddf[ddf["symbol"].isin(tickers)]
    out_path = out_dir / "symbols.parquet"
    out_dir.mkdir(parents=True, exist_ok=True)
    ddf.to_parquet(out_path, overwrite=True, write_index=False)
    return out_path


def download_msg_info(filtered_message_ids: set[int], out_dir: Path) -> Path:
    """Stream msg_info/*.csv and keep rows whose message_id is in the filter set.

    `filtered_message_ids` is typically derived from the symbols download; we
    only want length/important_words for messages that mentioned our universe.
    """
    ddf = dd.read_csv(
        CATEGORIES["msg_info"]["pattern"],
        storage_options=STORAGE_OPTIONS,
        dtype={"message_id": "int64", "length": "int64", "important_words": "string"},
    )
    ddf = ddf[ddf["message_id"].isin(filtered_message_ids)]
    out_path = out_dir / "msg_info.parquet"
    out_dir.mkdir(parents=True, exist_ok=True)
    ddf.to_parquet(out_path, overwrite=True, write_index=False)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers-file", type=Path, required=True)
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["symbols", "msg_info"],
        choices=list(CATEGORIES.keys()),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("data/raw/stocktwits"))
    args = parser.parse_args()

    tickers = load_tickers(args.tickers_file)
    print(f"Filtering to {len(tickers)} biotech tickers.")

    if "symbols" in args.categories:
        print(f"Streaming {CATEGORIES['symbols']['pattern']}")
        path = download_symbols(tickers, args.out_dir)
        print(f"Wrote {path}")
    if "msg_info" in args.categories:
        import pandas as pd

        symbols_parquet = args.out_dir / "symbols.parquet"
        if not symbols_parquet.exists():
            raise RuntimeError("Run the symbols category first; msg_info joins on its message_id set.")
        msg_ids = set(pd.read_parquet(symbols_parquet, columns=["message_id"])["message_id"].tolist())
        print(f"Joining msg_info on {len(msg_ids):,} filtered message_ids.")
        path = download_msg_info(msg_ids, args.out_dir)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
