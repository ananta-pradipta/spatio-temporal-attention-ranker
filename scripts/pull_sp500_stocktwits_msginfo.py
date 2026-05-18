"""Memory-efficient msg_info pull for the S&P 500 StockTwits corpus.

The vendored ``src.mtgn.data.download_stocktwits`` uses
``ddf[ddf.message_id.isin(set_of_36M_ids)]`` which OOMs on a 7.4 GB box
because each dask worker materialises the full Python set. This script
does the same filter but streams partition-by-partition with explicit
chunked reads, never holding more than ~1M rows in memory at once.

Strategy:
  1. Load the message_id filter set from the symbols.parquet partitions.
  2. List msg_info/*.csv partitions in s3://stocktwits-nyu via fsspec.
  3. For each partition, read in 1M-row chunks, filter on message_id, append.
  4. Write a single consolidated parquet at the end.

Output: data/raw/stocktwits_sp500/msg_info.parquet
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm


S3_PATTERN = "s3://stocktwits-nyu/dataset/v1/data/csv/msg_info"
OUT_PATH = Path("data/raw/stocktwits_sp500/msg_info.parquet")


def load_filter_ids(symbols_dir: Path) -> set[int]:
    parts = sorted(symbols_dir.glob("*.parquet"))
    print(f"Loading filter ids from {len(parts)} symbol partitions...")
    ids: set[int] = set()
    for p in tqdm(parts, ncols=80):
        f = pq.ParquetFile(str(p))
        ids.update(f.read(columns=["message_id"]).column("message_id").to_pylist())
    print(f"  filter set size: {len(ids):,}")
    return ids


def stream_filter_msg_info(filter_ids: set[int], out_dir: Path) -> int:
    """Stream msg_info S3 partitions; write one parquet shard per partition.

    Avoids holding the full filtered set in memory. Memory ceiling is one
    chunk (1M rows) plus the filter set (~300 MB for 36M ids).
    """
    import s3fs
    fs = s3fs.S3FileSystem(anon=True)
    bucket = "stocktwits-nyu/dataset/v1/data/csv/msg_info"
    files = sorted(fs.ls(bucket))
    files = [f for f in files if f.endswith(".csv")]
    print(f"S3 partitions to scan: {len(files)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    chunk_size = 1_000_000
    total_kept = 0
    for i, f in enumerate(files):
        size_mb = fs.info(f).get("size", 0) / 1e6
        shards: list[pd.DataFrame] = []
        with fs.open(f, mode="rb") as fh:
            for chunk in pd.read_csv(
                fh, chunksize=chunk_size,
                dtype={"message_id": "int64", "length": "int64", "important_words": "string"},
            ):
                hits = chunk[chunk["message_id"].isin(filter_ids)]
                if len(hits):
                    shards.append(hits)
        if shards:
            part = pd.concat(shards, ignore_index=True)
            shard_path = out_dir / f"part-{i:04d}.parquet"
            part.to_parquet(shard_path, index=False)
            kept = len(part)
            total_kept += kept
        else:
            kept = 0
        print(f"  [{i+1}/{len(files)}] {Path(f).name}: {size_mb:.0f} MB scanned, kept {kept:,} rows  (cum: {total_kept:,})")
    return total_kept


def main() -> None:
    symbols_dir = Path("data/raw/stocktwits_sp500/symbols.parquet")
    filter_ids = load_filter_ids(symbols_dir)

    # Write per-partition shards into msg_info.parquet/ directory (matches dask convention)
    out_dir = OUT_PATH if OUT_PATH.suffix == ".parquet" else Path(str(OUT_PATH) + ".parquet")
    n = stream_filter_msg_info(filter_ids, out_dir)
    print(f"\nWrote {out_dir}: {n:,} rows across {len(list(out_dir.glob('*.parquet')))} shards")


if __name__ == "__main__":
    main()
