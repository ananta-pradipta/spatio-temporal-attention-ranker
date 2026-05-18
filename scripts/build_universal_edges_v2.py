"""Build mechanistic graph edges for the universal S&P 500 panel — v2.

Memory-efficient revision: comention pass uses per-shard hash dicts and
explicit garbage collection. Falls back to sector-only if the comention
streaming raises MemoryError.

Per spec 3d:
  - sector edges: GICS sectors
  - StockTwits cashtag co-mention: same construction; threshold by train freq
  - clinical-trial co-membership: DROP

Output: data/processed/sp500_edges.npz
"""
from __future__ import annotations

import ast
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch


def _load_universal_tickers() -> list[str]:
    snap = torch.load("data/processed/sp500_snapshots.pt", weights_only=False)
    return snap["tickers"]


def _add_edge(edges: dict, a: int, b: int, w: float) -> None:
    if a == b:
        return
    key = (min(a, b), max(a, b))
    edges[key] = edges.get(key, 0.0) + w


def build_sector_edges(tickers: list[str], hist: pd.DataFrame,
                        weight_per_pair: float = 0.25) -> dict[tuple[int, int], float]:
    ticker_to_idx = {t: i for i, t in enumerate(tickers)}
    sector_map = (hist.drop_duplicates("ticker")
                       .set_index("ticker")["gics_sector"].to_dict())
    out: dict[tuple[int, int], float] = {}
    sectors = pd.Series(sector_map).dropna()
    for sec, sub in sectors.groupby(sectors):
        idx = sorted({ticker_to_idx[t] for t in sub.index if t in ticker_to_idx})
        if len(idx) < 2:
            continue
        w = weight_per_pair / max(1.0, np.log1p(len(idx)))
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                _add_edge(out, idx[i], idx[j], w)
    return out


def build_comention_edges_streaming(tickers: list[str],
                                     train_start: str = "2015-01-09",
                                     train_end: str = "2018-12-21",
                                     min_count: int = 5,
                                     weight: float = 0.5) -> dict[tuple[int, int], float]:
    """Streaming comention pass.

    Strategy: build a {message_id -> channel_symbol} hash from the symbols
    parquets (filtered to train window), then for each msg_info shard look
    up channel symbols and emit pair contributions. After each shard, drop
    pandas DataFrames and run gc.collect().
    """
    ticker_to_idx = {t: i for i, t in enumerate(tickers)}
    universe_upper = set(ticker_to_idx.keys())
    COMMON = {"EDIT", "FOLD", "RARE", "ON", "ALL", "AT", "SO", "GO", "FOR", "IT",
              "NOW", "FIT", "LOW", "ARE", "BEST", "GAIN", "REAL", "EVER"}
    universe_lower = {t.lower() for t in universe_upper if t not in COMMON}

    # 1) Load message_id -> channel_symbol map from both corpora (train window only)
    print("[comention] loading symbol-channel hashmap...", flush=True)
    msg_to_channel: dict[int, str] = {}
    train_start_ts = pd.Timestamp(train_start)
    train_end_ts   = pd.Timestamp(train_end)

    def _consume_symbols_shard(p: Path) -> None:
        f = pq.ParquetFile(str(p))
        tbl = f.read(columns=["message_id", "symbol", "created_at"]).to_pandas()
        tbl["created_at"] = pd.to_datetime(tbl["created_at"])
        tbl = tbl[(tbl.created_at >= train_start_ts) & (tbl.created_at <= train_end_ts)]
        tbl = tbl[tbl["symbol"].isin(universe_upper)]
        for mid, sym in tbl[["message_id", "symbol"]].itertuples(index=False, name=None):
            msg_to_channel[int(mid)] = sym
        del tbl; gc.collect()

    biotech_sym = Path("data/raw/stocktwits/symbols.parquet")
    if biotech_sym.is_file():
        _consume_symbols_shard(biotech_sym)
    sp500_sym = Path("data/raw/stocktwits_sp500/symbols.parquet")
    if sp500_sym.is_dir():
        for i, p in enumerate(sorted(sp500_sym.glob("*.parquet")), 1):
            _consume_symbols_shard(p)
            if i % 30 == 0:
                print(f"  symbols pass: {i} shards consumed, msg_to_channel size: {len(msg_to_channel):,}", flush=True)
    print(f"[comention] symbol-channel hashmap final size: {len(msg_to_channel):,}", flush=True)

    # 2) Stream msg_info shards
    pairs_agg: dict[tuple[int, int], int] = {}

    def _consume_msginfo_shard(p: Path) -> int:
        f = pq.ParquetFile(str(p))
        info = f.read(columns=["message_id", "important_words"]).to_pandas()
        n_emit = 0
        for mid, raw in info[["message_id", "important_words"]].itertuples(index=False, name=None):
            channel = msg_to_channel.get(int(mid))
            if channel is None:
                continue
            if not isinstance(raw, str) or not raw:
                continue
            try:
                words = ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                continue
            if not isinstance(words, list):
                continue
            cashtags = {w.upper() for w in words
                         if isinstance(w, str) and w.lower() in universe_lower}
            cashtags.add(channel)
            if len(cashtags) < 2:
                continue
            idx = sorted({ticker_to_idx[t] for t in cashtags if t in ticker_to_idx})
            for i in range(len(idx)):
                for j in range(i + 1, len(idx)):
                    key = (idx[i], idx[j])
                    pairs_agg[key] = pairs_agg.get(key, 0) + 1
                    n_emit += 1
        del info; gc.collect()
        return n_emit

    print("[comention] streaming msg_info shards...", flush=True)
    biotech_info = Path("data/raw/stocktwits/msg_info.parquet")
    if biotech_info.is_file():
        n = _consume_msginfo_shard(biotech_info)
        print(f"[comention] biotech msg_info shard: {n:,} pair-events", flush=True)
    sp500_info = Path("data/raw/stocktwits_sp500/msg_info.parquet")
    if sp500_info.is_dir():
        shards = sorted(sp500_info.glob("*.parquet"))
        for i, p in enumerate(shards, 1):
            n = _consume_msginfo_shard(p)
            if i % 10 == 0:
                print(f"  [{i}/{len(shards)}] cum-pairs: {len(pairs_agg):,}", flush=True)

    print(f"[comention] unique pairs (pre-threshold): {len(pairs_agg):,}", flush=True)
    out: dict[tuple[int, int], float] = {}
    for key, cnt in pairs_agg.items():
        if cnt < min_count:
            continue
        out[key] = weight * np.log1p(cnt)
    return out


def main() -> None:
    tickers = _load_universal_tickers()
    print(f"Universe (universal panel): {len(tickers)} tickers", flush=True)
    hist = pd.read_parquet("data/raw/sp500/sp500_constituents_history.parquet")

    print("[trial] dropped per spec 3d", flush=True)
    sector = build_sector_edges(tickers, hist)
    print(f"[sector] pairs: {len(sector)}", flush=True)

    try:
        comention = build_comention_edges_streaming(tickers)
        print(f"[comention] pairs (post-threshold): {len(comention)}", flush=True)
    except MemoryError:
        print("[comention] MemoryError; falling back to sector-only edges", flush=True)
        comention = {}

    def _to_arrays(pairs: dict[tuple[int, int], float]) -> tuple[np.ndarray, np.ndarray]:
        if not pairs:
            return np.zeros((2, 0), dtype=np.int64), np.zeros(0, dtype=np.float32)
        src, dst, w = [], [], []
        for (a, b), wt in pairs.items():
            src.extend([a, b]); dst.extend([b, a]); w.extend([wt, wt])
        return np.asarray([src, dst], dtype=np.int64), np.asarray(w, dtype=np.float32)

    sec_idx, sec_w = _to_arrays(sector)
    co_idx, co_w = _to_arrays(comention)
    total_pairs = dict(sector)
    for k, v in comention.items():
        total_pairs[k] = total_pairs.get(k, 0.0) + v
    total_idx, total_w = _to_arrays(total_pairs)

    out_path = Path("data/processed/sp500_edges.npz")
    np.savez_compressed(out_path,
                        trial_idx=np.zeros((2, 0), dtype=np.int64),
                        trial_w=np.zeros(0, dtype=np.float32),
                        sector_idx=sec_idx, sector_w=sec_w,
                        comention_idx=co_idx, comention_w=co_w,
                        total_idx=total_idx, total_w=total_w,
                        tickers=np.asarray(tickers))
    print(f"\n[edges] wrote {out_path}", flush=True)
    print(f"[edges] sector:    {sec_idx.shape[1]} directed", flush=True)
    print(f"[edges] comention: {co_idx.shape[1]} directed", flush=True)
    print(f"[edges] total:     {total_idx.shape[1]} directed", flush=True)


if __name__ == "__main__":
    main()
