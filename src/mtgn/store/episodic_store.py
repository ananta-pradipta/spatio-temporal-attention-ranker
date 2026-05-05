"""Episodic store for MTGN: salience-gated KV cache backed by FAISS HNSW.

Stores raw node memory snapshots s_i(t) at write time. Per memo §6.4
option B, the W_q / W_k / W_v projections are applied at QUERY time, not
write time, so the store stays stable as the projection matrices train.

Retrieval returns the top-K neighbors plus their metadata, including
timestamps (for TGAT-style time encoding on keys) and the realized
forward return for the quantile-loss risk head's non-parametric sample.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


try:
    import faiss  # type: ignore
except ImportError as e:  # pragma: no cover
    raise ImportError("faiss-cpu is required; install via `pip install faiss-cpu`.") from e


@dataclass
class StoreConfig:
    dim: int = 128
    hnsw_m: int = 32
    ef_construction: int = 200
    ef_search: int = 64


@dataclass
class StoredEntry:
    ticker_id: int
    time: float                  # normalized timestamp (days since epoch start)
    memory: np.ndarray           # [dim] raw s_i(t)
    meta: dict[str, Any] = field(default_factory=dict)


class EpisodicStore:
    """Salience-gated KV cache with FAISS HNSW index for kNN retrieval."""

    def __init__(self, cfg: StoreConfig | None = None):
        self.cfg = cfg or StoreConfig()
        self.index = faiss.IndexHNSWFlat(self.cfg.dim, self.cfg.hnsw_m)
        self.index.hnsw.efConstruction = self.cfg.ef_construction
        self.index.hnsw.efSearch = self.cfg.ef_search
        self._entries: list[StoredEntry] = []

    @property
    def size(self) -> int:
        return len(self._entries)

    def write(self, entry: StoredEntry) -> None:
        if entry.memory.dtype != np.float32:
            entry.memory = entry.memory.astype(np.float32)
        if entry.memory.shape != (self.cfg.dim,):
            raise ValueError(f"entry.memory must be shape ({self.cfg.dim},)")
        self._entries.append(entry)
        self.index.add(entry.memory[None, :])

    def write_batch(self, entries: list[StoredEntry]) -> None:
        if not entries:
            return
        for e in entries:
            if e.memory.dtype != np.float32:
                e.memory = e.memory.astype(np.float32)
        matrix = np.stack([e.memory for e in entries])
        self._entries.extend(entries)
        self.index.add(matrix)

    def retrieve(
        self, query: np.ndarray, k: int, t_max: float | None = None,
        self_ticker_id: int | None = None, mode: str = "cross_entity",
    ) -> tuple[list[StoredEntry], np.ndarray]:
        """kNN retrieval with causality and mode filters.

        Over-retrieves then filters (FAISS does not support attribute
        filters natively). `mode` in {"cross_entity", "self_only"}.
        Returns (entries, distances).
        """
        if self.size == 0:
            return [], np.zeros((0,), dtype=np.float32)
        over_k = min(self.size, max(k * 4, 32))
        q = np.ascontiguousarray(query[None, :].astype(np.float32))
        dists, idx = self.index.search(q, over_k)
        dists = dists[0]
        idx = idx[0]

        picked: list[StoredEntry] = []
        picked_dists: list[float] = []
        for d, i in zip(dists, idx):
            if i < 0 or i >= self.size:
                continue
            entry = self._entries[i]
            if t_max is not None and entry.time >= t_max:
                continue
            if mode == "self_only" and entry.ticker_id != self_ticker_id:
                continue
            picked.append(entry)
            picked_dists.append(float(d))
            if len(picked) >= k:
                break

        return picked, np.asarray(picked_dists, dtype=np.float32)

    # Convenience helpers for the training loop
    def stack_memory(self, entries: list[StoredEntry]) -> np.ndarray:
        if not entries:
            return np.zeros((0, self.cfg.dim), dtype=np.float32)
        return np.stack([e.memory for e in entries])

    def stack_times(self, entries: list[StoredEntry]) -> np.ndarray:
        return np.asarray([e.time for e in entries], dtype=np.float32)

    def stack_forward_returns(self, entries: list[StoredEntry]) -> np.ndarray:
        """Used by the quantile head as a non-parametric conditional distribution."""
        return np.asarray(
            [e.meta.get("forward_return_h", np.nan) for e in entries], dtype=np.float32
        )


__all__ = ["EpisodicStore", "StoreConfig", "StoredEntry"]
