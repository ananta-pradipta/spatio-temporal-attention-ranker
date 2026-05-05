"""Graph construction for MTGN Phase 1 training.

Phase 1 edges are rolling return correlations plus StockTwits co-mention
edges. For the first end-to-end run we use a single static correlation
edge set computed from the training-period returns; this matches
DySTAGE's correlation-edge convention. A refresh cadence and co-mention
integration are Phase 1b refinements.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GraphConfig:
    correlation_window_days: int = 60
    correlation_threshold: float = 0.3
    max_degree: int | None = 30   # cap neighbors per node for tractability


def build_correlation_edges(
    x: np.ndarray, cfg: GraphConfig | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return edge_index [2, E] and edge_weight [E] from daily returns.

    `x[:, :, 0]` is assumed to be the log_return column (the first entry
    in panel.FEATURE_COLS).
    """
    cfg = cfg or GraphConfig()
    returns = x[:, :, 0]                                  # [T, N]
    r = returns[-cfg.correlation_window_days :]           # trailing window
    # Pearson correlation across tickers on the window
    r = r - r.mean(axis=0, keepdims=True)
    std = r.std(axis=0, ddof=0, keepdims=True).clip(min=1e-8)
    r_norm = r / std
    corr = (r_norm.T @ r_norm) / r.shape[0]               # [N, N]
    np.fill_diagonal(corr, 0.0)

    # Symmetric threshold
    mask = np.abs(corr) > cfg.correlation_threshold
    if cfg.max_degree is not None:
        # Keep top-K by absolute correlation per row
        abs_corr = np.where(mask, np.abs(corr), 0.0)
        keep = np.zeros_like(mask)
        for i in range(corr.shape[0]):
            row = abs_corr[i]
            if row.sum() == 0:
                continue
            k = min(cfg.max_degree, int((row > 0).sum()))
            top_idx = np.argpartition(-row, k - 1)[:k]
            keep[i, top_idx] = True
        mask = keep | keep.T  # re-symmetrize

    src, dst = np.nonzero(mask)
    edge_index = np.stack([src, dst], axis=0).astype(np.int64)
    edge_weight = corr[src, dst].astype(np.float32)
    return edge_index, edge_weight
