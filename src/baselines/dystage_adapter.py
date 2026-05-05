"""DySTAGE adapter for the v2 RAG-STAR baseline protocol.

Wraps the vendored DySTAGE architecture (NJIT-Fintech-Lab, ICAIF 2024,
Gu et al.) with our 244-ticker biotech panel construction. Builds the
per-day graph representation that the upstream model expects:

    - Adjacency A_t: undirected, sparse, Pearson correlation graph
      thresholded at |rho| >= gamma over a 60-day window
    - Edge features E_t: multi-scale Pearson at windows {5, 10, 20, 60}
      (4 scales)
    - Shortest path L_t: BFS on the unweighted adjacency, capped at 10
      (the upstream SpatialEncoding embedding size)
    - Node features X_t: 22-d panel features with the first column
      reserved as the prior-day return (the upstream model treats the
      first feature column as the supervised target)
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix


@dataclass
class DySTAGEGraphConfig:
    """Hyperparameters for per-day graph construction."""

    corr_window: int = 60          # primary correlation window for adjacency
    corr_threshold: float = 0.3    # |rho| threshold for edge inclusion
    edge_scales: tuple = (5, 10, 20, 60)  # multi-scale edge features
    shortest_path_cap: int = 10    # truncate shortest paths beyond this
    min_overlap: int = 5           # minimum non-NaN samples for valid corr


def _pearson_one_window(returns: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Pearson correlation over a single window. NaN-safe.

    Args:
        returns: [W, N] log-returns slice for window of length W
        mask: [W, N] tradable mask for the same slice

    Returns:
        rho: [N, N] Pearson correlations (0 where overlap insufficient)
    """
    W, N = returns.shape
    rho = np.zeros((N, N), dtype=np.float32)
    if W < 2:
        return rho
    masked = np.where(mask, returns, 0.0)
    valid_count = mask.sum(axis=0).astype(np.float32)
    centered = masked - masked.mean(axis=0, keepdims=True)
    sd = centered.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-8, 1e-8, sd)
    norm = centered / sd
    rho_full = (norm.T @ norm) / float(W)
    overlap = np.minimum(valid_count[:, None], valid_count[None, :])
    rho_full = np.where(overlap >= 5, rho_full, 0.0)
    np.fill_diagonal(rho_full, 0.0)
    return rho_full.astype(np.float32)


def _build_adjacency(rho: np.ndarray, threshold: float) -> sp.csr_matrix:
    """Threshold |rho| >= gamma to a binary undirected adjacency. Diagonal zero."""
    A = (np.abs(rho) >= threshold).astype(np.float32)
    np.fill_diagonal(A, 0.0)
    return sp.csr_matrix(A)


def _row_normalise(adj: sp.csr_matrix) -> sp.csr_matrix:
    """Row-normalise adjacency for the upstream `_preprocess_adj` style."""
    rowsum = np.asarray(adj.sum(axis=1)).flatten()
    rowsum = np.where(rowsum > 0, rowsum, 1.0)
    r_inv = sp.diags(1.0 / rowsum)
    return r_inv.dot(adj).tocsr()


def _shortest_paths(adj_dense: np.ndarray, cap: int) -> np.ndarray:
    """All-pairs unweighted shortest path lengths via BFS from each node.

    Returns an [N, N] integer matrix of path lengths, with unreachable
    pairs set to `cap` (matching the upstream SpatialEncoding embedding
    bound of 11 = cap + 1).
    """
    n = adj_dense.shape[0]
    spl = np.full((n, n), cap, dtype=np.int64)
    np.fill_diagonal(spl, 0)
    binary = (adj_dense > 0).astype(np.int8)
    for src in range(n):
        if binary[src].sum() == 0:
            continue
        visited = np.zeros(n, dtype=bool)
        visited[src] = True
        queue = deque([(src, 0)])
        while queue:
            node, d = queue.popleft()
            if d >= cap:
                continue
            for neighbour in np.flatnonzero(binary[node]):
                if not visited[neighbour]:
                    visited[neighbour] = True
                    spl[src, neighbour] = d + 1
                    queue.append((neighbour, d + 1))
    return spl


def build_day_graph(
    t: int,
    log_returns: np.ndarray,
    tradable: np.ndarray,
    node_features: np.ndarray,
    cfg: DySTAGEGraphConfig,
) -> Data:
    """Construct one day's `torch_geometric.data.Data` for DySTAGE.

    Args:
        t: trading-day index (0 .. T-1).
        log_returns: [T, N] log-return panel.
        tradable: [T, N] boolean active mask.
        node_features: [T, N, F] node feature tensor (already z-scored).
        cfg: graph-construction hyperparameters.

    Returns:
        A torch_geometric Data object with x, edge_index, edge_weight,
        edge_feat, and shortest_path_len attributes consistent with the
        vendored DySTAGE expectations.
    """
    T, N = log_returns.shape
    cw = cfg.corr_window
    if t < cw - 1:
        # too early in the panel: degenerate empty graph
        x = torch.from_numpy(node_features[t]).float()
        adj = sp.csr_matrix((N, N), dtype=np.float32)
        edge_index, edge_weight = from_scipy_sparse_matrix(adj)
        edge_feat = torch.zeros(N, N, len(cfg.edge_scales), dtype=torch.float32)
        spl = torch.full((N, N), cfg.shortest_path_cap, dtype=torch.long)
        return Data(
            x=x, edge_index=edge_index, edge_weight=edge_weight.float(),
            edge_feat=edge_feat, shortest_path_len=spl,
        )

    rho_primary = _pearson_one_window(
        log_returns[t - cw + 1: t + 1], tradable[t - cw + 1: t + 1],
    )
    adj = _build_adjacency(rho_primary, cfg.corr_threshold)
    adj_norm = _row_normalise(adj)
    edge_index, edge_weight = from_scipy_sparse_matrix(adj_norm)

    # Multi-scale edge features.
    edge_feat = np.zeros((N, N, len(cfg.edge_scales)), dtype=np.float32)
    for s, w in enumerate(cfg.edge_scales):
        if t < w - 1:
            continue
        rho_s = _pearson_one_window(
            log_returns[t - w + 1: t + 1], tradable[t - w + 1: t + 1],
        )
        edge_feat[:, :, s] = rho_s

    spl = _shortest_paths(adj.toarray(), cap=cfg.shortest_path_cap)

    x = torch.from_numpy(node_features[t]).float()
    return Data(
        x=x, edge_index=edge_index, edge_weight=edge_weight.float(),
        edge_feat=torch.from_numpy(edge_feat),
        shortest_path_len=torch.from_numpy(spl),
    )


@dataclass
class DySTAGEArgs:
    """Container matching the upstream `args.<attr>` access pattern."""

    hist_time_steps: int = 12
    spatial: bool = True
    centrality: bool = True
    edge: bool = True
    n_heads: int = 4
    node_dim: int = 64
    attention_layers: int = 2
    temporal_head_config: str = "4"
    temporal_layer_config: str = "64"
    temporal_drop: float = 0.5
    residual: bool = True


__all__ = [
    "DySTAGEGraphConfig",
    "DySTAGEArgs",
    "build_day_graph",
]
