"""Patch construction for STAR.

Given a sparse edge set and edge weights, precompute top-N neighbors per
ticker. At train time, build a spatio-temporal (N+1, W, F) patch for each
ticker: row 0 is the ticker itself, rows 1..N are its top-N graph
neighbors, and the W dimension is the last-W-day history.

Per memo Section 5.3: the graph is a preprocessing step; the Transformer
processes the flat patch as a sequence. There is no graph message passing
inside the model.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


def precompute_top_neighbors(edge_index: np.ndarray, edge_weight: np.ndarray,
                             num_nodes: int, N: int = 8) -> np.ndarray:
    """Returns a [num_nodes, N] int64 array of top-N neighbor indices per node,
    ranked by summed edge weight. Self excluded. Missing slots filled with -1."""
    weights = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for k in range(edge_index.shape[1]):
        u, v = int(edge_index[0, k]), int(edge_index[1, k])
        w = float(edge_weight[k]) if edge_weight is not None and k < len(edge_weight) else 1.0
        weights[u, v] += w
        weights[v, u] += w
    np.fill_diagonal(weights, -np.inf)
    top = np.full((num_nodes, N), -1, dtype=np.int64)
    for i in range(num_nodes):
        row = weights[i]
        valid = np.where(np.isfinite(row))[0]
        if valid.size == 0:
            continue
        k = min(N, valid.size)
        top[i, :k] = valid[np.argsort(-row[valid])[:k]]
    return top


def build_patches(x_window: Tensor, mask_window: Tensor,
                  top_neighbors: Tensor, active_idx: Tensor) -> tuple[Tensor, Tensor]:
    """
    x_window:      [W, num_nodes, F] feature window
    mask_window:   [W, num_nodes] bool active mask
    top_neighbors: [num_nodes, N] int64, -1 for missing
    active_idx:    [num_active] int64 — tickers to build patches for

    Returns patches: [num_active, N+1, W, F], patch_mask: [num_active, N+1, W].
    """
    W, num_nodes, F = x_window.shape
    num_active = active_idx.shape[0]
    N = top_neighbors.shape[1]
    device = x_window.device

    # Gather self: [num_active, W, F]
    self_feat = x_window[:, active_idx, :].transpose(0, 1)   # [num_active, W, F]
    self_mask = mask_window[:, active_idx].transpose(0, 1)   # [num_active, W]

    # Gather neighbors: [num_active, N, W, F]
    nbr_idx = top_neighbors[active_idx]                      # [num_active, N]
    valid = nbr_idx >= 0                                     # [num_active, N]
    safe_idx = nbr_idx.clamp(min=0)                          # avoid -1 gather

    # x_window[:, safe_idx] is tricky — do it via advanced indexing
    # Want: patches[a, n, w, f] = x_window[w, safe_idx[a, n], f]
    flat_idx = safe_idx.view(-1)                             # [num_active*N]
    nbr_feat = x_window[:, flat_idx, :].view(W, num_active, N, F).permute(1, 2, 0, 3)
    nbr_mask = mask_window[:, flat_idx].view(W, num_active, N).permute(1, 2, 0)
    nbr_mask = nbr_mask & valid.unsqueeze(-1)

    # Stack self as row 0
    patches = torch.cat([self_feat.unsqueeze(1), nbr_feat], dim=1)   # [A, N+1, W, F]
    patch_mask = torch.cat([self_mask.unsqueeze(1), nbr_mask], dim=1)  # [A, N+1, W]
    return patches, patch_mask


__all__ = ["precompute_top_neighbors", "build_patches"]
