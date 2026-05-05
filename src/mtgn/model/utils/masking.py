"""Inactive-node masking utilities.

Prevents pre-Initial-Public-Offering (IPO) or delisted tickers from
contributing corrupted (nonzero post-normalization) messages to active-
node representations via the spatial Graph Attention Network (GAT).
"""
from __future__ import annotations

from torch import Tensor


def mask_features_and_edges(features: Tensor, edge_index: Tensor,
                            active_mask: Tensor) -> tuple[Tensor, Tensor]:
    """Zero features for inactive tickers; filter edges to only active-active."""
    af = active_mask.to(dtype=features.dtype).unsqueeze(-1)
    features_m = features * af
    ea = active_mask[edge_index[0]] & active_mask[edge_index[1]]
    return features_m, edge_index[:, ea]


__all__ = ["mask_features_and_edges"]
