"""K-means snapshot initialization for MARS.

Runs a single no-grad forward of the spatial Graph Attention Network
(GAT) over the training slice to collect per-(ticker, day) spatial
embeddings, then k-means clusters them into `num_snapshots` centroids
and writes those into the snapshot bank.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


@torch.no_grad()
def kmeans_snapshot_init(spatial_outputs: Tensor, num_snapshots: int,
                         random_state: int = 42) -> Tensor:
    """spatial_outputs: [M, D] flat tensor of collected embeddings (active cells only).
    Returns centroids: [num_snapshots, D] tensor."""
    from sklearn.cluster import KMeans
    x = spatial_outputs.detach().cpu().numpy().astype(np.float32)
    km = KMeans(n_clusters=num_snapshots, random_state=random_state,
                n_init=10, max_iter=100)
    km.fit(x)
    return torch.from_numpy(km.cluster_centers_.astype(np.float32))


__all__ = ["kmeans_snapshot_init"]
