"""Snapshot attention — Stage 3 of MARS.

A fixed-size learnable bank of K snapshot vectors (regime archetypes)
that each ticker's temporal embedding cross-attends to. The bank is
initialized from k-means centroids on training-set spatial embeddings
(see `snapshot_init.py`) and further trained end-to-end.

Design notes (per MARS/STAR memo Section 4.2 and debugging lessons):
- Unlike MTGN's episodic store, the bank has a FIXED small size (K=32).
  This avoids the noisy-retrieval and moving-target pathologies that
  sank MTGN-Full's cross-entity attention.
- The contribution is mixed into the backbone representation via a
  learnable scalar `alpha` (initialized small so the backbone dominates
  early in training) and a residual + LayerNorm.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class SnapshotAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_snapshots: int, alpha_init: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_snapshots = num_snapshots
        # Snapshot bank: [K, D]. Small-scale init; k-means overrides if available.
        self.snapshots = nn.Parameter(torch.randn(num_snapshots, hidden_dim) * 0.02)
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def forward(self, h: Tensor) -> tuple[Tensor, Tensor]:
        """h: [N, D]. Returns (z: [N, D], attn_weights: [N, K])."""
        q = self.W_q(h)                                          # [N, D]
        k = self.W_k(self.snapshots)                             # [K, D]
        v = self.W_v(self.snapshots)                             # [K, D]
        logits = q @ k.transpose(-1, -2) / math.sqrt(self.hidden_dim)   # [N, K]
        attn = F.softmax(logits, dim=-1)
        ctx = attn @ v                                           # [N, D]
        z = self.norm(h + self.alpha * ctx)
        return z, attn


__all__ = ["SnapshotAttention"]
