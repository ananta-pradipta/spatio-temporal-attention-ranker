"""Cross-sectional LayerNorm.

Normalizes a batch of per-ticker hidden states across ACTIVE tickers
within a single day, with learnable per-feature scale and shift. This
was critical to unblocking MTGN training in Combined v1; both MARS and
STAR depend on it.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn


class CrossSectionalLayerNorm(nn.Module):
    def __init__(self, hidden_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(hidden_dim))
        self.shift = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, h: Tensor, active_mask: Tensor) -> Tensor:
        """h: [N, D], active_mask: [N] bool. Returns h_out: [N, D], inactive rows zeroed."""
        if active_mask.sum() < 2:
            return h
        h_act = h[active_mask]
        mu = h_act.mean(dim=0, keepdim=True)
        sd = h_act.std(dim=0, keepdim=True).clamp(min=self.eps)
        h_out = torch.zeros_like(h)
        h_out[active_mask] = ((h_act - mu) / sd) * self.scale + self.shift
        return h_out


__all__ = ["CrossSectionalLayerNorm"]
