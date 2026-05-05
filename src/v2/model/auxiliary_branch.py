"""Auxiliary virtue-of-complexity branch for RT-CSGA+Aux.

Per spec Section 5.3 of `docs/specs/rt_csga_spec.md`. A two-layer
MLP plus a score head, gated by a FIXED (non-trainable) buffer at
``sigmoid(gate_init) ~= 0.0067`` when ``gate_init = -5.0``.

Key contract:
    - The gate is a buffer, NOT a parameter (so the network cannot
      drive it to 0 or 1 and break the regularisation interpretation).
    - The branch returns exactly ``zeros(N)`` at inference (not
      training) -- this is the explicit drop required by spec
      Section 1.4 invariant 5.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn


class AuxiliaryBranch(nn.Module):
    """Mostly-unused MLP branch with fixed small gate, dropped at inference."""

    def __init__(
        self,
        in_dim: int = 128,
        hidden_dim: int = 64,
        gate_init: float = -5.0,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.score_head = nn.Linear(hidden_dim, 1)
        # FIXED gate: buffer, not parameter. sigmoid(-5) ~= 0.0067.
        gate_value = torch.sigmoid(torch.tensor(gate_init, dtype=torch.float32))
        self.register_buffer("gate", gate_value)

    def forward(self, h: Tensor) -> Tensor:
        """[N, in_dim] -> [N]. Returns zeros(N) when not training."""
        if not self.training:
            return torch.zeros(h.shape[0], device=h.device, dtype=h.dtype)
        z = self.mlp(h)
        return self.gate.to(z.dtype) * self.score_head(z).squeeze(-1)


__all__ = ["AuxiliaryBranch"]
