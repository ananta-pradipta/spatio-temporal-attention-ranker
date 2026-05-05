"""Regime-conditioned softmax temperature for RT-CSGA.

Per spec Section 5.2 of `docs/specs/rt_csga_spec.md`:

    tau = softplus(a + b * r_t) + tau_floor

Two learnable scalars ``a`` (intercept, init 0.0) and ``b`` (slope on
the regime indicator r_t, init 1.0; positive sign expected). The
softplus keeps tau strictly positive; the 1e-3 floor prevents
softmax numerical instability on extreme rate-vol days.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class RegimeTemperature(nn.Module):
    """tau = softplus(a + b * r_t) + tau_floor."""

    def __init__(
        self,
        a_init: float = 0.0,
        b_init: float = 1.0,
        tau_floor: float = 1e-3,
    ) -> None:
        super().__init__()
        self.a = nn.Parameter(torch.tensor(a_init, dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor(b_init, dtype=torch.float32))
        self.tau_floor = float(tau_floor)

    def forward(self, r_t: Tensor) -> Tensor:
        """[scalar or shape ()] -> scalar tau (>= tau_floor)."""
        raw = self.a + self.b * r_t
        return F.softplus(raw) + self.tau_floor


__all__ = ["RegimeTemperature"]
