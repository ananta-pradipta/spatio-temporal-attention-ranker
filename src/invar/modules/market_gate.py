"""MASTER-style market-guided gate for InVAR v4 + Phase 4 ablations.

Reference:
  Li et al. (2024). MASTER: Market-Guided Stock Transformer for Stock
  Price Forecasting. AAAI 2024. Section 3.1, equation 1.

The gate applies a per-feature multiplicative scaling alpha(m) computed
from the macro state vector m.

Two gate forms are supported (Phase 4 ablation matrix Section 8):
  - "softmax_F": alpha = F * softmax(W m + b / beta). Sums to F.
                 Forces feature competition. Identity at init via
                 zero-init proj.
  - "sigmoid":   alpha = 2 * sigmoid(W m + b). Range [0, 2]. No
                 competition. Identity at init via zero-init proj
                 (since 2*sigmoid(0) = 1).

The output dimension is determined by ``num_features`` and is the same
for both forms. Apply alpha element-wise to the input tensor.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MarketGate(nn.Module):
    """alpha(m) = F * softmax_beta(W m + b); x_tilde = alpha (Hadamard) x."""

    def __init__(
        self,
        num_features: int,
        market_dim: int,
        beta_init: float = 2.0,
        learn_beta: bool = True,
        hidden_dim: int = 0,
        dropout: float = 0.0,
        gate_form: str = "softmax_F",   # "softmax_F" or "sigmoid"
    ) -> None:
        super().__init__()
        if gate_form not in ("softmax_F", "sigmoid"):
            raise ValueError(f"unknown gate_form: {gate_form!r}")
        self.gate_form = gate_form
        self.F = num_features
        self.Fp = market_dim
        if hidden_dim and hidden_dim > 0:
            self.proj = nn.Sequential(
                nn.Linear(market_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_features),
            )
            final = self.proj[-1]
        else:
            self.proj = nn.Linear(market_dim, num_features)
            final = self.proj
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        if learn_beta:
            self.log_beta = nn.Parameter(torch.tensor(float(beta_init)).log())
        else:
            self.register_buffer(
                "log_beta", torch.tensor(float(beta_init)).log()
            )
        self.learn_beta = learn_beta

    @property
    def beta(self) -> torch.Tensor:
        return self.log_beta.exp().clamp(min=1.0e-3, max=20.0)

    def forward(
        self, x: torch.Tensor, m: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Args:
            x : ``(B, N, T, F)`` ticker-batch panel feature tensor.
            m : ``(B, Fp)`` or ``(B, T, Fp)`` macro state. If 3-d, use
                the last time step (the prediction date).

        Returns ``(x_tilde, alpha)`` where alpha is ``(B, F)``.
        """
        if m.dim() == 3:
            m = m[:, -1, :]
        logits = self.proj(m)
        if self.gate_form == "softmax_F":
            alpha = torch.softmax(logits / self.beta, dim=-1) * self.F
        else:  # sigmoid
            # 2 * sigmoid -> range [0, 2], identity at zero-init logits.
            alpha = 2.0 * torch.sigmoid(logits)
        if x.dim() == 4:
            # (B, N, T, F) -> broadcast over N and T.
            x_tilde = x * alpha.unsqueeze(1).unsqueeze(2)
        elif x.dim() == 3:
            # (B, N, d) -> broadcast over N (post-tokenizer location).
            x_tilde = x * alpha.unsqueeze(1)
        else:
            raise ValueError(f"unsupported x dim: {x.dim()}")
        return x_tilde, alpha


__all__ = ["MarketGate"]
