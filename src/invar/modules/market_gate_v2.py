"""MarketGateV2 for InVAR-v6.

Extends the v4 MarketGate to consume a richer macro state vector
(typically from MacroWindowEncoder) instead of the raw macro_dim
last-step vector, and supports three gate forms (Section 4):

  - ``softmax_F``        : alpha = F * softmax(logits / beta).
                           Sums to F, forces feature competition.
                           Identity at init via zero-init projection.
  - ``sigmoid_centered`` : alpha = sigmoid(logits) / mean(sigmoid).
                           Mean = 1 by construction; per-feature scaling
                           without competition.
  - ``sigmoid_residual`` : alpha = 1 + scale * tanh(logits).
                           Symmetric perturbation around 1; scale=0.25
                           keeps alpha in [0.75, 1.25] at saturation.

Output shape: alpha ``(B, F)``. Applied via Hadamard product over the
F dimension on the input panel ``x``.

The v4 MarketGate is preserved unchanged; v6 callers select V2 via
``cfg.use_market_gate_v2``.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MarketGateV2(nn.Module):
    def __init__(
        self,
        num_features: int = 26,
        macro_state_dim: int = 64,
        gate_form: str = "softmax_F",
        hidden_dim: int = 64,
        beta_init: float = 2.0,
        learn_beta: bool = True,
        dropout: float = 0.1,
        identity_init: bool = True,
        residual_scale: float = 0.25,
    ) -> None:
        super().__init__()
        if gate_form not in ("softmax_F", "sigmoid_centered", "sigmoid_residual"):
            raise ValueError(f"unknown gate_form: {gate_form!r}")
        self.num_features = num_features
        self.macro_state_dim = macro_state_dim
        self.gate_form = gate_form
        self.residual_scale = residual_scale

        if hidden_dim and hidden_dim > 0:
            self.proj = nn.Sequential(
                nn.Linear(macro_state_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_features),
            )
            final = self.proj[-1]
        else:
            self.proj = nn.Linear(macro_state_dim, num_features)
            final = self.proj
        if identity_init:
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

        if learn_beta:
            self.log_beta = nn.Parameter(torch.tensor(float(beta_init)).log())
        else:
            self.register_buffer(
                "log_beta", torch.tensor(float(beta_init)).log(),
            )
        self.learn_beta = learn_beta

    @property
    def beta(self) -> torch.Tensor:
        return self.log_beta.exp().clamp(min=1.0e-3, max=20.0)

    def forward(
        self, x: torch.Tensor, macro_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Args:
            x           : ``(B, N, L, F)`` panel feature tensor.
            macro_state : ``(B, macro_state_dim)`` from
                          MacroWindowEncoder.

        Returns ``(x_gated, alpha)`` with ``alpha : (B, F)``.
        """
        if macro_state.dim() == 1:
            macro_state = macro_state.unsqueeze(0)
        logits = self.proj(macro_state)                         # (B, F)
        if self.gate_form == "softmax_F":
            alpha = torch.softmax(logits / self.beta, dim=-1) * self.num_features
        elif self.gate_form == "sigmoid_centered":
            raw = torch.sigmoid(logits)
            alpha = raw / raw.mean(dim=-1, keepdim=True).clamp(min=1.0e-6)
        else:  # sigmoid_residual
            alpha = 1.0 + self.residual_scale * torch.tanh(logits)
        if x.dim() == 4:
            x_gated = x * alpha.unsqueeze(1).unsqueeze(2)
        elif x.dim() == 3:
            x_gated = x * alpha.unsqueeze(1)
        else:
            raise ValueError(f"unsupported x dim: {x.dim()}")
        return x_gated, alpha


__all__ = ["MarketGateV2"]
