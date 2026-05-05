"""Feature-wise Linear Modulation (FiLM) — Stage 3 of STAR.

A small Multi-Layer Perceptron (MLP) projects a 7-dim per-day risk
vector into (gamma, beta) per-hidden-dim, and each active ticker's
representation is modulated as `gamma * h + beta`. This is the
architecturally-forced risk-aware signal (see `mars-and-star-
implementation.md` Section 5.3).

Unlike plain concatenation of risk features (which a ranking model
cannot use because the scalar is uniform across the cross-section),
FiLM modulation survives cross-sectional ranking because each hidden
unit is multiplied/shifted in a regime-dependent way.
"""
from __future__ import annotations

from torch import Tensor, nn


class RiskFiLM(nn.Module):
    def __init__(self, risk_dim: int, hidden_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(risk_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.W_gamma = nn.Linear(hidden_dim, hidden_dim)
        self.W_beta  = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, h: Tensor, risk_features: Tensor) -> Tensor:
        """h: [N, D], risk_features: [risk_dim]. Returns [N, D]."""
        e = self.encoder(risk_features)      # [D]
        gamma = self.W_gamma(e)              # [D]
        beta  = self.W_beta(e)               # [D]
        return h * gamma.unsqueeze(0) + beta.unsqueeze(0)


__all__ = ["RiskFiLM"]
