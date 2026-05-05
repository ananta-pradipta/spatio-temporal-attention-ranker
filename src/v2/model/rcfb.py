"""Regime-Conditional Factor Blending (RCFB).

Score-space blender between the DOW backbone's score_dow and a small
fixed-weight factor portfolio, gated by a regime descriptor. Per spec
``docs/specs/rcfb_implementation_prompt.md`` Section 3:

    score_dow_z[i]   = (score_dow[i] - mean_active) / (std_active + eps)
    factor_combo[i]  = w_r * z(rev) + w_s * z(soc) + w_c * z(cash) + w_v * z(lowvol)
    gate_in          = concat(m_state[16], cs_struct[4])
    g_t              = sigmoid(GateMLP(gate_in))
    final_score[i]   = (1 - g_t) * score_dow_z[i] + g_t * factor_combo[i]

The four factor weights are FIXED constants stored as a non-learnable
buffer (`register_buffer`); their value
(w_r=0.45, w_s=0.25, w_c=0.20, w_v=0.10) is drawn from per-factor
standalone IC analysis on the training window per the spec
appendix.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class RCFB(nn.Module):
    """Regime-conditional score-space blender (DOW score + fixed factor)."""

    def __init__(
        self,
        cs_struct_dim: int = 4,
        m_state_dim: int = 16,
        gate_hidden: int = 32,
        gate_init_bias: float = -3.0,
        factor_weights: tuple[float, float, float, float] = (0.45, 0.25, 0.20, 0.10),
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.cs_struct_dim = cs_struct_dim
        self.m_state_dim = m_state_dim
        self.eps = float(eps)

        # Validate factor weights: non-negative, sum to 1.0.
        if any(w < 0 for w in factor_weights):
            raise ValueError(f"factor_weights must be non-negative; got {factor_weights}")
        if abs(sum(factor_weights) - 1.0) > 1e-6:
            raise ValueError(
                f"factor_weights must sum to 1.0; got {factor_weights} "
                f"(sum={sum(factor_weights):.6f})"
            )
        # Stored as buffer (not parameter): moves with .to(device), not learned.
        self.register_buffer(
            "factor_weights",
            torch.tensor(factor_weights, dtype=torch.float32),
        )

        # Gate MLP: same shape as CSID v1's.
        gate_in = m_state_dim + cs_struct_dim
        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_in, gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, 1),
        )
        with torch.no_grad():
            self.gate_mlp[-1].bias.fill_(float(gate_init_bias))

    @staticmethod
    def _per_day_z(x: Tensor, eps: float) -> Tensor:
        """[N] -> [N] per-day z-score (std with unbiased=False)."""
        if x.numel() < 2:
            return x
        mu = x.mean()
        sd = x.std(unbiased=False)
        return (x - mu) / (sd + eps)

    def forward(
        self,
        score_dow: Tensor,    # [N], rank-head output for active tickers
        rev: Tensor,          # [N], -log_return_5d, raw
        soc: Tensor,          # [N],  st_volume_change_30d, raw
        cash: Tensor,         # [N],  cash_runway_q, raw
        lowvol: Tensor,       # [N], -realized_vol_60d, raw
        m_state: Tensor,      # [m_state_dim]
        cs_struct: Tensor,    # [cs_struct_dim]
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return (final_score, g_t, factor_combo)."""
        n = score_dow.shape[0]
        device = score_dow.device
        dtype = score_dow.dtype
        gate_in = torch.cat([m_state, cs_struct], dim=-1)
        g_t = torch.sigmoid(self.gate_mlp(gate_in)).squeeze()

        if n < 2:
            # Degenerate cross-section; pass score_dow through unchanged.
            return (
                score_dow,
                g_t,
                torch.zeros_like(score_dow),
            )

        score_dow_z = self._per_day_z(score_dow, self.eps)

        # Per-day z-score each factor input.
        rev_z = self._per_day_z(rev, self.eps)
        soc_z = self._per_day_z(soc, self.eps)
        cash_z = self._per_day_z(cash, self.eps)
        lowvol_z = self._per_day_z(lowvol, self.eps)

        # Fixed-weight combination.
        w = self.factor_weights.to(dtype)
        factor_combo = (
            w[0] * rev_z + w[1] * soc_z + w[2] * cash_z + w[3] * lowvol_z
        )

        final_score = (1.0 - g_t) * score_dow_z + g_t * factor_combo
        return final_score, g_t, factor_combo


__all__ = ["RCFB"]
