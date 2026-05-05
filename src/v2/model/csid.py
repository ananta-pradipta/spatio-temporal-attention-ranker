"""Cross-Sectional Idiosyncratic Distillation (CSID).

Single embedding-space residualisation layer applied between STAR's
``z_final`` and the rank head. Per spec
``docs/specs/csid_implementation_prompt.md`` (Section 3 of the source
document):

    z_bar = mean over active tickers of z_final
    v0    = z_bar / ||z_bar||
    v1    = (z_final - z_bar)^T @ ((z_final - z_bar) @ v0)    [power iter]
    v     = v1 / ||v1||
    beta  = z_final @ v
    alpha = sigmoid(GateMLP([m_state, cs_struct]))           [scalar in (0, 1)]
    z_idio = z_final - alpha * beta * v

The gate MLP final-layer bias is initialised to -3.0 so alpha starts
at sigmoid(-3) ~= 0.047, making the layer near-identity at init. The
gate is per-day (one alpha per day, not per ticker), reflecting the
spec's choice that the regime decides aggressiveness, not the ticker.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn


class CSID(nn.Module):
    """Embedding-space cross-sectional residualisation, regime-gated."""

    def __init__(
        self,
        embed_dim: int,
        cs_struct_dim: int,
        m_state_dim: int = 16,
        gate_hidden: int = 32,
        gate_init_bias: float = -3.0,
        power_iter_steps: int = 1,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.cs_struct_dim = cs_struct_dim
        self.m_state_dim = m_state_dim
        self.gate_hidden = gate_hidden
        self.power_iter_steps = max(1, int(power_iter_steps))
        self.eps = float(eps)

        # GateMLP: Linear(m_state + cs_struct, hidden) -> ReLU -> Linear(hidden, 1)
        gate_in = m_state_dim + cs_struct_dim
        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_in, gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, 1),
        )
        with torch.no_grad():
            self.gate_mlp[-1].bias.fill_(float(gate_init_bias))

    def forward(
        self,
        z_final: Tensor,    # [N, D]
        m_state: Tensor,    # [m_state_dim]
        cs_struct: Tensor,  # [cs_struct_dim]
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return (z_idio, alpha, v).

        z_final must already be sliced to active rows (no padding rows).
        """
        n, d = z_final.shape
        device = z_final.device
        dtype = z_final.dtype
        if n < 2:
            return (
                z_final,
                torch.zeros((), device=device, dtype=dtype),
                torch.zeros(d, device=device, dtype=dtype),
            )

        # Step 1: cross-sectional centroid.
        z_bar = z_final.mean(dim=0)                                  # [D]
        # Step 2: power iteration.
        v0 = z_bar / (z_bar.norm(p=2) + self.eps)                    # [D]
        diff = z_final - z_bar.unsqueeze(0)                          # [N, D]
        v = v0
        for _ in range(self.power_iter_steps):
            # v_next = diff^T (diff v) without forming the (D, D) cov.
            tmp = diff @ v                                           # [N]
            v_next = diff.transpose(-1, -2) @ tmp                    # [D]
            v = v_next / (v_next.norm(p=2) + self.eps)
        # Step 3: per-ticker projection scalar.
        beta = z_final @ v                                           # [N]
        # Step 4: regime-conditioned aggressiveness.
        gate_in = torch.cat([m_state, cs_struct], dim=-1)            # [16 + C]
        alpha_logit = self.gate_mlp(gate_in).squeeze()               # scalar
        alpha = torch.sigmoid(alpha_logit)
        # Step 5: residualise.
        z_idio = z_final - alpha * beta.unsqueeze(-1) * v.unsqueeze(0)
        return z_idio, alpha, v


__all__ = ["CSID"]
