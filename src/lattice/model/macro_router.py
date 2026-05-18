"""MacroMoERouter: mixture-of-experts over macro state.

Per spec section 6.5.

Given a 24-d macro state, a router produces softmax weights over n_experts
experts. Each expert is a small MLP that produces a per-stock residual.
Auxiliary load-balance loss prevents expert collapse.

Init: router weights small so the routing distribution is approximately
uniform at iteration 0.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class MacroMoERouterConfig:
    macro_dim: int = 24
    n_experts: int = 4
    d_model: int = 128
    balance_loss_weight: float = 0.01
    hidden_dim: int = 32


class _Expert(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

    def forward(self, z: Tensor) -> Tensor:
        return self.net(z)


class MacroMoERouter(nn.Module):
    """Macro-state-conditioned mixture of expert residuals."""

    def __init__(self, cfg: MacroMoERouterConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or MacroMoERouterConfig()
        self.cfg = cfg
        self.router = nn.Sequential(
            nn.Linear(cfg.macro_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.n_experts),
        )
        with torch.no_grad():
            # Small final-layer weights so routing is near-uniform at init.
            self.router[-1].weight.mul_(0.01)
            self.router[-1].bias.zero_()
        self.experts = nn.ModuleList([
            _Expert(cfg.d_model) for _ in range(cfg.n_experts)
        ])

    def forward(self, macro_state: Tensor, z: Tensor) -> tuple[Tensor, Tensor]:
        """Apply MoE residual over per-stock embeddings.

        Args:
            macro_state: [B, macro_dim].
            z: [B, N, d_model] per-stock embeddings.

        Returns:
            (residual, balance_loss).
            residual: [B, N, d_model] weighted sum of expert outputs.
            balance_loss: scalar tensor; sum over experts of (fraction-of-tokens
                routed to expert) * (mean routing probability for that expert).
        """
        B, N, D = z.shape
        E = self.cfg.n_experts
        gate_logits = self.router(macro_state)            # [B, E]
        gate_probs = torch.softmax(gate_logits, dim=-1)   # [B, E]

        # Per-expert outputs
        expert_outputs = torch.stack(
            [expert(z) for expert in self.experts], dim=2,  # [B, N, E, D]
        )
        # Apply per-day gate (broadcast across N)
        weighted = expert_outputs * gate_probs.unsqueeze(1).unsqueeze(-1)  # [B, N, E, D]
        residual = weighted.sum(dim=2)                                       # [B, N, D]

        # Balance loss: Switch Transformer recipe.
        # f_i = fraction of tokens routed (argmax) to expert i.
        # P_i = mean routing probability for expert i.
        # loss = E * sum_i (f_i * P_i)
        argmax_expert = gate_probs.argmax(dim=-1)         # [B]
        f = torch.zeros(E, device=z.device)
        for i in range(E):
            f[i] = (argmax_expert == i).float().mean()
        P = gate_probs.mean(dim=0)                          # [E]
        balance_loss = E * (f * P).sum()

        return residual, balance_loss


__all__ = ["MacroMoERouter", "MacroMoERouterConfig"]
