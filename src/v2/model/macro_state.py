"""MacroStateEncoder for DOW-epiSTAR v2.

Per spec Section D, encodes the per-day macro feature vector into:
    - macro_state[t] in R^d_macro    (32-d main embedding)
    - macro_gate_state[t] in R^16    (low-dim summary for gating)

The 16-d gate state feeds lambda_macro, alpha_rate gate, and graph
source gate.
"""
from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn


@dataclass
class MacroStateConfig:
    """Hyperparameters for the macro state encoder."""

    input_dim: int = 0
    hidden_dim: int = 64
    out_dim: int = 32
    gate_state_dim: int = 16
    dropout: float = 0.1


class MacroStateEncoder(nn.Module):
    """Two-headed encoder producing (macro_state, macro_gate_state)."""

    def __init__(self, cfg: MacroStateConfig) -> None:
        super().__init__()
        assert cfg.input_dim > 0, "input_dim must be set"
        self.cfg = cfg
        self.shared = nn.Sequential(
            nn.LayerNorm(cfg.input_dim),
            nn.Linear(cfg.input_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )
        self.main_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.out_dim),
            nn.LayerNorm(cfg.out_dim),
        )
        self.gate_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.gate_state_dim),
            nn.LayerNorm(cfg.gate_state_dim),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """[..., input_dim] -> ([..., out_dim], [..., gate_state_dim])."""
        h = self.shared(x)
        return self.main_head(h), self.gate_head(h)


__all__ = ["MacroStateEncoder", "MacroStateConfig"]
