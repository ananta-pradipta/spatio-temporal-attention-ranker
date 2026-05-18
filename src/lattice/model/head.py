"""AdditiveResidualHead.

Per spec section 6.8.

Final score:

    y_hat = backbone_score(z_combined) + lambda_macro * residual_score(z_combined, expert_output)

where lambda_macro is a learnable scalar initialized via sigmoid(bias - 3) so
that lambda_macro is approximately 0.05 at iteration 0.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class AdditiveResidualHeadConfig:
    d_model: int = 128
    hidden_dim: int = 64
    dropout: float = 0.1
    lambda_macro_init: float = 0.05  # sigmoid(-3 + bias) approx


class AdditiveResidualHead(nn.Module):
    """Backbone score + macro-conditioned residual."""

    def __init__(self, cfg: AdditiveResidualHeadConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or AdditiveResidualHeadConfig()
        self.cfg = cfg
        self.backbone_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, 1),
        )
        self.residual_head = nn.Sequential(
            nn.Linear(cfg.d_model * 2, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, 1),
        )
        # lambda_macro scalar with init bias so sigmoid(bias) ~= lambda_macro_init.
        # sigmoid(-3) ~= 0.0474; close enough to 0.05.
        bias_init = torch.log(torch.tensor(cfg.lambda_macro_init / (1 - cfg.lambda_macro_init)))
        self.lambda_macro_bias = nn.Parameter(bias_init)

    def forward(self, z: Tensor, expert_output: Tensor) -> Tensor:
        """Compute final per-(day, ticker) score.

        Args:
            z: [B, N, d_model] combined embedding.
            expert_output: [B, N, d_model] residual from MoE router.

        Returns:
            [B, N] scalar score per active ticker.
        """
        backbone = self.backbone_head(z).squeeze(-1)               # [B, N]
        z_combined = torch.cat([z, expert_output], dim=-1)
        residual = self.residual_head(z_combined).squeeze(-1)       # [B, N]
        lambda_macro = torch.sigmoid(self.lambda_macro_bias)
        return backbone + lambda_macro * residual


__all__ = ["AdditiveResidualHead", "AdditiveResidualHeadConfig"]
