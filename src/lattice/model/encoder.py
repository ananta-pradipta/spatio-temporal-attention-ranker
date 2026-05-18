"""PerStockEncoder: iTransformer-style variate attention.

Per spec section 6.2.

Each of the 30 panel feature variates becomes a token over the 60-day
lookback window, projected to d_model. Self-attention runs across the 30
variate tokens. Output is a per-stock embedding via mean-pool over tokens.

Reference: docs/lattice_design_rationale.md section "iTransformer encoder".
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class PerStockEncoderConfig:
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1
    n_features: int = 30
    lookback: int = 60


class PerStockEncoder(nn.Module):
    """One stock-day -> d_model embedding via variate-axis attention.

    Input  shape: [B, N, T, F]   (batch_days, n_active_tickers, lookback, n_features)
    Output shape: [B, N, d_model]
    """

    def __init__(self, cfg: PerStockEncoderConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or PerStockEncoderConfig()
        self.cfg = cfg
        # Project each variate's 60-day series to a d_model token.
        self.variate_proj = nn.Linear(cfg.lookback, cfg.d_model)
        # Variate-axis transformer.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.n_heads,
            dim_feedforward=cfg.d_model * 4, dropout=cfg.dropout,
            batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_layers)
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(self, x: Tensor) -> Tensor:
        """Encode per-stock features.

        Args:
            x: [B, N, T, F] feature tensor; missing-history cells should be
                zero-filled before passing here.

        Returns:
            [B, N, d_model] per-stock embeddings.
        """
        B, N, T, F = x.shape
        # Reshape to (B*N, F, T) so each variate is a row-token.
        x_flat = x.reshape(B * N, T, F).transpose(1, 2)  # [B*N, F, T]
        tokens = self.variate_proj(x_flat)               # [B*N, F, d_model]
        attended = self.transformer(tokens)              # [B*N, F, d_model]
        attended = self.norm(attended)
        # Mean-pool across the F variate tokens.
        z = attended.mean(dim=1)                          # [B*N, d_model]
        return z.view(B, N, self.cfg.d_model)


__all__ = ["PerStockEncoder", "PerStockEncoderConfig"]
