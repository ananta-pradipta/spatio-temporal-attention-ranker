"""TemporalStockTokenizer: per-ticker (Tw, F) -> d via shared MLP."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class TemporalTokenizerConfig:
    n_features: int = 26
    lookback: int = 20
    d_model: int = 128
    hidden: int = 256
    dropout: float = 0.10


class TemporalStockTokenizer(nn.Module):
    """Convert each stock's 20-day feature history into one d-dim token.

    Forward:
      x_window : (B, Tw, N, F)
      -> permute to (B, N, Tw, F) -> flatten to (B, N, Tw*F)
      -> shared MLP -> (B, N, d_model)
    """

    def __init__(self, cfg: TemporalTokenizerConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or TemporalTokenizerConfig()
        self.cfg = cfg
        self.net = nn.Sequential(
            nn.Linear(cfg.lookback * cfg.n_features, cfg.hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
        )

    def forward(self, x_window: Tensor) -> Tensor:
        B, Tw, N, F = x_window.shape
        x = x_window.permute(0, 2, 1, 3).contiguous()
        x = x.view(B, N, Tw * F)
        return self.net(x)


__all__ = ["TemporalStockTokenizer", "TemporalTokenizerConfig"]
