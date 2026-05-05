"""DurationExposureEncoder for DOW-epiSTAR v2.

Per spec Section C, encodes the per-(day, ticker) feature vector
(fundamentals + risk/liquidity + social + age/history + rolling macro
betas) into a low-dimensional duration-exposure embedding. The
embedding is learned end-to-end with the ranking loss; no direct
supervision.

Interpretation hint: post-training the embedding should cluster
tickers by (high-duration biotech / financing-fragile / cash-rich
defensive / sentiment-driven catalyst / large-cap defensive).
"""
from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn


@dataclass
class DurationExposureConfig:
    """Hyperparameters for the duration encoder."""

    input_dim: int = 0          # set at construction time
    hidden_dim: int = 64
    out_dim: int = 32
    dropout: float = 0.1


class DurationExposureEncoder(nn.Module):
    """LayerNorm + Linear + GELU + Dropout + Linear + LayerNorm."""

    def __init__(self, cfg: DurationExposureConfig) -> None:
        super().__init__()
        assert cfg.input_dim > 0, "input_dim must be set"
        self.cfg = cfg
        self.net = nn.Sequential(
            nn.LayerNorm(cfg.input_dim),
            nn.Linear(cfg.input_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.out_dim),
            nn.LayerNorm(cfg.out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """[..., input_dim] -> [..., out_dim]."""
        return self.net(x)


__all__ = ["DurationExposureEncoder", "DurationExposureConfig"]
