"""StockMixer baseline for InVAR Phase 3.

Reference:
  Fan et al. (2024). "StockMixer: A Simple yet Strong MLP-Based Architecture
  for Stock Price Forecasting." AAAI 2024.
  https://github.com/SJTU-DMTai/StockMixer

Adaptation to LATTICE's variable-N_t cross-section:

  - Time-mixing  : MLP over the L=60 lookback axis (per ticker, per channel).
  - Channel-mixing: MLP over the F=26 feature axis (per ticker, per timestep).
  - Cross-stock-mixing: permutation-invariant SET-style layer
                        (mean over N -> linear -> broadcast back), which
                        preserves the spirit of StockMixer (no attention,
                        no per-position parameters) while handling the
                        variable cross-section size that LATTICE produces.
  - 4 mixer blocks; final per-ticker mean-pool over (L, F) and ranking head.

Loss: same hybrid as InVAR with auxiliary heads disabled. Training uses
the generic ``train_baseline.py``.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class StockMixerConfig:
    n_features: int = 26
    lookback: int = 60
    d_model: int = 128
    time_hidden: int = 60
    channel_hidden: int = 64
    stock_hidden: int = 64
    n_blocks: int = 4
    dropout: float = 0.1
    head_hidden: int = 64


class _MlpBlock(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class _MixerBlock(nn.Module):
    """Time-mix + Channel-mix + Cross-stock-mix with residual connections.

    Input/output shape: (N, L, F).
    """

    def __init__(self, cfg: StockMixerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.norm_time = nn.LayerNorm(cfg.lookback)
        self.time_mix = _MlpBlock(cfg.lookback, cfg.time_hidden, cfg.dropout)
        self.norm_channel = nn.LayerNorm(cfg.n_features)
        self.channel_mix = _MlpBlock(cfg.n_features, cfg.channel_hidden, cfg.dropout)
        # Cross-stock: per (timestep, channel), mean pool over N then a
        # small MLP, then broadcast back. Permutation-invariant.
        self.norm_stock = nn.LayerNorm(cfg.n_features)
        self.stock_mlp = nn.Sequential(
            nn.Linear(cfg.n_features, cfg.stock_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.stock_hidden, cfg.n_features),
        )

    def forward(self, x: Tensor) -> Tensor:
        # Time-mix: (N, L, F) -> permute to (N, F, L), mix L, permute back.
        z = x.transpose(1, 2)                                  # (N, F, L)
        z = z + self.time_mix(self.norm_time(z))
        x_t = z.transpose(1, 2)                                 # (N, L, F)
        # Channel-mix
        x_c = x_t + self.channel_mix(self.norm_channel(x_t))
        # Cross-stock: per (L, F) cell, broadcast mean across N.
        z2 = self.norm_stock(x_c)
        # Mean over N axis with shape (L, F), then MLP -> (L, F).
        pool = z2.mean(dim=0)                                   # (L, F)
        cross = self.stock_mlp(pool)                            # (L, F)
        return x_c + cross.unsqueeze(0).expand_as(x_c)


class StockMixer(nn.Module):
    """Cross-sectional StockMixer with the InVAR forward signature."""

    def __init__(self, cfg: StockMixerConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or StockMixerConfig()
        self.blocks = nn.ModuleList(
            [_MixerBlock(self.cfg) for _ in range(self.cfg.n_blocks)]
        )
        self.norm_out = nn.LayerNorm(self.cfg.n_features)
        self.head = nn.Sequential(
            nn.Linear(self.cfg.n_features, self.cfg.head_hidden),
            nn.GELU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(self.cfg.head_hidden, 1),
        )

    def forward(
        self, features: Tensor, macro: Tensor, mask: Tensor,
        return_attn: bool = False,
    ) -> dict[str, Tensor]:
        x = features
        for block in self.blocks:
            x = block(x)
        x = self.norm_out(x)
        # Pool over (L, F): we mean-pool over L first (channels-last),
        # then map to scalar via the head.
        pooled = x.mean(dim=1)                                  # (N, F)
        y_hat = self.head(pooled).squeeze(-1)                   # (N,)
        m = mask.float()
        y_hat = y_hat * m
        return {
            "y_hat": y_hat,
            "regime_logits": torch.zeros(8, device=features.device),
            "vol_hat": torch.zeros_like(y_hat),
            "attn_weights": [] if return_attn else None,
        }


__all__ = ["StockMixerConfig", "StockMixer"]
