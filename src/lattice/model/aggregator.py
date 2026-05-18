"""CrossSectionalAggregator: StockMixer-style stock mixing.

Per spec section 6.3.

Three mixing pathways per layer:
  - Stock-to-market: mean-pool active per-stock embeddings to a market vector.
  - Market-to-stock: broadcast market vector to each stock and concatenate.
  - Stock-to-stock: multi-head attention over the active set with masking.

The active universe is variable per day; uses masking, not zero-padding,
in the mixing block.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class CrossSectionalAggregatorConfig:
    d_model: int = 128
    n_mixing_layers: int = 2
    n_attn_heads: int = 4
    dropout: float = 0.1


class _MixingLayer(nn.Module):
    def __init__(self, cfg: CrossSectionalAggregatorConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.market_concat_proj = nn.Linear(2 * cfg.d_model, cfg.d_model)
        self.s2s_attn = nn.MultiheadAttention(
            embed_dim=cfg.d_model, num_heads=cfg.n_attn_heads,
            dropout=cfg.dropout, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model * 4),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model * 4, cfg.d_model),
        )

    def forward(self, z: Tensor, active_mask: Tensor) -> Tensor:
        """One mixing layer.

        Args:
            z: [B, N, d_model] per-stock embeddings.
            active_mask: [B, N] bool, True where ticker is active on that day.
        """
        B, N, D = z.shape
        # Stock-to-market: masked mean
        m = active_mask.unsqueeze(-1).float()  # [B, N, 1]
        n_active = m.sum(dim=1).clamp(min=1.0)  # [B, 1]
        market = (z * m).sum(dim=1) / n_active  # [B, D]
        # Market-to-stock: concat + linear
        market_b = market.unsqueeze(1).expand(B, N, D)  # [B, N, D]
        z_with_market = torch.cat([z, market_b], dim=-1)
        z_after_m2s = self.market_concat_proj(z_with_market)  # [B, N, D]
        z = self.norm1(z + z_after_m2s)

        # Stock-to-stock attention with masking
        # MultiheadAttention key_padding_mask wants True where to MASK OUT.
        # So we pass ~active_mask.
        key_padding = ~active_mask  # [B, N]
        attn_out, _ = self.s2s_attn(
            z, z, z, key_padding_mask=key_padding, need_weights=False,
        )
        z = self.norm2(z + attn_out + self.ffn(attn_out))
        # Zero out inactive cells so they don't leak into downstream.
        z = z * m
        return z


class CrossSectionalAggregator(nn.Module):
    """Aggregator that mixes per-stock embeddings across the cross-section."""

    def __init__(self, cfg: CrossSectionalAggregatorConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or CrossSectionalAggregatorConfig()
        self.cfg = cfg
        self.layers = nn.ModuleList([
            _MixingLayer(cfg) for _ in range(cfg.n_mixing_layers)
        ])

    def forward(self, z: Tensor, active_mask: Tensor) -> Tensor:
        """Refine per-stock embeddings via mixing.

        Args:
            z: [B, N, d_model] per-stock embeddings.
            active_mask: [B, N] bool active mask.

        Returns:
            [B, N, d_model] refined embeddings (zero on inactive cells).
        """
        for layer in self.layers:
            z = layer(z, active_mask)
        return z


__all__ = ["CrossSectionalAggregator", "CrossSectionalAggregatorConfig"]
