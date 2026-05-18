"""MASTER baseline for InVAR Phase 3.

Reference:
  Li et al. (2024). "MASTER: Market-Guided Stock Transformer for Stock Price
  Forecasting." AAAI 2024. https://github.com/SJTU-DMTai/MASTER

Adaptation to LATTICE:
  - Per-stock encoder: MLP over the (L, F) panel features producing a
    d-dim per-ticker token.
  - Market-guided gating: a macro encoder over the L=60 macro lookback
    produces a market vector m_t in R^d; per-ticker tokens are gated
    element-wise by sigmoid(W_g [v_i || m_t]).
  - Cross-stock self-attention: variate-axis Transformer encoder over
    the gated tokens.
  - Ranking head: linear over the per-ticker representation.

This adaptation preserves MASTER's two key innovations (market gating
and cross-stock attention) while fitting the LATTICE per-day batch
shape (N_t, L, F).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class MasterConfig:
    n_features: int = 26
    lookback: int = 60
    macro_dim: int = 24
    d_model: int = 128
    n_heads: int = 4
    ffn_dim: int = 256
    n_layers: int = 3
    dropout: float = 0.1
    encoder_hidden: int = 128
    macro_hidden: int = 64
    head_hidden: int = 64


class _PerStockEncoder(nn.Module):
    """(L, F) -> (d) per ticker. Two-layer MLP over flattened panel."""

    def __init__(self, cfg: MasterConfig) -> None:
        super().__init__()
        self.flat_dim = cfg.lookback * cfg.n_features
        self.mlp = nn.Sequential(
            nn.Linear(self.flat_dim, cfg.encoder_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.encoder_hidden, cfg.d_model),
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: (N, L, F) -> (N, d)
        return self.mlp(x.flatten(1))


class _MacroEncoder(nn.Module):
    """(L, F_macro) -> (d). MLP over flattened macro lookback."""

    def __init__(self, cfg: MasterConfig) -> None:
        super().__init__()
        self.flat_dim = cfg.lookback * cfg.macro_dim
        self.mlp = nn.Sequential(
            nn.Linear(self.flat_dim, cfg.macro_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.macro_hidden, cfg.d_model),
        )

    def forward(self, m: Tensor) -> Tensor:
        return self.mlp(m.flatten())


class _MarketGate(nn.Module):
    """sigmoid(W_g [v_i || m_t]) gating per ticker."""

    def __init__(self, cfg: MasterConfig) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(cfg.d_model + cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )

    def forward(self, v: Tensor, m_t: Tensor) -> Tensor:
        # v: (N, d), m_t: (d,)
        N = v.shape[0]
        cat = torch.cat([v, m_t.expand(N, -1)], dim=-1)
        g = torch.sigmoid(self.gate(cat))
        return v * g


class _CrossStockBlock(nn.Module):
    """Standard variate-axis self-attention block."""

    def __init__(self, cfg: MasterConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.attn = nn.MultiheadAttention(
            cfg.d_model, cfg.n_heads, dropout=cfg.dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.ffn_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.ffn_dim, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, v: Tensor) -> Tensor:
        # v: (N, d) -> treat as a single sequence of length N, batch 1.
        h = self.norm1(v).unsqueeze(0)                          # (1, N, d)
        a, _ = self.attn(h, h, h, need_weights=False)
        v = v + a.squeeze(0)
        v = v + self.ffn(self.norm2(v))
        return v


class Master(nn.Module):
    """Market-guided stock Transformer with cross-stock attention."""

    def __init__(self, cfg: MasterConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or MasterConfig()
        self.stock_enc = _PerStockEncoder(self.cfg)
        self.macro_enc = _MacroEncoder(self.cfg)
        self.gate = _MarketGate(self.cfg)
        self.blocks = nn.ModuleList(
            [_CrossStockBlock(self.cfg) for _ in range(self.cfg.n_layers)]
        )
        self.norm_out = nn.LayerNorm(self.cfg.d_model)
        self.head = nn.Sequential(
            nn.Linear(self.cfg.d_model, self.cfg.head_hidden),
            nn.GELU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(self.cfg.head_hidden, 1),
        )

    def forward(
        self, features: Tensor, macro: Tensor, mask: Tensor,
        return_attn: bool = False,
    ) -> dict[str, Tensor]:
        v = self.stock_enc(features)                            # (N, d)
        m_t = self.macro_enc(macro)                              # (d,)
        v = self.gate(v, m_t)
        for block in self.blocks:
            v = block(v)
        v = self.norm_out(v)
        y_hat = self.head(v).squeeze(-1)                         # (N,)
        msk = mask.float()
        y_hat = y_hat * msk
        return {
            "y_hat": y_hat,
            "regime_logits": torch.zeros(8, device=features.device),
            "vol_hat": torch.zeros_like(y_hat),
            "attn_weights": [] if return_attn else None,
        }


__all__ = ["MasterConfig", "Master"]
