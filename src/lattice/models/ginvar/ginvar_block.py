"""GInVARBlock: pre-norm Transformer block with graph-guided attention."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn

from src.lattice.models.ginvar.graph_guided_attention import (
    GraphGuidedInvertedAttention, GraphGuidedAttentionConfig,
)


@dataclass
class GInVARBlockConfig:
    d_model: int = 128
    n_heads: int = 4
    ffn_hidden: int = 512                  # 4 * d_model
    dropout: float = 0.10
    graph_mode: str = "graph_bias_and_mask"
    top_k: int = 16
    beta_init: float = 1.0
    learn_beta: bool = True


class GInVARBlock(nn.Module):
    """Pre-norm graph-guided inverted attention block."""

    def __init__(self, cfg: GInVARBlockConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or GInVARBlockConfig()
        self.cfg = cfg
        attn_cfg = GraphGuidedAttentionConfig(
            d_model=cfg.d_model, n_heads=cfg.n_heads, dropout=cfg.dropout,
            graph_mode=cfg.graph_mode, top_k=cfg.top_k,
            beta_init=cfg.beta_init, learn_beta=cfg.learn_beta,
        )
        self.norm_attn = nn.LayerNorm(cfg.d_model)
        self.attn = GraphGuidedInvertedAttention(attn_cfg)
        self.norm_ffn = nn.LayerNorm(cfg.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.ffn_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.ffn_hidden, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(
        self, h: Tensor, A_graph: Tensor | None, active_mask: Tensor,
        return_attn: bool = False,
    ) -> tuple[Tensor, Optional[Tensor]]:
        h_norm = self.norm_attn(h)
        attn_out, attn_w = self.attn(h_norm, A_graph, active_mask,
                                       return_attn=return_attn)
        h = h + attn_out
        h = h + self.ffn(self.norm_ffn(h))
        return h, attn_w


__all__ = ["GInVARBlock", "GInVARBlockConfig"]
