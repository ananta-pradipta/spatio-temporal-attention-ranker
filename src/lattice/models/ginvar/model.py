"""G-InVAR: Graph-Guided Inverted VARiate Attention Ranker.

forward(x_window, A_graph, active_mask, macro_state) -> dict:
  - scores : (B, N) cross-sectional ranking score (zero on inactive)
  - extras : dict (currently empty; reserved for diagnostics)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import Tensor, nn

from src.lattice.models.ginvar.temporal_tokenizer import (
    TemporalStockTokenizer, TemporalTokenizerConfig,
)
from src.lattice.models.ginvar.ginvar_block import GInVARBlock, GInVARBlockConfig


@dataclass
class GInVARConfig:
    n_features: int = 26
    lookback: int = 20
    macro_dim: int = 24
    d_model: int = 128
    n_heads: int = 4
    ffn_hidden: int = 512
    n_layers: int = 2
    dropout: float = 0.10
    graph_mode: str = "graph_bias_and_mask"
    top_k: int = 16
    beta_init: float = 1.0
    learn_beta: bool = True
    tokenizer_hidden: int = 256
    head_hidden: int = 64


class GInVAR(nn.Module):
    """G-InVAR forward pass.

    forward signature:
        x_window     : (B, Tw, N, F)
        A_graph      : (B, N, N) or None (for graph_mode == "dense")
        active_mask  : (B, N) bool
        macro_state  : optional (B, M); currently unused at the model
                       level (graph blending consumes macro_state in the
                       trainer, the model itself is conditioned on the
                       blended A_graph alone).

    returns:
        scores  : (B, N)
        extras  : dict reserved for diagnostics (graph stats, attn maps)
    """

    def __init__(self, cfg: GInVARConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or GInVARConfig()
        self.cfg = cfg
        self.tokenizer = TemporalStockTokenizer(TemporalTokenizerConfig(
            n_features=cfg.n_features, lookback=cfg.lookback,
            d_model=cfg.d_model, hidden=cfg.tokenizer_hidden,
            dropout=cfg.dropout,
        ))
        block_cfg = GInVARBlockConfig(
            d_model=cfg.d_model, n_heads=cfg.n_heads,
            ffn_hidden=cfg.ffn_hidden, dropout=cfg.dropout,
            graph_mode=cfg.graph_mode, top_k=cfg.top_k,
            beta_init=cfg.beta_init, learn_beta=cfg.learn_beta,
        )
        self.blocks = nn.ModuleList(
            [GInVARBlock(block_cfg) for _ in range(cfg.n_layers)]
        )
        self.head_norm = nn.LayerNorm(cfg.d_model)
        self.rank_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.head_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.head_hidden, 1),
        )

    def forward(
        self, x_window: Tensor, A_graph: Tensor | None, active_mask: Tensor,
        macro_state: Tensor | None = None, return_attn: bool = False,
    ) -> tuple[Tensor, dict]:
        h = self.tokenizer(x_window)
        attn_per_block: list = []
        for block in self.blocks:
            h, attn_w = block(h, A_graph, active_mask, return_attn=return_attn)
            if return_attn and attn_w is not None:
                attn_per_block.append(attn_w.detach())
        h = self.head_norm(h)
        scores = self.rank_head(h).squeeze(-1)
        scores = scores.masked_fill(~active_mask, 0.0)
        extras: dict = {}
        if return_attn:
            extras["attn_per_block"] = attn_per_block
        return scores, extras


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


__all__ = ["GInVAR", "GInVARConfig", "count_parameters"]
