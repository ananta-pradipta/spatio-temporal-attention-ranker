"""GraphGuidedInvertedAttention: stock-token attention biased by graph priors.

Modes (per spec section 4):
  - "dense"          : no graph bias and no graph mask (baseline iTransformer-style)
  - "graph_bias"     : add ``beta_head * log(A_graph + eps)`` to attention logits
  - "graph_mask"     : restrict attention to top-K graph neighbors
  - "graph_bias_and_mask" : both

The attention is computed over the cross-stock dimension N (each stock
attends to other stocks at the same time step). Inactive tickers always
receive ``-inf`` attention logits via ``active_mask``.

If a query ticker has zero valid graph neighbors after top-K masking,
fall back to attending over its sector group (computed from the graph
itself if available, else over all active tickers with low graph bias).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn


@dataclass
class GraphGuidedAttentionConfig:
    d_model: int = 128
    n_heads: int = 4
    dropout: float = 0.10
    graph_mode: str = "graph_bias_and_mask"   # dense / graph_bias / graph_mask / both
    top_k: int = 16
    beta_init: float = 1.0
    learn_beta: bool = True
    eps: float = 1.0e-6


class GraphGuidedInvertedAttention(nn.Module):
    """Multi-head attention over stock tokens with graph-informed logits."""

    def __init__(self, cfg: GraphGuidedAttentionConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or GraphGuidedAttentionConfig()
        self.cfg = cfg
        assert cfg.d_model % cfg.n_heads == 0, "d_model must be divisible by n_heads"
        self.d_head = cfg.d_model // cfg.n_heads

        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.k_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

        beta_init_value = float(cfg.beta_init) * torch.ones(cfg.n_heads)
        if cfg.learn_beta:
            self.beta = nn.Parameter(beta_init_value)
        else:
            self.register_buffer("beta", beta_init_value, persistent=True)

    def forward(
        self, h: Tensor, A_graph: Tensor | None, active_mask: Tensor,
        return_attn: bool = False,
    ) -> tuple[Tensor, Optional[Tensor]]:
        """Args:
            h           : (B, N, d_model)
            A_graph     : (B, N, N) or None (for "dense" mode)
            active_mask : (B, N) bool

        Returns ``(h_out, attn_weights or None)``.
        """
        B, N, D = h.shape
        H = self.cfg.n_heads
        Dh = self.d_head

        q = self.q_proj(h).view(B, N, H, Dh).transpose(1, 2)
        k = self.k_proj(h).view(B, N, H, Dh).transpose(1, 2)
        v = self.v_proj(h).view(B, N, H, Dh).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(Dh)

        # Active-mask: inactive query rows are still attended OUT-of by active
        # keys (we do not mask query-rows here; the rank head zeroes inactive
        # output rows). Inactive KEY columns get -inf so no token attends to them.
        col_mask = ~active_mask
        scores = scores.masked_fill(col_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        # Graph bias and / or graph mask.
        graph_mode = self.cfg.graph_mode
        if graph_mode != "dense" and A_graph is not None:
            A = A_graph.clamp(min=0.0)
            log_A = torch.log(A + self.cfg.eps).unsqueeze(1)
            beta = self.beta.view(1, H, 1, 1)

            if graph_mode in ("graph_bias", "graph_bias_and_mask"):
                scores = scores + beta * log_A

            if graph_mode in ("graph_mask", "graph_bias_and_mask"):
                K = min(self.cfg.top_k, N)
                topk = torch.topk(A, k=K, dim=-1)
                topk_mask = torch.zeros_like(A, dtype=torch.bool)
                topk_mask.scatter_(-1, topk.indices, True)
                # Always allow the diagonal so tokens can attend to themselves.
                eye = torch.eye(N, dtype=torch.bool, device=A.device).unsqueeze(0)
                topk_mask = topk_mask | eye
                # Restrict to active columns.
                topk_mask = topk_mask & active_mask.unsqueeze(1)
                # Fallback: if a query row has no valid neighbors, allow all
                # active columns (with low graph bias because A is small there).
                empty_rows = (~topk_mask).all(dim=-1, keepdim=True)
                fallback = active_mask.unsqueeze(1).expand_as(topk_mask)
                topk_mask = torch.where(empty_rows, fallback, topk_mask)
                # Apply mask: -inf where not allowed.
                scores = scores.masked_fill(
                    ~topk_mask.unsqueeze(1), float("-inf"),
                )

        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.attn_dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        out = self.resid_dropout(self.out_proj(out))
        return out, attn if return_attn else None


__all__ = ["GraphGuidedInvertedAttention", "GraphGuidedAttentionConfig"]
