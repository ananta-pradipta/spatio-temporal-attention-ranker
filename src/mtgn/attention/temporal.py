"""Episodic temporal attention for MTGN: second attention layer over retrieved memory snapshots.

Implements the temporal (episodic) attention described in memo §6.3:

    q = W_q @ h_spatial                          # query projection
    keys   = [W_k @ entry.memory + phi(dt)]      # time-encoded keys
    values = [W_v @ entry.memory]
    h_temporal = MultiHeadAttention(q, keys, values)
    z = LayerNorm(h_spatial + h_temporal)        # residual fusion

Retrieval is done outside this module (via `EpisodicStore.retrieve`).
This module consumes the retrieved memory stacks.

Phase 1 QE scope: cross-entity retrieval on/off is an ablation dimension.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class TemporalAttentionConfig:
    memory_dim: int = 128
    hidden_dim: int = 128
    num_heads: int = 4
    time_dim: int = 32


class TimeEncoder(nn.Module):
    """TGAT-style functional time encoder phi(dt) = cos(omega*dt + phi)."""

    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dim) * 0.1)
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, dt: Tensor) -> Tensor:
        return torch.cos(dt.unsqueeze(-1) * self.weight + self.bias)


class EpisodicTemporalAttention(nn.Module):
    """Multi-head attention from h_spatial over retrieved memory snapshots.

    Stabilization (added 2026-04-13 after instrumented diagnostic showed
    attention entropy pinned at log(K), i.e. softmax collapse-to-uniform):
      * LayerNorm on query and key vectors before the dot product, so
        initialization-era q.k dot products have unit-variance regardless
        of upstream GAT / memory magnitudes.
      * learnable log-temperature scalar so the effective softmax
        temperature can train out of the flat regime.
    """

    def __init__(self, cfg: TemporalAttentionConfig):
        super().__init__()
        self.cfg = cfg
        self.W_q = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.W_k = nn.Linear(cfg.memory_dim + cfg.time_dim, cfg.hidden_dim)
        self.W_v = nn.Linear(cfg.memory_dim, cfg.hidden_dim)
        self.q_norm = nn.LayerNorm(cfg.hidden_dim)
        self.k_norm = nn.LayerNorm(cfg.hidden_dim)
        self.log_temperature = nn.Parameter(torch.zeros(1))   # effective temp = exp(log_t), init 1.0
        self.time_encoder = TimeEncoder(cfg.time_dim)
        assert cfg.hidden_dim % cfg.num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.head_dim = cfg.hidden_dim // cfg.num_heads
        self.out = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.norm = nn.LayerNorm(cfg.hidden_dim)

    def forward(
        self,
        h_spatial: Tensor,       # [N, hidden_dim]
        entries_memory: Tensor,  # [N, K, memory_dim]     retrieved raw s_i(t)
        entries_dt: Tensor,      # [N, K]                 time since entry, in days
        mask: Tensor,            # [N, K]  bool; True = valid candidate
    ) -> Tensor:
        """Return the residual-fused final embedding z_i(t)."""
        N, K, _ = entries_memory.shape
        H = self.cfg.num_heads

        # Time encoding
        t_enc = self.time_encoder(entries_dt.float())              # [N, K, time_dim]
        k_in = torch.cat([entries_memory, t_enc], dim=-1)           # [N, K, mem+time]

        q_full = self.q_norm(self.W_q(h_spatial))                   # [N, hidden]
        k_full = self.k_norm(self.W_k(k_in))                        # [N, K, hidden]
        q = q_full.view(N, 1, H, self.head_dim)                     # [N, 1, H, d]
        k = k_full.view(N, K, H, self.head_dim)                     # [N, K, H, d]
        v = self.W_v(entries_memory).view(N, K, H, self.head_dim)   # [N, K, H, d]

        # Scaled dot-product attention with learnable temperature.
        temperature = torch.exp(self.log_temperature).clamp(min=0.1, max=10.0)
        scores = (q * k).sum(-1) / math.sqrt(self.head_dim) * temperature   # [N, K, H]
        # Mask invalid candidates before softmax
        mask_h = mask.unsqueeze(-1).expand(-1, -1, H)               # [N, K, H]
        scores = scores.masked_fill(~mask_h, float("-inf"))
        # Nodes with zero valid candidates: produce zeros (no attention)
        no_valid = (~mask).all(dim=-1)                              # [N]
        attn = torch.softmax(scores, dim=1)                          # [N, K, H]
        attn = torch.where(no_valid.unsqueeze(-1).unsqueeze(-1), torch.zeros_like(attn), attn)

        h_temporal = (attn.unsqueeze(-1) * v).sum(dim=1)             # [N, H, d]
        h_temporal = h_temporal.reshape(N, self.cfg.hidden_dim)
        h_temporal = self.out(h_temporal)

        return self.norm(h_spatial + h_temporal)


__all__ = ["EpisodicTemporalAttention", "TemporalAttentionConfig", "TimeEncoder"]
