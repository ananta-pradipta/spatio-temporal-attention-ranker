"""CohortEmbedding: 4-axis cohort embedding (size, liquidity, sector, age).

Per spec section 6.7.

Concatenates four learned embeddings (size_decile, liquidity_decile, sector,
age_bucket) into a 64-d cohort vector and projects to d_model.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class CohortEmbeddingConfig:
    d_model: int = 128
    n_size_deciles: int = 10
    n_liquidity_deciles: int = 10
    n_sectors: int = 11
    n_age_buckets: int = 4
    embedding_dim: int = 16
    pad_idx: int = 0  # +1 offset on each axis to reserve 0 as "missing"


class CohortEmbedding(nn.Module):
    """Embed 4-axis cohort labels."""

    def __init__(self, cfg: CohortEmbeddingConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or CohortEmbeddingConfig()
        self.cfg = cfg
        # Each embedding has +1 vocab to reserve 0 for missing/padding.
        self.size_emb = nn.Embedding(cfg.n_size_deciles + 1, cfg.embedding_dim,
                                       padding_idx=cfg.pad_idx)
        self.liq_emb = nn.Embedding(cfg.n_liquidity_deciles + 1, cfg.embedding_dim,
                                      padding_idx=cfg.pad_idx)
        self.sector_emb = nn.Embedding(cfg.n_sectors + 1, cfg.embedding_dim,
                                          padding_idx=cfg.pad_idx)
        self.age_emb = nn.Embedding(cfg.n_age_buckets + 1, cfg.embedding_dim,
                                       padding_idx=cfg.pad_idx)
        self.proj = nn.Linear(4 * cfg.embedding_dim, cfg.d_model)

    def forward(
        self, size_decile: Tensor, liquidity_decile: Tensor,
        sector_id: Tensor, age_bucket: Tensor,
    ) -> Tensor:
        """Each input is [B, N] long; missing cells should be 0.

        Returns:
            [B, N, d_model] cohort embeddings (zeros for missing-cohort cells).
        """
        # Add 1 so that 0 stays as padding and label k gets index k+1.
        s = self.size_emb(size_decile.clamp(min=0).long() + 1)
        l = self.liq_emb(liquidity_decile.clamp(min=0).long() + 1)
        sec = self.sector_emb(sector_id.clamp(min=0).long() + 1)
        a = self.age_emb(age_bucket.clamp(min=0).long() + 1)
        cohort_concat = torch.cat([s, l, sec, a], dim=-1)
        return self.proj(cohort_concat)


__all__ = ["CohortEmbedding", "CohortEmbeddingConfig"]
