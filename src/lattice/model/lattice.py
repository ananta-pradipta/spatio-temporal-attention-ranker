"""LATTICE top-level module.

Per spec section 6.9. Composes the 7 sub-modules:

    PerStockEncoder -> CrossSectionalAggregator -> DualGraphBranch
    + CohortEmbedding -> DualRetrieval (regime + novelty)
    + MacroMoERouter -> AdditiveResidualHead

Forward signature:

    y_hat, balance_loss = lattice(
        panel_features,        # [B, N, T_lookback, F_panel]
        macro_state,           # [B, F_macro]
        cohort_labels,         # dict of [B, N] long tensors:
                                #   size_decile, liquidity_decile, sector_id, age_bucket
        regime_query_keys,     # [B, K_regime] day-level regime fingerprint
        novelty_query_keys,    # [B, N, K_novelty] per-(day, ticker) novelty signature
        novelty_sector_ids,    # [B, N] long
        active_mask,           # [B, N] bool
        day_index,             # [B] long, integer day indices for retrieval leakage gate
        corr_neighbor_idx,     # [B, N, K_top] long
        corr_neighbor_mask,    # [B, N, K_top] bool
    )
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from src.lattice.model.encoder import PerStockEncoder, PerStockEncoderConfig
from src.lattice.model.aggregator import (
    CrossSectionalAggregator, CrossSectionalAggregatorConfig,
)
from src.lattice.model.graph import DualGraphBranch, DualGraphBranchConfig
from src.lattice.model.macro_router import MacroMoERouter, MacroMoERouterConfig
from src.lattice.model.retrieval import DualRetrieval, DualRetrievalConfig
from src.lattice.model.cohort import CohortEmbedding, CohortEmbeddingConfig
from src.lattice.model.head import AdditiveResidualHead, AdditiveResidualHeadConfig


@dataclass
class LatticeConfig:
    encoder: PerStockEncoderConfig = None
    aggregator: CrossSectionalAggregatorConfig = None
    graph: DualGraphBranchConfig = None
    macro_router: MacroMoERouterConfig = None
    retrieval: DualRetrievalConfig = None
    cohort: CohortEmbeddingConfig = None
    head: AdditiveResidualHeadConfig = None
    d_model: int = 128

    def __post_init__(self):
        if self.encoder is None: self.encoder = PerStockEncoderConfig(d_model=self.d_model)
        if self.aggregator is None: self.aggregator = CrossSectionalAggregatorConfig(d_model=self.d_model)
        if self.graph is None: self.graph = DualGraphBranchConfig(d_model=self.d_model)
        if self.macro_router is None: self.macro_router = MacroMoERouterConfig(d_model=self.d_model)
        if self.retrieval is None: self.retrieval = DualRetrievalConfig(d_model=self.d_model)
        if self.cohort is None: self.cohort = CohortEmbeddingConfig(d_model=self.d_model)
        if self.head is None: self.head = AdditiveResidualHeadConfig(d_model=self.d_model)


class LATTICE(nn.Module):
    """LATTICE forward pass."""

    def __init__(self, cfg: LatticeConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or LatticeConfig()
        self.cfg = cfg
        self.encoder = PerStockEncoder(cfg.encoder)
        self.aggregator = CrossSectionalAggregator(cfg.aggregator)
        self.graph = DualGraphBranch(cfg.graph)
        self.cohort_emb = CohortEmbedding(cfg.cohort)
        self.retrieval = DualRetrieval(cfg.retrieval)
        self.macro_router = MacroMoERouter(cfg.macro_router)
        self.head = AdditiveResidualHead(cfg.head)

    def forward(
        self,
        panel_features: Tensor,
        macro_state: Tensor,
        cohort_size_decile: Tensor,
        cohort_liquidity_decile: Tensor,
        cohort_sector_id: Tensor,
        cohort_age_bucket: Tensor,
        regime_query_keys: Tensor,
        novelty_query_keys: Tensor,
        novelty_sector_ids: Tensor,
        active_mask: Tensor,
        day_index: Tensor,
        corr_neighbor_idx: Tensor,
        corr_neighbor_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """LATTICE forward pass.

        Returns:
            (y_hat, balance_loss).
            y_hat: [B, N] per-(day, ticker) score.
            balance_loss: scalar MoE load-balance auxiliary loss.
        """
        z_per_stock = self.encoder(panel_features)
        z_aggregated = self.aggregator(z_per_stock, active_mask)
        z_graph = self.graph(z_aggregated, active_mask, macro_state,
                              corr_neighbor_idx, corr_neighbor_mask)
        cohort_emb = self.cohort_emb(
            cohort_size_decile, cohort_liquidity_decile,
            cohort_sector_id, cohort_age_bucket,
        )
        z_combined = z_aggregated + z_graph + cohort_emb

        delta_regime, alpha_regime = self.retrieval.regime(
            z_combined, regime_query_keys, day_index,
        )
        delta_novelty, alpha_novelty = self.retrieval.novelty(
            z_combined, novelty_query_keys, novelty_sector_ids, day_index,
            active_mask,
        )
        z_combined = z_combined + alpha_regime * delta_regime + alpha_novelty * delta_novelty

        expert_output, balance_loss = self.macro_router(macro_state, z_combined)
        y_hat = self.head(z_combined, expert_output)

        # Zero out inactive cells in the score
        y_hat = y_hat * active_mask.float()
        return y_hat, balance_loss


__all__ = ["LATTICE", "LatticeConfig"]
