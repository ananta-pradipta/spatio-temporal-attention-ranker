"""epiDyReg-STAR: full v1 implementation per the spec.

Adds, on top of epiSTAR-full's "v0 minimal" unification:

    - Multi-source graph: static mechanistic + rolling return-correlation
      + rolling residual-correlation. Source mixing is regime-conditioned
      via a softmax over MLP(z_regime).
    - Age-aware graph rules: tickers with fewer than 60 trading days of
      history are excluded from rolling-correlation and residual-
      correlation candidate pools.
    - Richer episode key: the existing 14-dim risk + cross-sectional
      diagnostics are augmented with a 6-dim per-day graph summary
      (mean-absolute-corr, PC1-share proxy, graph density, neighbor
      turnover, normalised active count, score std).
    - Same STAR backbone, same episodic memory bank, same cross-
      attention fusion + retrieval gate as epiSTAR / epiSTAR-full. The
      contribution is in the graph and key richness, not new heads.

Per Section 21 of the spec ("Recommended First Implementation Variant"):
no FiLM, no auxiliary volatility head, no expert heads, no uncertainty
head. Single rank head. Cross-sectional Mean Squared Error on z-scored
5-day forward log returns.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from src.v2.model.episode_memory import EpisodeMemoryBank, EpisodeMemoryConfig
from src.v2.model.star_backbone import STARBackbone, STARBackboneConfig


@dataclass
class GraphSourceMixerConfig:
    """Hyperparameters for the regime-conditioned graph source mixer."""

    regime_dim: int = 14            # incoming raw regime-feature dim
    hidden_dim: int = 64
    num_sources: int = 3            # static, return-corr, residual-corr


class GraphSourceMixer(nn.Module):
    """Produces softmax weights over K=num_sources graph sources.

    Input:  z_regime in (regime_dim,)
    Output: weights in (num_sources,) summing to num_sources (so the
            average weight equals 1; the gate cannot collapse to zero
            on every source).
    """

    def __init__(self, cfg: GraphSourceMixerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.mlp = nn.Sequential(
            nn.LayerNorm(cfg.regime_dim),
            nn.Linear(cfg.regime_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.num_sources),
        )

    def forward(self, z_regime: Tensor) -> Tensor:
        """Args: z_regime (regime_dim,). Returns weights (num_sources,)."""
        logits = self.mlp(z_regime)
        return torch.softmax(logits, dim=-1) * self.cfg.num_sources


@dataclass
class EpiDyRegSTARConfig:
    """Hyperparameters for epiDyReg-STAR."""

    backbone: STARBackboneConfig = STARBackboneConfig()
    memory: EpisodeMemoryConfig = EpisodeMemoryConfig()
    mixer: GraphSourceMixerConfig = GraphSourceMixerConfig()
    episode_value_dim: int = 32
    cross_attn_heads: int = 4
    gate_hidden_dim: int = 64
    head_hidden_dim: int = 64
    head_dropout: float = 0.1
    disable_gate: bool = False
    disable_retrieval: bool = False


class EpiDyRegSTAR(nn.Module):
    """STAR backbone + multi-source dynamic graph + episodic retrieval.

    Mirrors EpiSTAR's interface but exposes a `gate_weights_for_day(
    raw_regime_key)` method that returns the K-dim source weights used
    by the trainer to combine pre-computed per-source candidate score
    matrices before top-K neighbor selection.
    """

    def __init__(self, cfg: EpiDyRegSTARConfig, episode_key_dim: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone = STARBackbone(cfg.backbone)
        # The graph-source mixer reads the raw regime key directly so the
        # trainer can compute candidate scores on CPU without a separate
        # pre-encoder forward pass.
        cfg.mixer.regime_dim = episode_key_dim
        self.graph_mixer = GraphSourceMixer(cfg.mixer)
        self.memory = EpisodeMemoryBank(
            cfg.memory, key_dim=episode_key_dim, value_dim=cfg.episode_value_dim
        )
        d = cfg.backbone.hidden_dim
        self.episode_value_proj = nn.Linear(cfg.episode_value_dim, d)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d, num_heads=cfg.cross_attn_heads,
            dropout=cfg.backbone.dropout, batch_first=True,
        )
        self.fusion_mlp = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.GELU(),
            nn.Dropout(cfg.backbone.dropout),
            nn.Linear(d, d),
        )
        gate_in_dim = 2 + 2 + episode_key_dim
        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_in_dim, cfg.gate_hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.gate_hidden_dim, 1),
        )
        self.rank_head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, cfg.head_hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.head_dropout),
            nn.Linear(cfg.head_hidden_dim, 1),
        )

    def gate_weights_for_day(self, raw_regime_key: Tensor) -> Tensor:
        """Per-day softmax weights over graph sources (in K-dim)."""
        return self.graph_mixer(raw_regime_key)

    def forward_day(
        self,
        patches: Tensor,
        patch_mask: Tensor,
        active_mask: Tensor,
        query_raw_key: Tensor,
        query_day_idx: int,
        allowed_day_indices: Tensor,
        gate_regime_scalars: Tensor,
    ) -> dict[str, Tensor]:
        """Forward pass for one trading day.

        Args:
            patches: [A, N+1, W, F] STAR input patches (built upstream
                from the multi-source-mixed dynamic neighbor list).
            patch_mask: [A, N+1, W] bool, True where observed.
            active_mask: [num_nodes] bool, True for active tickers.
            query_raw_key: [key_dim] raw regime + cs + graph-summary key.
            query_day_idx: integer day index of the query.
            allowed_day_indices: long tensor of allowed memory days.
            gate_regime_scalars: [2] standardized regime scalars used by
                the confidence gate (e.g., VIX z-score, avg pairwise corr).

        Returns:
            Dict with y_hat, alpha (gate weight), top1_sim, retrieved
            day indices, and z_star (backbone hidden representation).
        """
        cfg = self.cfg
        z_star = self.backbone.forward_day(patches, patch_mask, active_mask)
        device = z_star.device

        if cfg.disable_retrieval:
            y_hat = self.rank_head(z_star).squeeze(-1) * active_mask.float()
            return {
                "y_hat": y_hat,
                "alpha": torch.zeros((), device=device),
                "top1_sim": torch.zeros((), device=device),
                "retrieved_day_indices": torch.full(
                    (cfg.memory.top_m,), -1, dtype=torch.long, device=device
                ),
                "z_star": z_star,
            }

        retrieval = self.memory.retrieve(
            query_raw_key=query_raw_key,
            query_day_idx=query_day_idx,
            allowed_day_indices=allowed_day_indices,
        )
        ep_proj = self.episode_value_proj(retrieval["values"]).unsqueeze(0)

        active_idx = active_mask.nonzero(as_tuple=False).squeeze(-1)
        z_active = z_star[active_idx].unsqueeze(0)
        h_epi, _ = self.cross_attn(query=z_active, key=ep_proj, value=ep_proj)
        h_epi = h_epi.squeeze(0)
        z_active_sq = z_active.squeeze(0)

        q_std = self.memory.standardize_query(query_raw_key)
        gate_in = torch.cat([
            retrieval["top1_sim"].unsqueeze(0),
            retrieval["sim_entropy"].unsqueeze(0),
            gate_regime_scalars,
            q_std,
        ])
        if cfg.disable_gate:
            alpha = torch.ones((), device=device)
        else:
            alpha = torch.sigmoid(self.gate_mlp(gate_in)).squeeze()

        fused = self.fusion_mlp(torch.cat([z_active_sq, h_epi], dim=-1))
        z_final_active = z_active_sq + alpha * fused

        z_final = torch.zeros_like(z_star)
        z_final[active_idx] = z_final_active

        y_hat = self.rank_head(z_final).squeeze(-1) * active_mask.float()
        return {
            "y_hat": y_hat,
            "alpha": alpha.detach(),
            "top1_sim": retrieval["top1_sim"].detach(),
            "retrieved_day_indices": retrieval["day_indices"],
            "z_star": z_star,
        }


__all__ = [
    "EpiDyRegSTAR",
    "EpiDyRegSTARConfig",
    "GraphSourceMixer",
    "GraphSourceMixerConfig",
]
