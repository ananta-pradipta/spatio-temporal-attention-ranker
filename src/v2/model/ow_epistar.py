"""OW-epiSTAR: Open-World Survivorship-Aware Episodic Spatio-Temporal Attention Ranker.

v1 minimum implementation (per the spec's Section 21 staged plan).
Wraps epiSTAR-full with two additions:
    1. A ticker-level IPO analogue memory bank (Section 10).
    2. A dual-gated fusion that adds an `alpha_ipo` retrieval gate per
       (day, ticker) on top of the existing day-level `alpha_day` gate
       (Section 11).

Inherited from epiSTAR-full and unchanged: STAR backbone, dynamic
correlation graph (with optional shrinkage applied at the trainer),
day-level episodic memory bank, cross-attention fusion, day-level
retrieval gate, single rank head, cross-sectional MSE loss.

Deferred from the full spec for v1: security master, separated 8-mask
system, multi-source graph (epiDyReg-STAR experiment showed it hurt
fold-1), peer-based cold-start adapter, age embeddings inside STAR
backbone.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from src.v2.model.episode_memory import EpisodeMemoryBank, EpisodeMemoryConfig
from src.v2.model.ipo_memory import IPOAnalogueMemoryBank, IPOMemoryConfig
from src.v2.model.star_backbone import STARBackbone, STARBackboneConfig


@dataclass
class OWEpiSTARConfig:
    """Hyperparameters for OW-epiSTAR v1."""

    backbone: STARBackboneConfig = STARBackboneConfig()
    day_memory: EpisodeMemoryConfig = EpisodeMemoryConfig()
    ipo_memory: IPOMemoryConfig = IPOMemoryConfig()
    episode_value_dim: int = 32
    ipo_value_dim: int = 32
    cross_attn_heads: int = 4
    gate_hidden_dim: int = 64
    head_hidden_dim: int = 64
    head_dropout: float = 0.1
    disable_day_gate: bool = False
    disable_ipo_gate: bool = False
    disable_day_retrieval: bool = False
    disable_ipo_retrieval: bool = False


class OWEpiSTAR(nn.Module):
    """OW-epiSTAR with day-level + ticker-level dual retrieval."""

    def __init__(
        self,
        cfg: OWEpiSTARConfig,
        day_key_dim: int,
        ipo_key_dim: int,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone = STARBackbone(cfg.backbone)
        self.day_memory = EpisodeMemoryBank(
            cfg.day_memory, key_dim=day_key_dim, value_dim=cfg.episode_value_dim
        )
        self.ipo_memory = IPOAnalogueMemoryBank(
            cfg.ipo_memory, key_dim=ipo_key_dim, value_dim=cfg.ipo_value_dim
        )

        d = cfg.backbone.hidden_dim
        self.day_value_proj = nn.Linear(cfg.episode_value_dim, d)
        self.ipo_value_proj = nn.Linear(cfg.ipo_value_dim, d)
        self.day_cross_attn = nn.MultiheadAttention(
            embed_dim=d, num_heads=cfg.cross_attn_heads,
            dropout=cfg.backbone.dropout, batch_first=True,
        )
        self.ipo_cross_attn = nn.MultiheadAttention(
            embed_dim=d, num_heads=cfg.cross_attn_heads,
            dropout=cfg.backbone.dropout, batch_first=True,
        )
        self.day_fusion_mlp = nn.Sequential(
            nn.Linear(2 * d, d), nn.GELU(),
            nn.Dropout(cfg.backbone.dropout), nn.Linear(d, d),
        )
        self.ipo_fusion_mlp = nn.Sequential(
            nn.Linear(2 * d, d), nn.GELU(),
            nn.Dropout(cfg.backbone.dropout), nn.Linear(d, d),
        )
        # Day-level gate: 2 + 2 + day_key_dim (matches existing epiSTAR).
        day_gate_in_dim = 2 + 2 + day_key_dim
        self.day_gate_mlp = nn.Sequential(
            nn.Linear(day_gate_in_dim, cfg.gate_hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.gate_hidden_dim, 1),
        )
        # IPO-level gate (per-ticker): age + history-validity (2 dims) +
        # ipo_top1_sim + ipo_sim_entropy + 2 regime scalars + standardised
        # ipo query key.
        ipo_gate_in_dim = 2 + 2 + 2 + ipo_key_dim
        self.ipo_gate_mlp = nn.Sequential(
            nn.Linear(ipo_gate_in_dim, cfg.gate_hidden_dim),
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

    def forward_day(
        self,
        patches: Tensor,
        patch_mask: Tensor,
        active_mask: Tensor,
        day_query_key: Tensor,
        ipo_query_keys: Tensor,
        ipo_gate_features: Tensor,
        query_day_idx: int,
        allowed_day_indices: Tensor,
        gate_regime_scalars: Tensor,
    ) -> dict[str, Tensor]:
        """Forward pass for one trading day with dual retrieval.

        Args:
            patches, patch_mask, active_mask: same as epiSTAR.
            day_query_key: [day_key_dim] standardised regime+cs key for
                the day-level retrieval.
            ipo_query_keys: [num_active_tickers, ipo_key_dim] one IPO
                key per active ticker on this day.
            ipo_gate_features: [num_active_tickers, 2] per-ticker
                features (log1p_age, history_valid_ratio_60d) for the
                ticker-level IPO gate.
            query_day_idx: integer day index.
            allowed_day_indices: training-day allowlist for day memory.
            gate_regime_scalars: [2] standardised regime scalars (VIX z,
                avg pairwise corr).
        """
        cfg = self.cfg
        z_star = self.backbone.forward_day(patches, patch_mask, active_mask)
        device = z_star.device
        active_idx = active_mask.nonzero(as_tuple=False).squeeze(-1)
        z_active = z_star[active_idx]  # [A, D]

        # Day-level retrieval and fusion.
        if cfg.disable_day_retrieval:
            alpha_day = torch.zeros((), device=device)
            day_top1_sim = torch.zeros((), device=device)
            day_retrieved = torch.full((cfg.day_memory.top_m,), -1, dtype=torch.long, device=device)
            day_fusion = torch.zeros_like(z_active)
        else:
            day_ret = self.day_memory.retrieve(
                query_raw_key=day_query_key, query_day_idx=query_day_idx,
                allowed_day_indices=allowed_day_indices,
            )
            day_top1_sim = day_ret["top1_sim"].detach()
            day_retrieved = day_ret["day_indices"]
            day_proj = self.day_value_proj(day_ret["values"]).unsqueeze(0)
            h_day, _ = self.day_cross_attn(
                query=z_active.unsqueeze(0), key=day_proj, value=day_proj,
            )
            h_day = h_day.squeeze(0)
            day_q_std = self.day_memory.standardize_query(day_query_key)
            day_gate_in = torch.cat([
                day_ret["top1_sim"].unsqueeze(0),
                day_ret["sim_entropy"].unsqueeze(0),
                gate_regime_scalars,
                day_q_std,
            ])
            if cfg.disable_day_gate:
                alpha_day = torch.ones((), device=device)
            else:
                alpha_day = torch.sigmoid(self.day_gate_mlp(day_gate_in)).squeeze()
            day_fusion = alpha_day * self.day_fusion_mlp(torch.cat([z_active, h_day], dim=-1))

        # Per-ticker IPO retrieval and fusion (batched across active tickers).
        a = active_idx.shape[0]
        if cfg.disable_ipo_retrieval:
            alpha_ipo_list = torch.zeros(a, device=device)
            ipo_top1_sims = torch.zeros(a, device=device)
            ipo_retrieved_days = torch.full(
                (a, cfg.ipo_memory.top_m), -1, dtype=torch.long, device=device,
            )
            ipo_retrieved_tickers = ipo_retrieved_days.clone()
            ipo_fusion = torch.zeros_like(z_active)
        else:
            ipo_ret = self.ipo_memory.batch_retrieve(
                query_raw_keys=ipo_query_keys, query_day_idx=query_day_idx,
            )
            ipo_values = ipo_ret["values"]                         # [A, M, V]
            ipo_proj = self.ipo_value_proj(ipo_values)             # [A, M, D]
            z_q = z_active.unsqueeze(1)                            # [A, 1, D]
            h_ipo, _ = self.ipo_cross_attn(query=z_q, key=ipo_proj, value=ipo_proj)
            h_ipo = h_ipo.squeeze(1)                               # [A, D]

            ipo_q_std = (ipo_query_keys - self.ipo_memory.key_mean) / self.ipo_memory.key_std
            top1 = ipo_ret["top1_sim"].unsqueeze(-1)               # [A, 1]
            ent = ipo_ret["sim_entropy"].unsqueeze(-1)             # [A, 1]
            regime_broadcast = gate_regime_scalars.unsqueeze(0).expand(a, -1)
            ipo_gate_in = torch.cat(
                [ipo_gate_features, top1, ent, regime_broadcast, ipo_q_std], dim=-1
            )
            if cfg.disable_ipo_gate:
                alpha_ipo_per_t = torch.zeros(a, device=device)
            else:
                alpha_ipo_per_t = torch.sigmoid(
                    self.ipo_gate_mlp(ipo_gate_in)
                ).squeeze(-1)
            ipo_fusion = alpha_ipo_per_t.unsqueeze(-1) * self.ipo_fusion_mlp(
                torch.cat([z_active, h_ipo], dim=-1)
            )
            alpha_ipo_list = alpha_ipo_per_t.detach()
            ipo_top1_sims = ipo_ret["top1_sim"].detach()
            ipo_retrieved_days = ipo_ret["day_indices"]
            ipo_retrieved_tickers = ipo_ret["ticker_indices"]

        z_final_active = z_active + day_fusion + ipo_fusion
        z_final = torch.zeros_like(z_star)
        z_final[active_idx] = z_final_active
        y_hat = self.rank_head(z_final).squeeze(-1) * active_mask.float()

        return {
            "y_hat": y_hat,
            "alpha_day": alpha_day.detach(),
            "alpha_ipo": alpha_ipo_list,
            "day_top1_sim": day_top1_sim,
            "ipo_top1_sims": ipo_top1_sims,
            "day_retrieved": day_retrieved,
            "ipo_retrieved_days": ipo_retrieved_days,
            "ipo_retrieved_tickers": ipo_retrieved_tickers,
        }


__all__ = ["OWEpiSTAR", "OWEpiSTARConfig"]
