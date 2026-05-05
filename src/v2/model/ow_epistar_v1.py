"""OW-epiSTAR v1: focused survivorship-aware extension of epiSTAR-full.

Mirrors the v0 ``OWEpiSTAR`` (``src.v2.model.ow_epistar``) but with the
revised IPO retrieval-gate input that exactly matches Section E of the
v1 spec:

    [age_trading_days, log1p_age_trading_days, history_valid_ratio_20d,
     history_valid_ratio_60d, top1_ipo_similarity, ipo_similarity_entropy,
     has_fundamentals, st_labeled_ratio, realized_vol_20d, vix_z,
     avg_pairwise_corr_60d]

The day-level retrieval and gate are unchanged from epiSTAR-full. The
IPO retrieval is per-(day, ticker) and uses the 22-dim
``IPO_ANALOGUE_KEY_COLS`` defined in
``src.v2.model.ipo_analogue_memory``.

Final representation:

    h_final[t, i] = h_star[t, i]
                  + alpha_day[t]    * delta_day[t, i]
                  + alpha_ipo[t, i] * delta_ipo[t, i]

with both fusions implemented as alpha-gated MLPs over [h_star, h_*].
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from src.v2.model.episode_memory import EpisodeMemoryBank, EpisodeMemoryConfig
from src.v2.model.ipo_analogue_memory import IPOAnalogueMemoryBank, IPOMemoryConfig
from src.v2.model.star_backbone import STARBackbone, STARBackboneConfig


@dataclass
class OWEpiSTARV1Config:
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


# Per-active-ticker features the trainer must pass into ``forward_day``
# as ``ipo_gate_features``: [age_trading_days, log1p_age, hist20, hist60,
# has_fundamentals, st_labeled_ratio, realized_vol_20d].
IPO_GATE_TICKER_FEATURES = [
    "age_trading_days",
    "log1p_age_trading_days",
    "history_valid_ratio_20d",
    "history_valid_ratio_60d",
    "has_fundamentals",
    "st_labeled_ratio",
    "realized_vol_20d",
]
# IPO gate static input dim:
# 7 ticker features + 2 retrieval scalars (top1, entropy) + 2 macro
# scalars (VIX z, avg pairwise corr) = 11 dims (Section E of the spec).
IPO_GATE_INPUT_DIM = len(IPO_GATE_TICKER_FEATURES) + 2 + 2


class OWEpiSTARV1(nn.Module):
    """OW-epiSTAR v1 with day-level + ticker-level dual retrieval."""

    def __init__(
        self,
        cfg: OWEpiSTARV1Config,
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
        # Day-level gate: 2 (top1+entropy) + 2 (regime) + day_key_dim.
        day_gate_in_dim = 2 + 2 + day_key_dim
        self.day_gate_mlp = nn.Sequential(
            nn.Linear(day_gate_in_dim, cfg.gate_hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.gate_hidden_dim, 1),
        )
        self.ipo_gate_mlp = nn.Sequential(
            nn.Linear(IPO_GATE_INPUT_DIM, cfg.gate_hidden_dim),
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
        pre_rank_hook=None,
    ) -> dict[str, Tensor]:
        """Forward pass for one trading day.

        Args:
            patches, patch_mask, active_mask: same as epiSTAR.
            day_query_key: [day_key_dim] standardised regime+cs key for
                the day-level retrieval.
            ipo_query_keys: [num_active, ipo_key_dim] IPO key per active
                ticker.
            ipo_gate_features: [num_active, 7] per-ticker gate features
                in the order of ``IPO_GATE_TICKER_FEATURES``.
            query_day_idx: integer day index.
            allowed_day_indices: training-day allowlist.
            gate_regime_scalars: [2] standardised (VIX z, avg pairwise
                corr) for both gates.
            pre_rank_hook: optional callable taking ``(z_active)`` and
                returning ``(z_active_modified, hook_diag_dict)``. Used
                by CSID to splice cross-sectional residualisation
                between the STAR backbone and the cross-attention
                queries / rank head, per the CSID v1 spec.
        """
        cfg = self.cfg
        z_star = self.backbone.forward_day(patches, patch_mask, active_mask)
        device = z_star.device
        active_idx = active_mask.nonzero(as_tuple=False).squeeze(-1)
        z_active = z_star[active_idx]
        a = active_idx.shape[0]
        hook_diag: dict[str, Tensor] = {}
        if pre_rank_hook is not None and a >= 2:
            z_active, hook_diag = pre_rank_hook(z_active)

        # Day-level retrieval.
        if cfg.disable_day_retrieval:
            alpha_day = torch.zeros((), device=device)
            day_top1_sim = torch.zeros((), device=device)
            day_retrieved = torch.full(
                (cfg.day_memory.top_m,), -1, dtype=torch.long, device=device,
            )
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
            day_fusion = alpha_day * self.day_fusion_mlp(
                torch.cat([z_active, h_day], dim=-1)
            )

        # IPO retrieval (batched).
        if cfg.disable_ipo_retrieval:
            alpha_ipo_list = torch.zeros(a, device=device)
            ipo_top1_sims = torch.zeros(a, device=device)
            ipo_sim_entropies = torch.zeros(a, device=device)
            ipo_retrieved_days = torch.full(
                (a, cfg.ipo_memory.top_m), -1, dtype=torch.long, device=device,
            )
            ipo_retrieved_tickers = ipo_retrieved_days.clone()
            ipo_fusion = torch.zeros_like(z_active)
        else:
            ipo_ret = self.ipo_memory.batch_retrieve(
                query_raw_keys=ipo_query_keys, query_day_idx=query_day_idx,
            )
            ipo_values = ipo_ret["values"]
            ipo_proj = self.ipo_value_proj(ipo_values)
            z_q = z_active.unsqueeze(1)
            h_ipo, _ = self.ipo_cross_attn(query=z_q, key=ipo_proj, value=ipo_proj)
            h_ipo = h_ipo.squeeze(1)

            top1 = ipo_ret["top1_sim"].unsqueeze(-1)
            ent = ipo_ret["sim_entropy"].unsqueeze(-1)
            regime_broadcast = gate_regime_scalars.unsqueeze(0).expand(a, -1)
            ipo_gate_in = torch.cat(
                [ipo_gate_features, top1, ent, regime_broadcast], dim=-1
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
            ipo_sim_entropies = ipo_ret["sim_entropy"].detach()
            ipo_retrieved_days = ipo_ret["day_indices"]
            ipo_retrieved_tickers = ipo_ret["ticker_indices"]

        z_final_active = z_active + day_fusion + ipo_fusion
        z_final = torch.zeros_like(z_star)
        z_final[active_idx] = z_final_active
        y_hat = self.rank_head(z_final).squeeze(-1) * active_mask.float()

        out = {
            "y_hat": y_hat,
            "z_final": z_final,
            "alpha_day": alpha_day.detach(),
            "alpha_ipo": alpha_ipo_list,
            "day_top1_sim": day_top1_sim,
            "ipo_top1_sims": ipo_top1_sims,
            "ipo_sim_entropies": ipo_sim_entropies,
            "day_retrieved": day_retrieved,
            "ipo_retrieved_days": ipo_retrieved_days,
            "ipo_retrieved_tickers": ipo_retrieved_tickers,
        }
        # Surface CSID (or any pre-rank hook) diagnostics.
        for k, v in hook_diag.items():
            out[f"hook_{k}"] = v
        return out


__all__ = [
    "OWEpiSTARV1",
    "OWEpiSTARV1Config",
    "IPO_GATE_TICKER_FEATURES",
    "IPO_GATE_INPUT_DIM",
]
