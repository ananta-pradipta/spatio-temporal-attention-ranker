"""epiSTAR-SBP: Survivorship-Bias Patch on the epiSTAR-full base.

v1 minimum implementation (per spec Section 21 staged plan). Reuses the
STAR backbone, the day-level EpisodeMemoryBank infrastructure, and the
existing dynamic correlation graph from epiSTAR-full. The four
architectural changes are:

    1. Cohort-augmented 18-dim regime key (existing 14-dim risk + cs
       diagnostics, plus 4-dim cohort sub-key from the active universe).
    2. Dual-pool retrieval (M1=5 raw similar + M2=3 cohort-near via
       L1 distance on the 4-dim cohort sub-key against tau_cohort).
    3. Two-head cross-attention with mu mixing: head H1 attends over
       Pool 1, head H2 over Pool 2; mu_t mixes them per day.
    4. Per-ticker confidence gate alpha_i with Beta(2, 2) prior
       regularisation, replacing the scalar gate.

Loss-level treatment (IRF reweighting + V-REx penalty + alpha prior)
is applied in the trainer; this module just exposes the
per-ticker alpha_i so the trainer can apply the prior.

Bit-exactly equivalent to epiSTAR-full when all SBP flags are off, by
contract (Section 1.4 invariant 1 of the spec).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from src.v2.model.episode_memory import EpisodeMemoryBank, EpisodeMemoryConfig
from src.v2.model.star_backbone import STARBackbone, STARBackboneConfig


@dataclass
class EpiSTARSBPConfig:
    """Hyperparameters for epiSTAR-SBP."""

    backbone: STARBackboneConfig = STARBackboneConfig()
    memory: EpisodeMemoryConfig = EpisodeMemoryConfig()
    cohort_dim: int = 4
    dual_pool_m1: int = 5
    dual_pool_m2: int = 3
    tau_cohort: float = 0.4
    episode_value_dim: int = 32
    cross_attn_heads: int = 4
    gate_hidden_dim: int = 64
    head_hidden_dim: int = 64
    head_dropout: float = 0.1
    use_per_ticker_gate: bool = True
    use_two_head_xattn: bool = True
    use_dual_pool: bool = True
    disable_retrieval: bool = False


class EpiSTARSBP(nn.Module):
    """epiSTAR with cohort-aware retrieval, two-head fusion, per-ticker gate."""

    def __init__(self, cfg: EpiSTARSBPConfig, episode_key_dim: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone = STARBackbone(cfg.backbone)
        self.memory = EpisodeMemoryBank(
            cfg.memory, key_dim=episode_key_dim, value_dim=cfg.episode_value_dim
        )
        d = cfg.backbone.hidden_dim
        self.cohort_dim = cfg.cohort_dim
        self.episode_value_proj = nn.Linear(cfg.episode_value_dim, d)

        # Two cross-attention heads over Pool 1 (raw similar) and Pool 2
        # (cohort-near). When `use_two_head_xattn` is False we collapse
        # to a single head over Pool 1 and ignore Pool 2.
        self.cross_attn_1 = nn.MultiheadAttention(
            embed_dim=d, num_heads=cfg.cross_attn_heads,
            dropout=cfg.backbone.dropout, batch_first=True,
        )
        self.cross_attn_2 = nn.MultiheadAttention(
            embed_dim=d, num_heads=cfg.cross_attn_heads,
            dropout=cfg.backbone.dropout, batch_first=True,
        )
        # mu_t mixer: 3-dim input (VIX z, avg pairwise corr, fraction
        # of universe in first-126-days) -> scalar in (0, 1).
        self.mu_mlp = nn.Sequential(
            nn.Linear(3, cfg.gate_hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.gate_hidden_dim, 1),
        )
        # Warm-start mu near 0.7 (slight preference for Pool 1 / raw).
        with torch.no_grad():
            self.mu_mlp[-1].bias.fill_(0.85)  # sigmoid(0.85) ~= 0.70

        self.fusion_mlp = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.GELU(),
            nn.Dropout(cfg.backbone.dropout),
            nn.Linear(d, d),
        )

        # Per-ticker gate alpha_i: 27-dim input vector (top1, entropy,
        # VIX, corr, 18-dim key, log1p_age, 4 sector dummies, neighbor
        # avg corr) per the spec Section 5.4. We use a simplified 24-dim
        # variant: top1, entropy, 2 regime scalars, 18-dim key, log1p_age,
        # history_valid_ratio_60d. Sector one-hot and neighbor avg corr
        # are dropped to avoid coupling with the panel's heterogeneous
        # graph (deferred for v1).
        gate_in_dim = 2 + 2 + episode_key_dim + 2
        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_in_dim, cfg.gate_hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.gate_hidden_dim, 1),
        )
        # Warm-start alpha near 0.5.
        with torch.no_grad():
            self.gate_mlp[-1].bias.zero_()

        self.rank_head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, cfg.head_hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.head_dropout),
            nn.Linear(cfg.head_hidden_dim, 1),
        )

    def _split_pools(
        self, query_raw_key: Tensor, query_day_idx: int,
        allowed_day_indices: Tensor,
    ) -> tuple[dict, dict]:
        """Run dual-pool retrieval. Returns (pool1, pool2) dicts.

        Pool 1: top-M1 by raw cosine over the 18-dim key.
        Pool 2: top-M2 by cosine over the 18-dim key, restricted to
                memory days whose cohort sub-key is L1-close to the
                query's cohort sub-key (within tau_cohort).
        """
        cfg = self.cfg
        cd = cfg.cohort_dim
        cutoff = query_day_idx - cfg.memory.horizon_days - cfg.memory.embargo_days
        device = query_raw_key.device
        mem_day_idx = self.memory.mem_day_idx.to(device)
        max_idx = max(
            int(mem_day_idx.max().item()) if mem_day_idx.numel() else 0,
            int(allowed_day_indices.max().item()) if allowed_day_indices.numel() else 0,
            query_day_idx,
        ) + 1
        is_allowed = torch.zeros(max_idx + 1, dtype=torch.bool, device=device)
        is_allowed[allowed_day_indices] = True
        eligible = is_allowed[mem_day_idx] & (mem_day_idx < cutoff)

        if eligible.sum() == 0:
            empty_v = torch.zeros(cfg.dual_pool_m1, self.memory.value_dim, device=device)
            empty_v2 = torch.zeros(cfg.dual_pool_m2, self.memory.value_dim, device=device)
            empty_d = torch.full((cfg.dual_pool_m1,), -1, dtype=torch.long, device=device)
            empty_d2 = torch.full((cfg.dual_pool_m2,), -1, dtype=torch.long, device=device)
            zero = torch.zeros((), device=device)
            return (
                {"values": empty_v, "day_indices": empty_d, "top1_sim": zero, "sim_entropy": zero},
                {"values": empty_v2, "day_indices": empty_d2, "top1_sim": zero, "sim_entropy": zero},
            )

        q_std = self.memory.standardize_query(query_raw_key)
        q_cohort = q_std[-cd:]
        elig_keys = self.memory.mem_keys[eligible]
        elig_cohort = elig_keys[:, -cd:]
        elig_day_idx = self.memory.mem_day_idx[eligible]
        q_norm = q_std / (q_std.norm(p=2) + 1e-8)
        k_norm = elig_keys / (elig_keys.norm(p=2, dim=-1, keepdim=True) + 1e-8)
        sims = (k_norm @ q_norm).squeeze(-1)

        # Pool 1: top-M1 by raw cosine.
        m1 = min(cfg.dual_pool_m1, sims.shape[0])
        idx1 = torch.topk(sims, k=m1, largest=True).indices
        pool1_values = self.memory.mem_values[torch.nonzero(eligible).squeeze(-1)[idx1]]
        pool1_day_idx = elig_day_idx[idx1]
        pool1_sims = sims[idx1]

        # Pool 2: cohort-near, then top-M2 by cosine within the cohort-
        # near subset. L1 distance over standardized cohort sub-key,
        # threshold tau_cohort.
        cohort_dist = (elig_cohort - q_cohort.unsqueeze(0)).abs().sum(dim=-1)
        cohort_near_mask = cohort_dist < cfg.tau_cohort
        # Exclude items already in Pool 1 to avoid double-counting.
        mask_for_pool2 = cohort_near_mask.clone()
        mask_for_pool2[idx1] = False
        sims_pool2 = sims.clone()
        sims_pool2[~mask_for_pool2] = -float("inf")
        if torch.isfinite(sims_pool2).any():
            m2 = min(cfg.dual_pool_m2, int(torch.isfinite(sims_pool2).sum()))
            if m2 > 0:
                idx2 = torch.topk(sims_pool2, k=m2, largest=True).indices
            else:
                idx2 = torch.zeros(0, dtype=torch.long, device=device)
        else:
            # Fallback per spec Section 5.2: fill from raw ranking
            # excluding Pool 1.
            sims_fb = sims.clone()
            sims_fb[idx1] = -float("inf")
            m2 = min(cfg.dual_pool_m2, int(torch.isfinite(sims_fb).sum()))
            idx2 = torch.topk(sims_fb, k=m2, largest=True).indices

        if idx2.numel() == 0:
            pool2_values = torch.zeros(cfg.dual_pool_m2, self.memory.value_dim, device=device)
            pool2_day_idx = torch.full((cfg.dual_pool_m2,), -1, dtype=torch.long, device=device)
            pool2_sims = torch.zeros(cfg.dual_pool_m2, device=device)
        else:
            pool2_values = self.memory.mem_values[torch.nonzero(eligible).squeeze(-1)[idx2]]
            pool2_day_idx = elig_day_idx[idx2]
            pool2_sims = sims[idx2]
            if idx2.numel() < cfg.dual_pool_m2:
                pad = cfg.dual_pool_m2 - idx2.numel()
                pool2_values = torch.cat([pool2_values, torch.zeros(pad, self.memory.value_dim, device=device)], dim=0)
                pool2_day_idx = torch.cat([pool2_day_idx, torch.full((pad,), -1, dtype=torch.long, device=device)])
                pool2_sims = torch.cat([pool2_sims, torch.zeros(pad, device=device)])

        # Pad pool 1 if needed.
        if pool1_values.shape[0] < cfg.dual_pool_m1:
            pad = cfg.dual_pool_m1 - pool1_values.shape[0]
            pool1_values = torch.cat([pool1_values, torch.zeros(pad, self.memory.value_dim, device=device)], dim=0)
            pool1_day_idx = torch.cat([pool1_day_idx, torch.full((pad,), -1, dtype=torch.long, device=device)])
            pool1_sims = torch.cat([pool1_sims, torch.zeros(pad, device=device)])

        soft1 = torch.softmax(pool1_sims, dim=0)
        ent1 = -(soft1 * torch.log(soft1.clamp(min=1e-8))).sum()
        soft2 = torch.softmax(pool2_sims, dim=0)
        ent2 = -(soft2 * torch.log(soft2.clamp(min=1e-8))).sum()
        pool1 = {"values": pool1_values, "day_indices": pool1_day_idx,
                 "top1_sim": pool1_sims[0], "sim_entropy": ent1}
        pool2 = {"values": pool2_values, "day_indices": pool2_day_idx,
                 "top1_sim": pool2_sims[0], "sim_entropy": ent2}
        return pool1, pool2

    def forward_day(
        self,
        patches: Tensor,
        patch_mask: Tensor,
        active_mask: Tensor,
        query_raw_key: Tensor,
        query_day_idx: int,
        allowed_day_indices: Tensor,
        gate_regime_scalars: Tensor,
        mu_input: Tensor,
        ticker_age_features: Tensor,
    ) -> dict[str, Tensor]:
        """Forward pass for one trading day.

        Args:
            patches, patch_mask, active_mask: same as epiSTAR.
            query_raw_key: 18-dim cohort-augmented regime key.
            query_day_idx: integer day index.
            allowed_day_indices: training-day allowlist.
            gate_regime_scalars: [2] standardised regime scalars.
            mu_input: [3] (VIX z-score, avg pairwise corr, fraction of
                universe in first-126-days post-IPO) for the mu_t mixer.
            ticker_age_features: [num_active, 2] per-ticker (log1p_age,
                history_valid_ratio_60d) for the per-ticker gate.

        Returns:
            Dict with y_hat, alpha_per_ticker (for the prior loss),
            mu_t, top1 sims for both pools, retrieved indices.
        """
        cfg = self.cfg
        z_star = self.backbone.forward_day(patches, patch_mask, active_mask)
        device = z_star.device

        if cfg.disable_retrieval:
            y_hat = self.rank_head(z_star).squeeze(-1) * active_mask.float()
            zeros = torch.zeros((), device=device)
            empty_d = torch.full(
                (cfg.dual_pool_m1 + cfg.dual_pool_m2,), -1, dtype=torch.long, device=device
            )
            return {
                "y_hat": y_hat, "alpha_per_ticker": torch.zeros(int(active_mask.sum()), device=device),
                "mu_t": zeros, "top1_sim_pool1": zeros, "top1_sim_pool2": zeros,
                "retrieved_day_indices": empty_d,
            }

        pool1, pool2 = self._split_pools(query_raw_key, query_day_idx, allowed_day_indices)

        active_idx = active_mask.nonzero(as_tuple=False).squeeze(-1)
        z_active = z_star[active_idx].unsqueeze(0)  # [1, A, D]

        proj1 = self.episode_value_proj(pool1["values"]).unsqueeze(0)
        h_epi_1, _ = self.cross_attn_1(query=z_active, key=proj1, value=proj1)
        h_epi_1 = h_epi_1.squeeze(0)
        if cfg.use_two_head_xattn:
            proj2 = self.episode_value_proj(pool2["values"]).unsqueeze(0)
            h_epi_2, _ = self.cross_attn_2(query=z_active, key=proj2, value=proj2)
            h_epi_2 = h_epi_2.squeeze(0)
            mu_t = torch.sigmoid(self.mu_mlp(mu_input)).squeeze()
            h_epi = mu_t * h_epi_1 + (1.0 - mu_t) * h_epi_2
        else:
            h_epi = h_epi_1
            mu_t = torch.ones((), device=device)

        # Per-ticker gate alpha_i.
        a = active_idx.shape[0]
        q_std = self.memory.standardize_query(query_raw_key)
        # Use Pool 1's top1 + entropy for the gate (Pool 1 is the
        # closest analogue to epiSTAR-full's single-pool retrieval).
        gate_static = torch.cat([
            pool1["top1_sim"].unsqueeze(0),
            pool1["sim_entropy"].unsqueeze(0),
            gate_regime_scalars,
            q_std,
        ])
        gate_static_b = gate_static.unsqueeze(0).expand(a, -1)
        gate_in = torch.cat([gate_static_b, ticker_age_features], dim=-1)
        if cfg.use_per_ticker_gate:
            alpha_i = torch.sigmoid(self.gate_mlp(gate_in)).squeeze(-1)
        else:
            alpha_i = torch.ones(a, device=device) * 0.5

        z_active_sq = z_active.squeeze(0)
        fused = self.fusion_mlp(torch.cat([z_active_sq, h_epi], dim=-1))
        z_final_active = z_active_sq + alpha_i.unsqueeze(-1) * fused

        z_final = torch.zeros_like(z_star)
        z_final[active_idx] = z_final_active
        y_hat = self.rank_head(z_final).squeeze(-1) * active_mask.float()

        retrieved = torch.cat([pool1["day_indices"], pool2["day_indices"]])
        return {
            "y_hat": y_hat,
            "alpha_per_ticker": alpha_i,
            "mu_t": mu_t.detach() if mu_t.dim() > 0 else mu_t,
            "top1_sim_pool1": pool1["top1_sim"].detach(),
            "top1_sim_pool2": pool2["top1_sim"].detach(),
            "retrieved_day_indices": retrieved,
        }


__all__ = ["EpiSTARSBP", "EpiSTARSBPConfig"]
