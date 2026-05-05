"""epiSTAR: Episodic Spatio-Temporal Attention Ranker.

Composition:
    1. STAR backbone produces per-ticker hidden representations.
    2. EpisodeMemoryBank retrieves top-M leakage-safe historical episodes
       for the query day's regime context.
    3. Cross-attention fuses retrieved episode summaries with the query's
       STAR representation.
    4. A confidence gate decides how much retrieved context to mix in.
    5. A two-layer Multi-Layer Perceptron (MLP) rank head outputs scores.

The retrieval is per-day (one query, one set of M retrieved episodes for
all active tickers on that day). The cross-attention is per-ticker
(every active ticker uses the same M episodes but attends to them with
its own STAR representation as the query).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from src.v2.model.episode_memory import EpisodeMemoryBank, EpisodeMemoryConfig
from src.v2.model.star_backbone import STARBackbone, STARBackboneConfig


@dataclass
class EpiSTARConfig:
    """Hyperparameters for the epiSTAR model.

    Attributes:
        backbone: STAR backbone configuration.
        memory: Episode memory bank configuration.
        episode_value_dim: dimension of stored episode values (regime key
            features plus a few STAR day-level summary statistics).
        cross_attn_heads: number of heads in the cross-attention block.
        gate_hidden_dim: hidden width of the confidence gate MLP.
        head_hidden_dim: hidden width of the rank head MLP.
        head_dropout: dropout probability inside the rank head.
        disable_gate: ablation switch; if True, alpha is fixed to 1.0.
        disable_retrieval: ablation switch; if True, the model degenerates
            to a pure STAR forward pass.
    """

    backbone: STARBackboneConfig = STARBackboneConfig()
    memory: EpisodeMemoryConfig = EpisodeMemoryConfig()
    episode_value_dim: int = 32
    cross_attn_heads: int = 4
    gate_hidden_dim: int = 64
    head_hidden_dim: int = 64
    head_dropout: float = 0.1
    disable_gate: bool = False
    disable_retrieval: bool = False


class EpiSTAR(nn.Module):
    """epiSTAR model = STAR backbone + episodic retrieval + gated fusion."""

    def __init__(self, cfg: EpiSTARConfig, episode_key_dim: int) -> None:
        super().__init__()
        self.cfg = cfg

        self.backbone = STARBackbone(cfg.backbone)

        self.memory = EpisodeMemoryBank(
            cfg.memory, key_dim=episode_key_dim, value_dim=cfg.episode_value_dim
        )

        # Project episode values into the backbone hidden dimension before
        # cross-attention. This decouples value dim from hidden dim.
        d = cfg.backbone.hidden_dim
        self.episode_value_proj = nn.Linear(cfg.episode_value_dim, d)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d,
            num_heads=cfg.cross_attn_heads,
            dropout=cfg.backbone.dropout,
            batch_first=True,
        )
        self.fusion_mlp = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.GELU(),
            nn.Dropout(cfg.backbone.dropout),
            nn.Linear(d, d),
        )

        # Gate inputs (concatenated): top1_sim, sim_entropy, two regime
        # scalars (passed in by the trainer), and the standardized query key.
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
            patches: [A, N+1, W, F] STAR input patches for active tickers.
            patch_mask: [A, N+1, W] bool, True where observed.
            active_mask: [num_nodes] bool, True for active tickers.
            query_raw_key: [key_dim] raw episode key for the query day.
            query_day_idx: integer day index of the query.
            allowed_day_indices: long tensor of allowed memory days
                (training-day indices for the cleanest setting).
            gate_regime_scalars: [2] float tensor with two regime scalars
                used by the confidence gate (e.g., current VIX,
                current cross-sectional dispersion). Standardized upstream.

        Returns:
            Dict with:
                y_hat: [num_nodes] predicted scores (zero for inactive rows).
                alpha: scalar gate weight in (0, 1).
                top1_sim: scalar top-1 retrieval similarity.
                retrieved_day_indices: [M] long tensor of retrieved day indices.
                z_star: [num_nodes, hidden_dim] backbone representation.
        """
        cfg = self.cfg
        z_star = self.backbone.forward_day(patches, patch_mask, active_mask)
        device = z_star.device
        num_nodes, d = z_star.shape

        if cfg.disable_retrieval:
            y_hat = self.rank_head(z_star).squeeze(-1)
            y_hat = y_hat * active_mask.float()
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
        ep_values = retrieval["values"]
        ep_proj = self.episode_value_proj(ep_values).unsqueeze(0)  # [1, M, d]

        # Cross-attention: every active ticker queries the same M episodes.
        active_idx = torch.nonzero(active_mask, as_tuple=False).squeeze(-1)
        z_active = z_star[active_idx].unsqueeze(0)  # [1, A, d]
        h_epi, _ = self.cross_attn(query=z_active, key=ep_proj, value=ep_proj)
        h_epi = h_epi.squeeze(0)  # [A, d]
        z_active_sq = z_active.squeeze(0)  # [A, d]

        # Confidence gate: scalar per day, broadcast across active tickers.
        q_std = self.memory.standardize_query(query_raw_key)
        gate_in = torch.cat(
            [
                retrieval["top1_sim"].unsqueeze(0),
                retrieval["sim_entropy"].unsqueeze(0),
                gate_regime_scalars,
                q_std,
            ]
        )
        if cfg.disable_gate:
            alpha = torch.ones((), device=device)
        else:
            alpha = torch.sigmoid(self.gate_mlp(gate_in)).squeeze()

        fused = self.fusion_mlp(torch.cat([z_active_sq, h_epi], dim=-1))  # [A, d]
        z_final_active = z_active_sq + alpha * fused

        z_final = torch.zeros_like(z_star)
        z_final[active_idx] = z_final_active

        y_hat = self.rank_head(z_final).squeeze(-1)
        y_hat = y_hat * active_mask.float()

        return {
            "y_hat": y_hat,
            "alpha": alpha.detach(),
            "top1_sim": retrieval["top1_sim"].detach(),
            "retrieved_day_indices": retrieval["day_indices"],
            "z_star": z_star,
        }


__all__ = ["EpiSTAR", "EpiSTARConfig"]
