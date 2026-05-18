"""DualGraphBranch: correlation graph + learned attention graph.

Per spec section 6.4.

Two graph sources blended via macro-conditioned sigmoid:
  - A_corr: rolling 60-day Pearson correlation, reliability-shrunk by overlap,
    top-K=8 neighbors per ticker per day.
  - A_learned: dot-product attention over per-stock embeddings, top-K masked.

`alpha_blend = sigmoid(MLP(macro_state))`. Initialised at 0.5 by setting the
final-layer bias to 0 (sigmoid(0) = 0.5).

The correlation graph A_corr is computed externally (in the trainer's data
prep) and passed as a [B, N, K] index tensor. The learned attention graph is
computed inside this module from current per-stock embeddings.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class DualGraphBranchConfig:
    d_model: int = 128
    macro_dim: int = 24
    blend_init: float = 0.5
    top_k: int = 8


class DualGraphBranch(nn.Module):
    """Correlation + learned-attention graph blend with macro-conditioned gate."""

    def __init__(self, cfg: DualGraphBranchConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or DualGraphBranchConfig()
        self.cfg = cfg
        # Blend gate: macro -> scalar in (0, 1). Init bias 0 so blend starts 0.5.
        self.blend_mlp = nn.Sequential(
            nn.Linear(cfg.macro_dim, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, 1),
        )
        with torch.no_grad():
            self.blend_mlp[-1].bias.zero_()

        # Per-edge MLP transforming neighbor's embedding before aggregation.
        self.edge_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.norm = nn.LayerNorm(cfg.d_model)

    def _aggregate_topk(
        self, z: Tensor, neighbor_idx: Tensor, neighbor_mask: Tensor,
    ) -> Tensor:
        """Aggregate neighbor embeddings via mean over top-K.

        Args:
            z: [B, N, d_model] per-stock embeddings.
            neighbor_idx: [B, N, K] long indices into the N axis (-1 for pad).
            neighbor_mask: [B, N, K] bool, True for valid neighbors.
        """
        B, N, D = z.shape
        K = neighbor_idx.shape[-1]
        idx = neighbor_idx.clamp(min=0)  # avoid -1 gather
        idx_flat = idx.reshape(B, N * K, 1).expand(-1, -1, D)
        gathered = torch.gather(z, dim=1, index=idx_flat).view(B, N, K, D)
        gathered = gathered * neighbor_mask.unsqueeze(-1).float()
        n_neighbors = neighbor_mask.float().sum(dim=-1, keepdim=True).clamp(min=1.0)
        agg = gathered.sum(dim=2) / n_neighbors
        return self.edge_proj(agg)

    def _build_learned_topk(
        self, z: Tensor, active_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Per-day top-K via dot-product attention; mask-respecting."""
        B, N, D = z.shape
        K = self.cfg.top_k
        scores = torch.matmul(z, z.transpose(1, 2)) / (D ** 0.5)  # [B, N, N]
        # Mask out inactive columns
        col_mask = ~active_mask.unsqueeze(1).expand(B, N, N)
        scores = scores.masked_fill(col_mask, float("-inf"))
        # Mask the diagonal (self) so we don't pick self as neighbor
        eye = torch.eye(N, dtype=torch.bool, device=z.device).unsqueeze(0).expand(B, N, N)
        scores = scores.masked_fill(eye, float("-inf"))
        # Top-K
        top = torch.topk(scores, k=min(K, N - 1), dim=-1)
        idx = top.indices                                     # [B, N, K]
        valid = torch.isfinite(top.values)                    # [B, N, K]
        return idx, valid

    def forward(
        self, z: Tensor, active_mask: Tensor, macro_state: Tensor,
        corr_neighbor_idx: Tensor, corr_neighbor_mask: Tensor,
    ) -> Tensor:
        """Blend correlation and learned graphs.

        Args:
            z: [B, N, d_model] per-stock embeddings.
            active_mask: [B, N] active mask.
            macro_state: [B, macro_dim] per-day macro state.
            corr_neighbor_idx: [B, N, K] long indices for correlation graph.
            corr_neighbor_mask: [B, N, K] bool for valid corr neighbors.

        Returns:
            [B, N, d_model] graph-aggregated embeddings.
        """
        alpha = torch.sigmoid(self.blend_mlp(macro_state)).unsqueeze(-1)  # [B, 1, 1]
        z_corr = self._aggregate_topk(z, corr_neighbor_idx, corr_neighbor_mask)
        learned_idx, learned_mask = self._build_learned_topk(z, active_mask)
        z_learned = self._aggregate_topk(z, learned_idx, learned_mask)
        z_graph = alpha * z_corr + (1.0 - alpha) * z_learned
        # Zero inactive cells
        z_graph = z_graph * active_mask.unsqueeze(-1).float()
        return self.norm(z_graph)


__all__ = ["DualGraphBranch", "DualGraphBranchConfig"]
