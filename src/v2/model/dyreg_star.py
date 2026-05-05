"""DyReg-STAR: Dynamic Regime Graph Spatio-Temporal Attention Ranker.

Stage 1 implementation: STAR backbone with dynamic regime-aware neighbor
selection. The neighbor list is recomputed per day from a rolling-window
correlation graph (data up to day t only); the backbone otherwise matches
the pure STAR encoder. No mixture-of-experts heads in Stage 1.

Stage 2 (regime-gated expert heads, calm/stress/idiosyncratic) is left as
an extension; the rank head here is a single 2-layer GELU Multi-Layer
Perceptron (MLP) with LayerNorm.
"""
from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn

from src.v2.model.star_backbone import STARBackbone, STARBackboneConfig


@dataclass
class DyRegSTARConfig:
    """Hyperparameters for DyReg-STAR.

    Attributes:
        backbone: STAR backbone hyperparameters.
        head_hidden_dim: hidden width of the rank head MLP.
        head_dropout: dropout probability inside the rank head.
    """

    backbone: STARBackboneConfig = STARBackboneConfig()
    head_hidden_dim: int = 64
    head_dropout: float = 0.1


class DyRegSTAR(nn.Module):
    """Dynamic-graph variant of STAR.

    The model itself is the standard STAR backbone plus a rank head;
    the dynamic graph is consumed through patch construction at the
    training-loop layer (not inside this module).
    """

    def __init__(self, cfg: DyRegSTARConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone = STARBackbone(cfg.backbone)
        d = cfg.backbone.hidden_dim
        self.rank_head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, cfg.head_hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.head_dropout),
            nn.Linear(cfg.head_hidden_dim, 1),
        )

    def forward_day(
        self, patches: Tensor, patch_mask: Tensor, active_mask: Tensor
    ) -> dict[str, Tensor]:
        """Forward pass for one trading day.

        Args:
            patches: [A, N+1, W, F] STAR input patches built from the
                day-specific dynamic neighbor list.
            patch_mask: [A, N+1, W] bool, True where observed.
            active_mask: [num_nodes] bool, True for active tickers.

        Returns:
            Dict with y_hat: [num_nodes] predicted scores (zero for
            inactive rows) and z: backbone hidden representation.
        """
        z = self.backbone.forward_day(patches, patch_mask, active_mask)
        y_hat = self.rank_head(z).squeeze(-1)
        y_hat = y_hat * active_mask.float()
        return {"y_hat": y_hat, "z": z}


__all__ = ["DyRegSTAR", "DyRegSTARConfig"]
