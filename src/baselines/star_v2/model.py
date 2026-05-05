"""STAR baseline for the v2 paper (renamed from R-STAR).

Wraps the shared `STARBackbone` with a simple rank head, trained against
the Huber + inverse-volatility-weighted loss (`cs_robust_loss`). This is
the v1 "iter 10" recipe, distilled onto v2 infrastructure: same panel
loader, same fold definitions, same evaluation protocol as epiSTAR and
DyReg-STAR.

Distinct from epiSTAR (no episodic retrieval, no cross-attention) and
from DyReg-STAR (no dynamic graph, uses the static mechanistic graph).
"""
from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn

from src.v2.model.star_backbone import STARBackbone, STARBackboneConfig


@dataclass
class STARBaselineConfig:
    """Hyperparameters for the v2 STAR baseline."""

    backbone: STARBackboneConfig = STARBackboneConfig()
    head_hidden_dim: int = 64
    head_dropout: float = 0.1


class STARBaseline(nn.Module):
    """STAR backbone + 2-layer Multi-Layer Perceptron rank head."""

    def __init__(self, cfg: STARBaselineConfig) -> None:
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
        """Forward pass for one trading day."""
        z = self.backbone.forward_day(patches, patch_mask, active_mask)
        y_hat = self.rank_head(z).squeeze(-1)
        y_hat = y_hat * active_mask.float()
        return {"y_hat": y_hat, "z": z}


__all__ = ["STARBaseline", "STARBaselineConfig"]
