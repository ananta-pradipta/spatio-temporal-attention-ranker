"""FactorCalibrator for FC-DGraph-epiSTAR.

Per spec Part E. A small regime-mixed factor-pricing model. The
key idea: factor premia (theta) change across regimes; we learn
n_regimes regime templates of factor coefficients and route each
day to a soft mixture of regimes via an MLP over a regime-feature
vector. Output `score_factor = factor_exposures @ theta_t`, then
combine with the OW score via z-scoring + lambda gate.

The cross-sectional z-score on score_ow AND score_factor is
critical: it prevents either component from contaminating the other
with day-level constants and ensures the lambda combination is on
comparable scale.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class FactorCalibratorConfig:
    """Hyperparameters for the factor calibrator."""

    n_factors: int = 14
    regime_dim: int = 8
    n_regimes: int = 4
    router_hidden_dim: int = 32
    lambda_init_raw: float = -2.5    # sigmoid(-2.5) ~= 0.076 (small start)
    use_learned_lambda: bool = True
    deterministic_lambda_floor: float = 0.05
    deterministic_lambda_scale: float = 0.25


def masked_zscore(x: Tensor, mask: Tensor) -> Tensor:
    """Cross-sectional z-score over masked entries. Zeros elsewhere.

    Args:
        x: [N] float per-ticker score.
        mask: [N] bool active mask.

    Returns:
        [N] z-scored scores; 0 outside mask.
    """
    m = mask.to(x.dtype)
    n = m.sum().clamp(min=1.0)
    mu = (x * m).sum() / n
    var = ((x - mu) * m).pow(2).sum() / n
    sd = var.clamp(min=1e-12).sqrt()
    z = (x - mu) / sd.clamp(min=1e-6)
    return z * m


class FactorCalibrator(nn.Module):
    """Regime-mixed factor calibrator. score_total = z(score_ow) +
    lambda * z(score_factor)."""

    def __init__(self, cfg: FactorCalibratorConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.regime_router = nn.Sequential(
            nn.LayerNorm(cfg.regime_dim),
            nn.Linear(cfg.regime_dim, cfg.router_hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.router_hidden_dim, cfg.n_regimes),
        )
        # Regime-specific factor premia: [n_regimes, n_factors].
        self.theta = nn.Parameter(
            torch.zeros(cfg.n_regimes, cfg.n_factors, dtype=torch.float32)
        )
        # Learned lambda (raw logit; sigmoid in forward).
        self.lambda_raw = nn.Parameter(
            torch.tensor(cfg.lambda_init_raw, dtype=torch.float32)
        )

    def forward(
        self,
        score_ow: Tensor,             # [N]
        factor_exposures: Tensor,     # [N, K]
        regime_features: Tensor,      # [regime_dim]
        mask: Tensor,                 # [N] bool
        deterministic_lambda: Tensor | None = None,
    ) -> tuple[Tensor, dict]:
        """Per-day forward. Returns score_total [N] + diagnostics."""
        cfg = self.cfg

        # Regime mixture (soft).
        pi = F.softmax(self.regime_router(regime_features), dim=-1)   # [n_regimes]
        theta_t = pi @ self.theta                                       # [n_factors]

        # Per-ticker factor score.
        score_factor_raw = factor_exposures @ theta_t                   # [N]

        # Cross-sectional z-score per day on BOTH components (spec
        # Part E "Critical Implementation Rule").
        score_ow_z = masked_zscore(score_ow.float(), mask)
        score_factor_z = masked_zscore(score_factor_raw.float(), mask)

        if deterministic_lambda is not None:
            lambda_factor = deterministic_lambda.to(score_ow_z.dtype)
        elif cfg.use_learned_lambda:
            lambda_factor = torch.sigmoid(self.lambda_raw).to(score_ow_z.dtype)
        else:
            lambda_factor = torch.tensor(0.1, dtype=score_ow_z.dtype, device=score_ow_z.device)

        score_total = score_ow_z + lambda_factor * score_factor_z
        score_total = score_total * mask.to(score_total.dtype)

        diag = {
            "score_ow_z": score_ow_z.detach(),
            "score_factor_z": score_factor_z.detach(),
            "lambda_factor": lambda_factor.detach() if isinstance(lambda_factor, Tensor) else torch.tensor(float(lambda_factor)),
            "theta_t": theta_t.detach(),
            "regime_probs": pi.detach(),
        }
        return score_total, diag


__all__ = [
    "FactorCalibratorConfig",
    "FactorCalibrator",
    "masked_zscore",
]
