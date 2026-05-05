"""Shared loss functions for MARS / STAR / Combined v2.

cs_mse_loss — cross-sectional Mean Squared Error on z-scored forward
returns. Non-vanishing gradients at uninformative rankings, unlike
RankNet. This is what unblocked training in Combined v1.

pinball_loss — quantile regression loss for the risk head.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor


def cs_mse_loss(y_hat: Tensor, y_true: Tensor, mask: Tensor,
                label_smoothing: float = 0.0) -> Tensor:
    """Cross-sectional MSE on z-scored forward returns.

    Computes MSE(y_hat[mask], (1 - eps) * zscore(y_true[mask])), where z-score
    is taken across ACTIVE tickers on a single day and `eps = label_smoothing`
    shrinks the target toward zero to reduce overfitting.
    """
    m = mask.bool()
    yh = y_hat[m]; yt = y_true[m]
    if yt.numel() < 2:
        return torch.zeros((), device=y_hat.device, dtype=y_hat.dtype)
    mu = yt.mean()
    sd = yt.std().clamp(min=1e-6)
    yt_zs = (yt - mu) / sd
    if label_smoothing > 0:
        yt_zs = (1.0 - label_smoothing) * yt_zs
    return ((yh - yt_zs) ** 2).mean()


def pinball_loss(y_true: Tensor, q_hat: Tensor,
                 taus: tuple[float, ...], mask: Tensor) -> Tensor:
    """Quantile regression loss. q_hat: [N, Q], y_true: [N], mask: [N]."""
    m = mask.bool()
    if m.sum() < 2:
        return torch.zeros((), device=q_hat.device, dtype=q_hat.dtype)
    yt = y_true[m].unsqueeze(-1)           # [N, 1]
    qh = q_hat[m]                          # [N, Q]
    tau = torch.tensor(taus, device=q_hat.device, dtype=q_hat.dtype).unsqueeze(0)
    diff = yt - qh                         # [N, Q]
    return torch.maximum(tau * diff, (tau - 1) * diff).mean()


def cs_robust_loss(y_hat: Tensor, y_true: Tensor, mask: Tensor,
                   delta: float = 1.0, vol: Tensor | None = None,
                   label_smoothing: float = 0.0) -> Tensor:
    """Cross-sectional robust loss for regime-transition robustness.

    Two components (both orthogonal, composable):
      (a) Huber (delta) instead of MSE: caps the influence of large per-
          ticker errors so that extreme-return outliers don't dominate
          the gradient during drawdown regimes.
      (b) Inverse-volatility weighting (optional): per-ticker weight
          w_i = 1 / (1 + vol_i), normalized to sum to the number of
          active tickers. Down-weights high-vol tickers which are the
          draggers on fold 2 per per-ticker attribution (D1) and the
          fold-2-specific factor per D2.

    Set `vol=None` to disable the inverse-vol weighting (pure Huber
    loss). Set `delta=inf` to disable Huber (pure vol-weighted MSE).
    """
    m = mask.bool()
    yh = y_hat[m]; yt = y_true[m]
    if yt.numel() < 2:
        return torch.zeros((), device=y_hat.device, dtype=y_hat.dtype)
    mu = yt.mean()
    sd = yt.std().clamp(min=1e-6)
    yt_zs = (yt - mu) / sd
    if label_smoothing > 0:
        yt_zs = (1.0 - label_smoothing) * yt_zs
    diff = yh - yt_zs
    abs_diff = diff.abs()
    # Huber (delta=inf disables the cap and reduces to MSE)
    if not math.isfinite(delta):
        huber = 0.5 * diff ** 2
    else:
        huber = torch.where(
            abs_diff < delta,
            0.5 * diff ** 2,
            delta * (abs_diff - 0.5 * delta),
        )
    # Inverse-vol weights
    if vol is not None:
        v = vol[m].clamp(min=0.0)
        w = 1.0 / (1.0 + v)
        w = w * (w.numel() / w.sum().clamp(min=1e-6))
        return (w * huber).mean()
    return huber.mean()


def cs_group_relative_robust_loss(
    y_hat: Tensor, y_true: Tensor, mask: Tensor,
    neighbor_idx: Tensor,  # [N, K] graph neighbors (-1 pad), same graph used by the model
    delta: float = 1.0, vol: Tensor | None = None,
) -> Tensor:
    """Group-relative cross-sectional robust loss (Proposal B).

    Fold-2 diagnostic: PC1 variance share jumps to 40% (vs 27-32% on
    folds 1 and 3) so a single common factor dominates the cross-section.
    Global z-scoring contaminates the target with this common move.

    Group-relative fix: for each active ticker i, z-score the target
    using only tickers in i's graph neighborhood (ego + 8 graph neighbors,
    filtering inactive). The model then predicts "alpha within your
    neighborhood" rather than "alpha within the 84-ticker universe".

    Loss per ticker: Huber(delta) on (y_hat[i] - zs_group_i[i]) with
    optional inverse-volatility weighting.

    Causality: neighbor_idx is built from train-fold data only (the
    same graph the model uses for attention). y_true statistics are
    computed only from other tickers' ACTUAL returns on day t,
    preserving audit 2 (no leakage beyond what's already in the loss).
    """
    m = mask.bool()
    N = y_hat.shape[0]
    K = neighbor_idx.shape[1]
    device = y_hat.device

    # For each i, group = [i] + [nbr[i, 0..K-1] filtered to active].
    # Build [N, K+1] index matrix and a [N, K+1] validity mask.
    self_idx = torch.arange(N, device=device).unsqueeze(1)        # [N, 1]
    group_idx = torch.cat([self_idx, neighbor_idx], dim=1)        # [N, K+1]
    # -1 padding => force to self index, but mark invalid
    valid_nbr = neighbor_idx >= 0                                  # [N, K]
    group_valid = torch.cat([torch.ones(N, 1, dtype=torch.bool, device=device),
                             valid_nbr], dim=1)                    # [N, K+1]
    safe_idx = torch.where(group_idx >= 0, group_idx, self_idx.expand(-1, K + 1))
    # Cross-reference with active mask on the target
    active = m.to(torch.bool)
    group_active = active[safe_idx] & group_valid                  # [N, K+1]

    y_group = y_true[safe_idx]                                     # [N, K+1]
    y_group = torch.where(group_active, y_group, torch.zeros_like(y_group))
    count = group_active.to(y_true.dtype).sum(dim=1).clamp(min=1.0)  # [N]
    mu_g = y_group.sum(dim=1) / count                              # [N]
    # Variance
    centered = (y_group - mu_g.unsqueeze(1)) * group_active.to(y_true.dtype)
    var_g = (centered ** 2).sum(dim=1) / count.clamp(min=1.0)      # [N]
    sd_g = var_g.clamp(min=1e-6).sqrt()                            # [N]

    # Only use rows with at least 3 active members for the group stats;
    # otherwise fall back to global z-score on active tickers.
    enough = count >= 3.0
    y_global = y_true[active]
    if y_global.numel() >= 2:
        mu_gl = y_global.mean()
        sd_gl = y_global.std().clamp(min=1e-6)
    else:
        mu_gl = torch.zeros((), device=device, dtype=y_true.dtype)
        sd_gl = torch.ones((), device=device, dtype=y_true.dtype)
    mu_final = torch.where(enough, mu_g, mu_gl.expand_as(mu_g))
    sd_final = torch.where(enough, sd_g, sd_gl.expand_as(sd_g))

    yt_zs = (y_true - mu_final) / sd_final.clamp(min=1e-6)
    yt_active = yt_zs[m]
    yh_active = y_hat[m]
    if yt_active.numel() < 2:
        return torch.zeros((), device=device, dtype=y_hat.dtype)

    diff = yh_active - yt_active
    abs_diff = diff.abs()
    if not math.isfinite(delta):
        huber = 0.5 * diff ** 2
    else:
        huber = torch.where(
            abs_diff < delta,
            0.5 * diff ** 2,
            delta * (abs_diff - 0.5 * delta),
        )
    if vol is not None:
        v = vol[m].clamp(min=0.0)
        w = 1.0 / (1.0 + v)
        w = w * (w.numel() / w.sum().clamp(min=1e-6))
        return (w * huber).mean()
    return huber.mean()


__all__ = [
    "cs_mse_loss",
    "cs_robust_loss",
    "cs_group_relative_robust_loss",
    "pinball_loss",
]
