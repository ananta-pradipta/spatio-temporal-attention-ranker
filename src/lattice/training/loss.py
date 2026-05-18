"""Cohort-stratified ranking loss for LATTICE.

Per spec section 7.1.

Loss is the average of per-day per-cohort-axis (1 - rank_correlation):

  L_axis = mean_over_cohorts( 1 - rank_corr(y_hat[mask_c], z_target[mask_c]) )
  L_main = (L_size + L_liquidity + L_sector + L_age) / 4
  L      = L_main + lambda_balance * L_balance

For cohorts with fewer than 5 active tickers, contribute nothing (skip).

Rank correlation is implemented as a soft Spearman approximation: a
pairwise-sigmoid smoothed rank with temperature 0.01, then Pearson
correlation between soft ranks.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class LossConfig:
    rank_temperature: float = 0.01
    cohort_min_size: int = 5
    balance_loss_weight: float = 0.01
    top_decile_hinge_weight: float = 0.05


def soft_rank(x: Tensor, temperature: float = 0.01) -> Tensor:
    """Differentiable soft rank via pairwise sigmoid.

    soft_rank[i] = sum_j sigmoid((x[i] - x[j]) / temperature)

    Args:
        x: [N] vector.
        temperature: smoothing scale; tau -> 0 recovers hard rank.

    Returns:
        [N] soft-rank values in [0, N-1].
    """
    diffs = (x.unsqueeze(0) - x.unsqueeze(1)) / temperature  # [N, N]; row i, col j: x[j] - x[i]
    # We want soft_rank[i] = sum_j sigmoid((x[i] - x[j]) / tau) so use -diffs
    return torch.sigmoid(-diffs).sum(dim=0) - 0.5  # subtract 0.5 to remove self-pair contribution


def soft_spearman(y_hat: Tensor, y_true: Tensor, temperature: float = 0.01) -> Tensor:
    """Soft Spearman correlation between y_hat and y_true.

    Returns a scalar in [-1, 1] (with smoothing, slightly inside that range).
    """
    if y_hat.numel() < 2:
        return torch.zeros((), device=y_hat.device, dtype=y_hat.dtype)
    rh = soft_rank(y_hat, temperature)
    rt = soft_rank(y_true, temperature)
    rh = rh - rh.mean()
    rt = rt - rt.mean()
    num = (rh * rt).sum()
    den = (rh.pow(2).sum().sqrt() * rt.pow(2).sum().sqrt()).clamp(min=1e-9)
    return num / den


def per_axis_cohort_loss(
    y_hat: Tensor, z_target: Tensor, mask: Tensor, cohort: Tensor,
    n_cohorts: int, cfg: LossConfig,
) -> Tensor:
    """Mean-over-cohorts of (1 - soft_spearman) on one axis for one day.

    Cohorts with fewer than `cfg.cohort_min_size` valid points contribute 0.
    """
    losses = []
    for c in range(n_cohorts):
        m = mask & (cohort == c)
        if m.sum() < cfg.cohort_min_size:
            continue
        loss_c = 1.0 - soft_spearman(y_hat[m], z_target[m], cfg.rank_temperature)
        losses.append(loss_c)
    if not losses:
        return torch.zeros((), device=y_hat.device, dtype=y_hat.dtype)
    return torch.stack(losses).mean()


def cohort_stratified_ranking_loss(
    y_hat: Tensor,                    # [N]
    y_true: Tensor,                   # [N]
    active_mask: Tensor,              # [N] bool
    size_decile: Tensor,              # [N] long
    liquidity_decile: Tensor,         # [N] long
    sector_id: Tensor,                # [N] long
    age_bucket: Tensor,               # [N] long
    cfg: LossConfig | None = None,
) -> dict:
    """Compute the cohort-stratified ranking loss for one day.

    Args:
        y_hat, y_true: [N] predictions and 5-day forward log returns.
        active_mask: [N] bool.
        size_decile, liquidity_decile, sector_id, age_bucket: [N] long cohort
            labels (0-indexed; missing-cohort cells set to -1, which is
            filtered out by `mask & (cohort >= 0)` in per_axis_cohort_loss).
        cfg: LossConfig.

    Returns:
        dict with:
            "loss_size", "loss_liquidity", "loss_sector", "loss_age": per-axis scalar
            "loss_main": (1/4) * sum of per-axis losses
            "z_target": cross-sectionally z-scored y_true on active set
    """
    cfg = cfg or LossConfig()
    if active_mask.sum() < cfg.cohort_min_size:
        zero = torch.zeros((), device=y_hat.device, dtype=y_hat.dtype)
        return {"loss_size": zero, "loss_liquidity": zero,
                "loss_sector": zero, "loss_age": zero, "loss_main": zero}

    # Cross-sectional z-score on active set
    yt = y_true.clone()
    yt[~active_mask] = 0.0
    mu = yt[active_mask].mean()
    sd = yt[active_mask].std().clamp(min=1e-6)
    z_target = (y_true - mu) / sd

    losses = {}
    for label, cohort, n_cohorts in [
        ("size", size_decile, 10),
        ("liquidity", liquidity_decile, 10),
        ("sector", sector_id, 11),
        ("age", age_bucket, 4),
    ]:
        cohort_clean = cohort.clone()
        cohort_clean[cohort < 0] = -1  # mark missing
        valid_mask = active_mask & (cohort_clean >= 0)
        if valid_mask.sum() < cfg.cohort_min_size:
            losses[f"loss_{label}"] = torch.zeros((), device=y_hat.device, dtype=y_hat.dtype)
            continue
        losses[f"loss_{label}"] = per_axis_cohort_loss(
            y_hat, z_target, valid_mask, cohort_clean, n_cohorts, cfg,
        )

    losses["loss_main"] = (
        losses["loss_size"] + losses["loss_liquidity"]
        + losses["loss_sector"] + losses["loss_age"]
    ) / 4.0
    return losses


def top_decile_hinge_loss(
    y_hat: Tensor, z_target: Tensor, active_mask: Tensor, cfg: LossConfig,
) -> Tensor:
    """Hinge loss encouraging top-10% predictions to have positive z-target.

    L = mean_over_top10pct( max(0, 0.1 - z_target) )

    Optional auxiliary term per spec section 7.1 with default weight 0.05.
    """
    if active_mask.sum() < 10:
        return torch.zeros((), device=y_hat.device, dtype=y_hat.dtype)
    yh = y_hat[active_mask]
    zt = z_target[active_mask]
    n_top = max(1, int(0.1 * yh.shape[0]))
    top_idx = torch.topk(yh, k=n_top, largest=True).indices
    return torch.relu(0.1 - zt[top_idx]).mean()


__all__ = [
    "LossConfig", "soft_rank", "soft_spearman",
    "cohort_stratified_ranking_loss", "top_decile_hinge_loss",
    "per_axis_cohort_loss",
]
