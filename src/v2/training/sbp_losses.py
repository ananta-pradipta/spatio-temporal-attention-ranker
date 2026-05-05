"""Loss components for epiSTAR-SBP.

Three additions on top of cross-sectional MSE on z-scored 5-day forward
log returns:

    L_irf:          per-(day, ticker) inverse-cohort-frequency
                    reweighting of the rank loss.
    L_vrex:         variance of per-cohort mean rank losses (V-REx
                    penalty across cohort environments; Krueger et al.,
                    ICML 2021).
    L_alpha_prior:  Beta(2, 2) prior on per-ticker confidence gate
                    alpha_i to keep it away from saturation at 0 or 1.

The composite loss is L_total = L_rank + lambda_irf * L_irf
                             + lambda_vrex * L_vrex
                             + lambda_alpha_prior * L_alpha_prior

with the lambda_vrex ramp from 0 to its target over the first
vrex_warmup_epochs epochs.
"""
from __future__ import annotations

import torch
from torch import Tensor


def cs_mse_loss(
    y_hat: Tensor, y_true: Tensor, mask: Tensor,
    sample_weights: Tensor | None = None,
) -> Tensor:
    """Cross-sectional MSE on per-day z-scored true returns.

    Optional `sample_weights` of shape [num_nodes] applies per-(day,
    ticker) reweighting (used by IRF). Weights are normalised to mean 1
    across active tickers on the day so the loss scale is preserved.
    """
    m = mask.bool()
    yh = y_hat[m]; yt = y_true[m]
    if yt.numel() < 2:
        return torch.zeros((), device=y_hat.device, dtype=y_hat.dtype)
    mu = yt.mean(); sd = yt.std().clamp(min=1e-6)
    yt_zs = (yt - mu) / sd
    sq = (yh - yt_zs) ** 2
    if sample_weights is not None:
        w = sample_weights[m]
        w = w * (w.numel() / w.sum().clamp(min=1e-6))
        return (w * sq).mean()
    return sq.mean()


def cohort_irf_weights(
    cohort_buckets: Tensor, train_freq: Tensor, temper: float = 0.5
) -> Tensor:
    """Per-(day, ticker) IRF weights from cohort frequency.

    Args:
        cohort_buckets: [num_nodes] long tensor with values in
            [0, num_buckets). Inactive cells are ignored downstream.
        train_freq: [num_buckets] tensor of training-fold cohort
            frequencies (each value > 0).
        temper: tempering exponent. 0.5 is the spec's recommended
            square-root tempering (per Section 6.2).

    Returns:
        [num_nodes] float weights.
    """
    safe_freq = train_freq.clamp(min=1e-6)
    inv_freq = safe_freq ** (-temper)
    return inv_freq[cohort_buckets]


def vrex_penalty(env_losses: list[Tensor]) -> Tensor:
    """Variance of per-environment rank losses.

    Args:
        env_losses: list of scalar tensors, one per cohort bucket
            present in the current minibatch / accumulation window.
            Buckets with fewer than 2 active cells are excluded
            upstream.

    Returns:
        Scalar variance penalty (zero if fewer than 2 environments).
    """
    populated = [el for el in env_losses if el.numel() > 0]
    if len(populated) < 2:
        return torch.zeros(
            (), device=populated[0].device if populated else "cpu"
        )
    return torch.stack(populated).var(unbiased=False)


def alpha_beta_prior(alpha: Tensor) -> Tensor:
    """Negative log-likelihood of alpha_i under a Beta(2, 2) prior.

    Beta(2, 2) has density ~ x * (1 - x), peaking at 0.5. Penalising
    -log(x) - log(1 - x) keeps alpha_i in (0, 1) and away from saturation.
    """
    a = alpha.clamp(min=1e-6, max=1.0 - 1e-6)
    return -(torch.log(a) + torch.log1p(-a)).mean()


def compute_irf_freq(
    cohort_buckets_train: torch.Tensor, num_buckets: int = 4
) -> torch.Tensor:
    """Training-fold cohort-bucket relative frequencies.

    Args:
        cohort_buckets_train: 1-D tensor of cohort indices for all
            (day, ticker) cells in the training fold that pass the
            active mask.
        num_buckets: number of cohort categories.

    Returns:
        [num_buckets] tensor of frequencies summing to 1.
    """
    counts = torch.zeros(num_buckets, device=cohort_buckets_train.device)
    for k in range(num_buckets):
        counts[k] = (cohort_buckets_train == k).sum().float()
    total = counts.sum().clamp(min=1.0)
    return counts / total


__all__ = [
    "cs_mse_loss",
    "cohort_irf_weights",
    "vrex_penalty",
    "alpha_beta_prior",
    "compute_irf_freq",
]
