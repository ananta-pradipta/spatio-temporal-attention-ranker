"""InVAR-STAR auxiliary losses.

Three losses contribute to the composite training objective alongside the
primary MSE on cross-sectional z-scored targets:

  - throttle_kl_prior: histogram-based KL of the empirical beta_t batch
    distribution to a Beta(0.4, 0.6) bimodal prior. Encourages the gate to
    learn decisive on-or-off behaviour rather than constant intermediate
    gating. Bejnordi et al. (2019), adapted from per-channel gates to a
    scalar daily gate.
  - load_balance_loss: Switch-Transformer style auxiliary that pressures the
    router to spread token assignment uniformly across experts. Fedus et al.
    (2022).
  - weighted_pearson_ic_loss: negative Pearson correlation between predicted
    scores and next-day cross-sectional returns. Standard MASTER/Qlib
    surrogate for rank IC.

Composite: L_total = L_mse + 0.5 * L_neg_wic + 0.05 * L_throttle + 0.01 * L_balance
(weights are config-tunable; defaults from design doc Section 4.7).
"""
from __future__ import annotations

import torch
from torch import Tensor


def throttle_kl_prior(beta_batch: Tensor,
                      alpha0: float = 0.4, alpha1: float = 0.6,
                      n_bins: int = 10) -> Tensor:
    """Batch-shaping KL of empirical beta distribution to Beta(alpha0, alpha1).

    Histogram approximation with `n_bins` (default 10) bins on [0, 1]. The
    target Beta(0.4, 0.6) is bimodal at the extremes, pushing the gate
    toward decisive on-or-off behaviour rather than constant intermediate
    gating.

    Args:
        beta_batch: shape (B, 1) or (B,). Empirical beta_t values for the batch.
        alpha0: Beta prior parameter (default 0.4).
        alpha1: Beta prior parameter (default 0.6).
        n_bins: number of histogram bins on [0, 1].

    Returns:
        Scalar KL divergence (non-negative).
    """
    flat = beta_batch.flatten()
    bins = torch.linspace(0.0, 1.0, n_bins + 1, device=flat.device)
    hist = torch.histc(flat.detach(), bins=n_bins, min=0.0, max=1.0)
    p = (hist + 1.0e-3) / (hist.sum() + 1.0e-2)
    centers = 0.5 * (bins[1:] + bins[:-1])
    log_q = (
        (alpha0 - 1.0) * torch.log(centers.clamp_min(1.0e-6))
        + (alpha1 - 1.0) * torch.log((1.0 - centers).clamp_min(1.0e-6))
    )
    q = torch.softmax(log_q, dim=0)
    return (p * (p.clamp_min(1.0e-8).log() - q.clamp_min(1.0e-8).log())).sum()


def load_balance_loss(route_probs: Tensor) -> Tensor:
    """Switch-Transformer style load-balancing loss.

    Args:
        route_probs: shape (B, n_experts). Router softmax probabilities.

    Returns:
        Scalar load-balance loss. K * sum_k f_k * P_k where f_k is the
        empirical fraction of tokens assigned to expert k (via argmax) and
        P_k is the mean router probability for expert k.
    """
    K = route_probs.shape[-1]
    argmax = route_probs.argmax(dim=-1, keepdim=True)
    arange = torch.arange(K, device=route_probs.device)
    f = (argmax == arange).float().mean(dim=0)
    P = route_probs.mean(dim=0)
    return K * (f * P).sum()


def weighted_pearson_ic_loss(y_hat: Tensor, y: Tensor) -> Tensor:
    """Negative weighted Pearson correlation, cross-sectional.

    Standard MASTER/Qlib surrogate for rank IC. Per-batch Pearson correlation
    between predicted scores and next-day cross-sectional returns.

    Args:
        y_hat: shape (B, 1) or (B,). Predicted scores.
        y: shape (B, 1) or (B,). Target returns (z-scored cross-sectionally).

    Returns:
        Scalar negative Pearson correlation in [-1, 1]; we negate so it can be
        minimized as a loss.
    """
    yh = y_hat.flatten()
    yt = y.flatten()
    yh = yh - yh.mean()
    yt = yt - yt.mean()
    num = (yh * yt).sum()
    den = (yh.pow(2).sum().sqrt() * yt.pow(2).sum().sqrt()).clamp_min(1.0e-8)
    return -(num / den)
