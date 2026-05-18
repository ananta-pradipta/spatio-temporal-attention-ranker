"""G-InVAR loss: daily cross-sectional z-scored MSE."""
from __future__ import annotations

import torch
from torch import Tensor


def cs_zscored_mse_loss(
    scores: Tensor, target: Tensor, active_mask: Tensor,
    min_active: int = 50,
) -> Tensor:
    """Per-day MSE between cross-sectionally z-scored scores and targets.

    Args:
        scores       : (B, N)
        target       : (B, N)
        active_mask  : (B, N) bool
        min_active   : days with fewer than this many active stocks are
                       skipped (returns 0 if no day qualifies).

    Returns scalar tensor.
    """
    B, N = scores.shape
    losses = []
    for b in range(B):
        mask = active_mask[b]
        n = int(mask.sum().item())
        if n < min_active:
            continue
        s = scores[b][mask]
        t = target[b][mask]
        if torch.isfinite(s).sum() < min_active:
            continue
        if torch.isfinite(t).sum() < min_active:
            continue
        s = s - s.mean()
        s_std = s.std(unbiased=False).clamp(min=1.0e-8)
        s_z = s / s_std
        t = t - t.mean()
        t_std = t.std(unbiased=False).clamp(min=1.0e-8)
        t_z = t / t_std
        losses.append(((s_z - t_z) ** 2).mean())
    if not losses:
        return scores.sum() * 0.0
    return torch.stack(losses).mean()


__all__ = ["cs_zscored_mse_loss"]
