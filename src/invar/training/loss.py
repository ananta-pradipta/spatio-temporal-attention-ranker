"""InVAR hybrid loss.

Components and weights (per spec section "Loss"):

  - huber          weight 1.0   on (y_hat, y_cs)
  - listwise IC    weight 0.5   listwise_lambdarank_ic surrogate; we ship
                                a differentiable Pearson-IC variant
  - pairwise       weight 0.3   RSR-style margin loss on within-day
                                ticker pairs
  - regime CE      weight 0.1   on (regime_logits, GMM label)
  - vol MSE        weight 0.1   on (vol_hat, fwd_vol_20d)
  - entropy reg    weight 0.01  on the regime cross-attention weights
                                (negative entropy added to the loss so
                                a higher-entropy attention is preferred)
  - sinkhorn       weight 0.05  on retrieval-bank usage frequency

Listwise choice: we use a differentiable Pearson-IC surrogate (one minus
the within-day Pearson correlation between y_hat and y_cs), rather than
the original LambdaRank-NDCG formulation. The IC surrogate optimises
the headline metric directly and has stable gradients on the small
N_t cross-sections (N approximately 400 to 500). LambdaRank's pairwise
weights would also work here; the choice is documented per spec.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
import torch.nn.functional as F


@dataclass
class LossWeights:
    """Loss component weights.

    Defaults are ranking-aware: Huber regression on z-scored returns,
    listwise IC surrogate, and pairwise margin (the three "primary"
    ranking objectives). Auxiliary heads (regime_ce, vol_mse, entropy,
    sinkhorn) default to zero and must be opted into via
    ``loss_weights_for("full")``.

    The 2026-05-07 audit caught a regression where the v3 commit
    incorrectly defaulted listwise and pairwise to zero, which silently
    neutered any baseline call that used the bare LossWeights() default
    (MASTER, post-commit StockMixer F2 seeds 45, 46). Defaults are now
    restored.
    """

    huber: float = 1.0
    listwise: float = 0.5
    pairwise: float = 0.3
    regime_ce: float = 0.0
    vol_mse: float = 0.0
    entropy: float = 0.0
    sinkhorn: float = 0.0


def loss_weights_for(config: str) -> LossWeights:
    """Return preset LossWeights for ``config in {'minimal', 'ranking', 'full'}``.

    minimal: Huber 1.0 only (regression on z-scored returns; not
             ranking-aware). Used to A/B-test the ranking-loss
             contribution; not recommended as a baseline loss.
    ranking: Huber 1.0 + listwise IC 0.5 + pairwise margin 0.3.
             No auxiliary heads (regime_ce, vol_mse, entropy, sinkhorn
             all zero). Default for InVAR architecture comparisons
             against iTransformer.
    full:    v1 / v2 weights (huber 1, listwise 0.5, pairwise 0.3,
             regime_ce 0.1, vol_mse 0.1, entropy 0.01, sinkhorn 0.05).
    """
    if config == "minimal":
        return LossWeights(huber=1.0, listwise=0.0, pairwise=0.0,
                            regime_ce=0.0, vol_mse=0.0, entropy=0.0,
                            sinkhorn=0.0)
    if config == "ranking":
        return LossWeights()  # uses ranking-aware defaults
    if config == "full":
        return LossWeights(
            huber=1.0, listwise=0.5, pairwise=0.3,
            regime_ce=0.1, vol_mse=0.1, entropy=0.01, sinkhorn=0.05,
        )
    raise ValueError(f"unknown loss config: {config!r}")


def huber_loss(y_hat: Tensor, y_cs: Tensor, mask: Tensor,
                delta: float = 1.0) -> Tensor:
    """Standard Huber loss masked to the active subset."""
    if not mask.any():
        return y_hat.sum() * 0.0
    y_hat_a = y_hat[mask]
    y_cs_a = y_cs[mask]
    return F.huber_loss(y_hat_a, y_cs_a, delta=delta, reduction="mean")


def listwise_ic_loss(y_hat: Tensor, y_cs: Tensor, mask: Tensor,
                       eps: float = 1e-8) -> Tensor:
    """Differentiable Pearson-IC surrogate: 1 - corr(y_hat, y_cs).

    Computed on the active subset only. This is the listwise choice
    documented in the module docstring; the pairwise margin loss covers
    the LambdaRank-flavored objective separately.
    """
    if not mask.any():
        return y_hat.sum() * 0.0
    a = y_hat[mask]
    b = y_cs[mask]
    a = a - a.mean()
    b = b - b.mean()
    num = (a * b).sum()
    denom = torch.sqrt((a ** 2).sum() * (b ** 2).sum() + eps)
    rho = num / denom
    return 1.0 - rho


def pairwise_margin_loss(
    y_hat: Tensor, y_cs: Tensor, mask: Tensor,
    margin: float = 0.0, max_pairs: int = 4096,
) -> Tensor:
    """RSR-style pairwise margin loss within the day's active set.

    Pairs are sampled by the magnitude of the target return difference;
    the model is penalised whenever the predicted ordering disagrees
    with the target ordering by more than ``-margin``. Cap the number
    of pairs for tractability on N approximately 500 cross-sections.
    """
    if not mask.any():
        return y_hat.sum() * 0.0
    a_idx = mask.nonzero(as_tuple=True)[0]
    n = a_idx.numel()
    if n < 2:
        return y_hat.sum() * 0.0
    n_pairs = min(max_pairs, n * (n - 1) // 2)
    rng = torch.randint(low=0, high=n, size=(n_pairs * 2,), device=y_hat.device)
    i = a_idx[rng[: n_pairs]]
    j = a_idx[rng[n_pairs:]]
    keep = (i != j)
    if not keep.any():
        return y_hat.sum() * 0.0
    i = i[keep]; j = j[keep]
    diff_target = y_cs[i] - y_cs[j]
    diff_pred = y_hat[i] - y_hat[j]
    sign = torch.sign(diff_target)
    losses = F.relu(margin - sign * diff_pred)
    weight = diff_target.abs()
    return (losses * weight).sum() / (weight.sum() + 1e-8)


def regime_ce_loss(regime_logits: Tensor, regime_label: int) -> Tensor:
    """Cross-entropy of regime_logits against the day's GMM cluster id."""
    target = torch.tensor([regime_label], device=regime_logits.device,
                            dtype=torch.long)
    return F.cross_entropy(regime_logits.unsqueeze(0), target)


def vol_mse_loss(vol_hat: Tensor, vol_target: Tensor, mask: Tensor) -> Tensor:
    """MSE on per-ticker 20-day forward realised vol, masked by has_fwd_vol."""
    if not mask.any():
        return vol_hat.sum() * 0.0
    a = vol_hat[mask]
    b = vol_target[mask]
    return F.mse_loss(a, b, reduction="mean")


def regime_attn_entropy(attn_weights: list[dict] | None) -> Tensor:
    """Negative mean entropy of regime cross-attention weights across blocks.

    A higher-entropy attention distribution is preferred (more diverse
    use of regime tokens), so we return ``-entropy`` and let the trainer
    add it with a positive weight (which becomes a soft pull toward
    high entropy).
    """
    if not attn_weights:
        return torch.zeros((), requires_grad=True)
    entropies = []
    for blk in attn_weights:
        if blk is None:
            continue
        ca = blk.get("ca")
        if ca is None:
            continue
        # ca shape: (batch, num_heads, query_len, key_len) or simpler
        # depending on need_weights setting; flatten over all but last.
        p = ca.float()
        p = p.reshape(-1, p.shape[-1])
        p = p / (p.sum(dim=-1, keepdim=True) + 1e-9)
        ent = -(p * (p + 1e-9).log()).sum(dim=-1).mean()
        entropies.append(ent)
    if not entropies:
        first = next(
            (b["ca"] for b in attn_weights if b is not None and b.get("ca") is not None),
            None,
        )
        device = first.device if first is not None else "cpu"
        return torch.zeros((), device=device, requires_grad=True)
    return -torch.stack(entropies).mean()


def sinkhorn_balance_loss(
    usage_counts: Tensor, target: Tensor | None = None, eps: float = 0.05,
    n_iter: int = 5,
) -> Tensor:
    """Soft Sinkhorn-style balance penalty on bank-entry usage frequency.

    Args:
        usage_counts: ``(bank_size,)`` tensor of (soft) usage scores; we
            convert to a probability distribution and penalise its
            divergence from uniform via a Sinkhorn-regularised objective.
        target: optional target distribution; defaults to uniform.
        eps: Sinkhorn regulariser.
        n_iter: number of Sinkhorn iterations (currently unused; we use a
            cheap KL-to-uniform proxy that the spec calls "Sinkhorn-style".
            The proper OT-balanced formulation is an upgrade target.)
    """
    if usage_counts.numel() == 0:
        return torch.zeros((), requires_grad=True)
    p = usage_counts.float()
    p = p / (p.sum() + 1e-9)
    if target is None:
        target = torch.full_like(p, 1.0 / p.numel())
    return F.kl_div((p + 1e-9).log(), target, reduction="sum")


@dataclass
class LossOutput:
    total: Tensor
    components: dict[str, float]


def hybrid_loss(
    y_hat: Tensor,
    y_cs: Tensor,
    mask: Tensor,
    regime_logits: Tensor,
    regime_label: int,
    vol_hat: Tensor,
    vol_target: Tensor,
    has_vol_mask: Tensor,
    weights: LossWeights | None = None,
    attn_weights: list[dict] | None = None,
    bank_usage_counts: Tensor | None = None,
) -> LossOutput:
    w = weights or LossWeights()
    h = huber_loss(y_hat, y_cs, mask)
    lst = listwise_ic_loss(y_hat, y_cs, mask)
    pw = pairwise_margin_loss(y_hat, y_cs, mask)
    rce = regime_ce_loss(regime_logits, regime_label)
    vmse = vol_mse_loss(vol_hat, vol_target, has_vol_mask)
    ent = regime_attn_entropy(attn_weights)
    snk = (sinkhorn_balance_loss(bank_usage_counts)
            if bank_usage_counts is not None else torch.zeros((), device=h.device))

    total = (
        w.huber * h
        + w.listwise * lst
        + w.pairwise * pw
        + w.regime_ce * rce
        + w.vol_mse * vmse
        + w.entropy * ent
        + w.sinkhorn * snk
    )
    return LossOutput(
        total=total,
        components={
            "huber": float(h.detach().item()),
            "listwise": float(lst.detach().item()),
            "pairwise": float(pw.detach().item()),
            "regime_ce": float(rce.detach().item()),
            "vol_mse": float(vmse.detach().item()),
            "entropy": float(ent.detach().item()) if ent.requires_grad or ent.is_floating_point() else 0.0,
            "sinkhorn": float(snk.detach().item()),
        },
    )


__all__ = [
    "LossWeights", "LossOutput", "loss_weights_for",
    "huber_loss", "listwise_ic_loss", "pairwise_margin_loss",
    "regime_ce_loss", "vol_mse_loss", "regime_attn_entropy",
    "sinkhorn_balance_loss", "hybrid_loss",
]
