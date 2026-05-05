"""Proposal C: domain-adversarial regime invariance via gradient reversal.

Motivation:
  All three diagnostic-driven proposals (A uniform, A-gated, B group-relative,
  A+B combined) improve fold-2 mean but regress fold-1 below the bar, because
  they attack fold-2 mechanisms AT INFERENCE time — the learned representation
  itself still encodes regime information that hurts generalisation across
  regime transitions.

  Proposal C attacks this at REPRESENTATION time. It asks the encoder to
  produce hidden states that are predictive for ranking but NOT distinguishable
  by regime. A small auxiliary classifier tries to predict the k-means regime
  cluster from the cross-sectionally-normalised hidden state; a gradient
  reversal layer (GRL, Ganin et al. 2015) flips the gradient so the encoder
  is trained to FOOL the classifier. The resulting representation is regime-
  invariant by construction.

  This directly addresses the mechanism we could not measure causally in our
  hard-threshold gate analysis: rather than detect correlation-structure
  shift, we force the representation not to encode it.

Usage:
  - Pass `dann_lambda_max > 0` to enable. `dann_hidden > 0` sets classifier
    hidden size.
  - GRL scale lambda anneals linearly from 0 to dann_lambda_max over training
    (classic DANN schedule); encoder sees no adversarial pressure at epoch 0.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn


class GradientReversal(torch.autograd.Function):
    """Identity on forward, flipped-and-scaled on backward.

    y = x  (forward)
    dL/dx = -lambda * dL/dy  (backward)
    """

    @staticmethod
    def forward(ctx, x: Tensor, lambd: float) -> Tensor:
        ctx.lambd = lambd
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        return -ctx.lambd * grad_output, None


def grad_reverse(x: Tensor, lambd: float) -> Tensor:
    return GradientReversal.apply(x, lambd)


class RegimeDiscriminator(nn.Module):
    """Small MLP that predicts regime-cluster label from hidden state.

    Architecture: LayerNorm -> Linear(D, h) -> GELU -> Dropout -> Linear(h, K).
    K = number of regime clusters (k-means K in REMTrainConfig).
    """

    def __init__(self, d_model: int, hidden: int, num_classes: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, z: Tensor) -> Tensor:
        return self.net(z)


__all__ = ["GradientReversal", "grad_reverse", "RegimeDiscriminator"]
