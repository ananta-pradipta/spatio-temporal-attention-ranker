"""RegimeXer-iT building blocks: FiLM modulation and invariance gate.

The CCC loss component (C6) was dropped per 2026-05-11 update; all
RegimeXer-iT arms use the existing hybrid loss (Huber + listwise IC +
pairwise margin) from `src.invar.training.loss`. This module therefore
defines only FiLM and the invariance gate.

Design reference: docs/mait_design.md is unrelated; the RegimeXer-iT spec
was provided by the user 2026-05-11 (attachment in Discord). See
docstrings below for the exact mathematical definitions.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn


class FiLMBlock(nn.Module):
    """Feature-wise Linear Modulation over variate tokens.

    Given stock variate tokens H of shape (N, F, d) and a per-stock macro
    context c of shape (N, d), produces gamma and beta of shape (N, F)
    each and returns `gamma.unsqueeze(-1) * H + beta.unsqueeze(-1)`.

    Initialization: gamma weights = 0, gamma bias = 1; beta weights = 0,
    beta bias = 0. So at step 0 the FiLM block is the identity for any
    input. The model only departs from identity once gradients move the
    weights.
    """

    def __init__(self, d_model: int, n_panel: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_panel = n_panel
        self.mlp_gamma = nn.Linear(d_model, n_panel)
        self.mlp_beta = nn.Linear(d_model, n_panel)
        with torch.no_grad():
            nn.init.zeros_(self.mlp_gamma.weight)
            nn.init.ones_(self.mlp_gamma.bias)
            nn.init.zeros_(self.mlp_beta.weight)
            nn.init.zeros_(self.mlp_beta.bias)

    def forward(self, H: Tensor, c: Tensor) -> Tensor:
        """Apply FiLM modulation.

        Args:
            H: (N, F, d) stock variate tokens.
            c: (N, d) per-stock macro context.

        Returns:
            (N, F, d) tokens with feature-wise affine modulation.
        """
        gamma = self.mlp_gamma(c)
        beta = self.mlp_beta(c)
        return gamma.unsqueeze(-1) * H + beta.unsqueeze(-1)


class InvarianceGate(nn.Module):
    """Per-stock invariance gate alpha in [0, 1].

    Computes alpha = sigmoid(MLP(concat([c, abs(c - c_running_mean)]))),
    where c is the per-stock macro context (N, d) and c_running_mean is
    an EMA over training-time batches (decay 0.99), frozen at eval.

    When alpha = 0 the mixed output equals the invariant pathway (no
    macro). When alpha = 1 the mixed output equals the macro-conditioned
    pathway. The L_alpha regularizer in the training loss pushes alpha
    toward zero by default, so the model only opens the gate when the
    rank-IC signal justifies it.
    """

    def __init__(self, d_model: int, hidden: int = 64,
                 ema_decay: float = 0.99) -> None:
        super().__init__()
        self.d_model = d_model
        self.ema_decay = ema_decay
        # Buffer, not Parameter: running mean of c. Updated in `update_running_mean`
        # which is called from the training loop on every forward pass.
        self.register_buffer("c_running_mean", torch.zeros(d_model))
        self.register_buffer("ema_initialized", torch.tensor(0.0))
        self.mlp = nn.Sequential(
            nn.Linear(2 * d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    @torch.no_grad()
    def update_running_mean(self, c_batch: Tensor) -> None:
        """Update EMA of per-stock c over the current training batch.

        Args:
            c_batch: (N, d) macro context for the current day.
        """
        batch_mean = c_batch.mean(dim=0)
        if self.ema_initialized.item() == 0.0:
            self.c_running_mean.copy_(batch_mean.detach())
            self.ema_initialized.fill_(1.0)
        else:
            self.c_running_mean.mul_(self.ema_decay).add_(
                batch_mean.detach() * (1.0 - self.ema_decay),
            )

    def forward(self, c: Tensor) -> Tensor:
        """Compute alpha gate values.

        Args:
            c: (N, d) per-stock macro context.

        Returns:
            (N, 1) alpha values in [0, 1].
        """
        c_centered = c - self.c_running_mean
        gate_input = torch.cat([c, c_centered.abs()], dim=-1)
        return torch.sigmoid(self.mlp(gate_input))
