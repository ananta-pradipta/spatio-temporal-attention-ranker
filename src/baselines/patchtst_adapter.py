"""Adapter around PatchTST (Nie et al., ICLR 2023) for our biotech panel.

PatchTST = patch-tokenised channel-independent transformer for time
series. Architecture:
  (a) RevIN per-instance reversible normalisation across the lookback.
  (b) Patch embedding: each channel's lookback is unfolded into
      ``patch_num`` patches of length ``patch_len`` with ``stride``,
      then linearly projected to ``d_model``.
  (c) Channel-independent transformer encoder (3 layers by default,
      shared weights across channels via reshape).
  (d) Original ``Flatten_Head`` (from upstream PatchTST_backbone.py)
      flattens the per-channel ``[d_model, patch_num]`` slab and
      projects to ``target_window`` per channel.

Our adaptation for cross-sectional ranking:
  - We keep the original ``Flatten_Head`` with ``target_window=1`` so
    each of the ``F`` channels produces one scalar (matches the
    paper's load-bearing readout exactly, only with horizon 1).
  - We add a learnable cross-channel mixer ``nn.Linear(F, 1)`` at the
    very end to aggregate the ``F`` per-channel scalars into a single
    score per ticker. This replaces the previous
    ``mean-across-channels`` aggregation, which was effectively
    ``Linear(F, 1)`` with all weights frozen at ``1/F`` and destroyed
    cross-feature interactions before the loss could see them.
  - Input per active day: ``x_window`` of shape (N_active, T=20, F=22)
    (matches FactorVAE / MASTER / StockMixer conventions).
  - patch_len=4, stride=2 (so 20 -> 9 patches), matching the paper's
    short-lookback recommendation.
  - d_model=128, 8 heads, d_ff=256, 3 encoder layers (paper defaults
    for the small config; we drop n_heads from 16 to 8 because 16 with
    d_model=128 gives only 8 dims/head which underfits in our
    cross-sectional regime, while 8 heads gives 16 dims/head matching
    every other transformer baseline we run).
  - RevIN ON with learnable affine (paper's strong default for
    return-style series).

The adapter mirrors ``master_adapter.MASTERAdapter`` in shape so the
trainer is a near-clone of ``train_master_v2.py``.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.baselines.vendored.patchtst import PatchTST_backbone


@dataclass
class PatchTSTHyperparams:
    """PatchTST architecture knobs.

    Defaults: ICLR 2023 small configuration with patch_len/stride sized
    for our T=20 lookback. Hidden dim matches the paper; head count is
    adjusted from the paper's 16 down to 8 so each head has 16 dims (a
    healthier ratio for our 244-ticker biotech universe than the 8
    dims/head the paper used on long-horizon forecasting).
    """
    d_feat: int = 22                 # F: number of channels (panel features)
    context_window: int = 20         # T: lookback length
    patch_len: int = 4               # P: patch length (paper rec for short T)
    stride: int = 2                  # S: stride between patches; 20 -> 9 patches
    d_model: int = 128               # transformer hidden width
    n_heads: int = 8                 # multi-head attention heads
    d_ff: int = 256                  # feed-forward width
    n_layers: int = 3                # transformer encoder layers
    dropout: float = 0.1             # residual + ffn dropout
    attn_dropout: float = 0.0        # attention-weights dropout
    head_dropout: float = 0.1        # dropout inside Flatten_Head
    revin: bool = False              # reversible instance normalisation
                                     # (disabled for cross-sectional ranking:
                                     # RevIN's per-(ticker, lookback) zero-mean
                                     # / unit-std normalisation strips per-ticker
                                     # level information that the cross-sectional
                                     # head needs)
    affine: bool = True              # learnable RevIN affine params
    subtract_last: bool = False      # paper's standard mode
    norm: str = "LayerNorm"          # 'LayerNorm' or 'BatchNorm'
    res_attention: bool = True       # Realformer-style residual attention
    pre_norm: bool = False           # paper uses post-norm
    pe: str = "zeros"                # learned positional encoding
    learn_pe: bool = True
    target_window: int = 1           # per-channel scalar (cross-sectional ranking)
    individual_head: bool = False    # shared head across channels (paper default)


class _FlattenHead(nn.Module):
    """Faithful port of PatchTST's ``Flatten_Head``.

    Source: ``PatchTST_supervised/layers/PatchTST_backbone.py`` in
    https://github.com/yuqinie98/PatchTST (Apache-2.0). For each of
    ``n_vars`` channels, flatten the post-encoder slab
    ``[d_model, patch_num] -> [d_model * patch_num]`` and project to
    ``target_window`` via a linear layer (shared across channels by
    default; per-channel if ``individual=True``).

    For our cross-sectional ranking task we set ``target_window=1`` so
    each channel produces one scalar per ticker per day, matching the
    paper's readout shape exactly with the smallest possible horizon.
    """

    def __init__(self, individual: bool, n_vars: int, nf: int,
                 target_window: int, head_dropout: float = 0.0):
        super().__init__()
        self.individual = individual
        self.n_vars = n_vars
        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for _ in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [bs, nvars, d_model, patch_num]
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:, i, :, :])      # [bs, d_model*patch_num]
                z = self.linears[i](z)                   # [bs, target_window]
                z = self.dropouts[i](z)
                x_out.append(z)
            return torch.stack(x_out, dim=1)             # [bs, nvars, target_window]
        x = self.flatten(x)                              # [bs, nvars, d_model*patch_num]
        x = self.linear(x)                               # [bs, nvars, target_window]
        x = self.dropout(x)
        return x


class PatchTSTAdapter(nn.Module):
    """PatchTST wrapped for our (N_active, T, F) panel format.

    Public surface used by ``train_patchtst_v2.py``:
        forward(x_window) -> (N_active,) score per active ticker.

    There is no auxiliary loss (PatchTST is a pure regression model);
    the trainer applies ``cs_mse_loss`` on the scalar score, mirroring
    MASTER, StockMixer, and DySTAGE.

    Internals:
      x_window: (N, T, F)
        permute -> (N, F, T)                        [bs, nvars, seq_len]
        backbone -> (N, F, d_model, patch_num)
        Flatten_Head(target_window=1) -> (N, F, 1)  per-paper readout
        squeeze -> (N, F)
        cross-channel mixer Linear(F, 1) -> (N, 1)  learnable replacement
                                                    for the old mean
        squeeze -> (N,)
    """

    def __init__(self, hp: PatchTSTHyperparams):
        super().__init__()
        self.hp = hp
        self.backbone = PatchTST_backbone(
            c_in=hp.d_feat,
            context_window=hp.context_window,
            patch_len=hp.patch_len,
            stride=hp.stride,
            n_layers=hp.n_layers,
            d_model=hp.d_model,
            n_heads=hp.n_heads,
            d_ff=hp.d_ff,
            norm=hp.norm,
            attn_dropout=hp.attn_dropout,
            dropout=hp.dropout,
            res_attention=hp.res_attention,
            pre_norm=hp.pre_norm,
            pe=hp.pe,
            learn_pe=hp.learn_pe,
            revin=hp.revin,
            affine=hp.affine,
            subtract_last=hp.subtract_last,
        )
        # Original PatchTST flatten head with target_window=1: each
        # channel's [d_model, patch_num] slab is flattened and projected
        # to one scalar.
        nf = hp.d_model * self.backbone.patch_num
        self.flatten_head = _FlattenHead(
            individual=hp.individual_head,
            n_vars=hp.d_feat,
            nf=nf,
            target_window=hp.target_window,
            head_dropout=hp.head_dropout,
        )
        # Learnable cross-channel mixer. Replaces the previous
        # mean-across-channels reduction (which was equivalent to a
        # Linear(F, 1) with weights frozen at 1/F).
        self.channel_mixer = nn.Linear(hp.d_feat, 1)

    def forward(self, x_window: torch.Tensor) -> torch.Tensor:
        """Score per active ticker.

        Args:
            x_window: (N_active, T, F).

        Returns:
            y_hat: (N_active,) raw scalar scores.
        """
        # PatchTST_backbone expects [bs, nvars, seq_len].
        z = x_window.permute(0, 2, 1).contiguous()        # (N, F, T)
        z = self.backbone(z)                              # (N, F, d_model, patch_num)
        z = self.flatten_head(z)                          # (N, F, target_window=1)
        z = z.squeeze(-1)                                 # (N, F)
        y_hat = self.channel_mixer(z).squeeze(-1)         # (N,)
        return y_hat


__all__ = ["PatchTSTAdapter", "PatchTSTHyperparams"]
