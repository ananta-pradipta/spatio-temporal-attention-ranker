"""Cleaned vendored PatchTST modules (ICLR 2023).

Adapted from https://github.com/yuqinie98/PatchTST under the Apache-2.0
license. The upstream module hierarchy spans three files
(PatchTST_backbone.py, PatchTST_layers.py, RevIN.py); we consolidate the
classes the adapter needs into this single file and strip:
  - the forecasting head (``Flatten_Head``) which projects to
    ``target_window`` and is replaced by a per-channel pooled scalar
    score in ``PatchTSTAdapter``,
  - the optional series decomposition (``moving_avg``, ``series_decomp``
    and the dual-backbone path in ``Model``),
  - all positional-encoding variants we do not use (Coord1d, Coord2d,
    sincos); we keep only learned ``zeros`` since that is the upstream
    default and works well with short lookbacks,
  - the upstream training loop, dataloaders, and Qlib glue,
  - the ``pretrain_head`` branch (we are not running masked
    pretraining).

Sharp-edge fixes vs the upstream source:
  - The upstream ``PatchTST_backbone.forward`` ends with a
    forecasting-shape denorm step. We expose the post-encoder tensor
    ``[bs, nvars, d_model, patch_num]`` directly so the adapter can
    apply its own pooled scalar head with no shape acrobatics.
  - We remove the upstream ``np.inf`` masked_fill in
    ``_ScaledDotProductAttention``; for our cross-sectional path we do
    not use ``key_padding_mask`` or ``attn_mask`` so the codepath is
    inert, but ``-np.inf`` is fp16-unsafe and keeping it would break
    the ``torch.amp.autocast`` path. We replace it with
    ``torch.finfo(scores.dtype).min`` so the ops are fp16-safe even if
    a future caller passes a mask.
  - We default ``norm='LayerNorm'`` (the upstream default is
    ``'BatchNorm'``); BatchNorm1d behaves badly when the per-day batch
    is small (some days have ~150 active stocks but the inner reshape
    yields ``bs*nvars`` ~ 150*22 = 3300 tokens, fine for BN, but
    LayerNorm is more robust to per-day batch variability and matches
    the PyTorch reference Transformer convention).

Public surface used by ``patchtst_adapter.py``:
    PatchTST_backbone   patch + RevIN + transformer encoder, no head
    RevIN               reversible instance normalisation (used inside
                        the backbone but exported for ablation)
    TSTEncoder          (re-exported for inspection)
    TSTEncoderLayer     (re-exported for inspection)
    TSTiEncoder         (re-exported for inspection)
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ============================================================================
# Utilities (adapted from PatchTST_layers.py).
# ============================================================================


class Transpose(nn.Module):
    """Transpose two dims as an nn.Module for use inside nn.Sequential."""

    def __init__(self, *dims: int, contiguous: bool = False):
        super().__init__()
        self.dims = dims
        self.contiguous = contiguous

    def forward(self, x: Tensor) -> Tensor:
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        return x.transpose(*self.dims)


def get_activation_fn(activation):
    if callable(activation):
        return activation()
    if activation.lower() == "relu":
        return nn.ReLU()
    if activation.lower() == "gelu":
        return nn.GELU()
    raise ValueError(f"{activation} is not available; use 'relu' or 'gelu'.")


def positional_encoding(pe: str, learn_pe: bool, q_len: int, d_model: int) -> nn.Parameter:
    """Subset of upstream positional_encoding; we only keep 'zeros' (the
    upstream default), which is a learned tensor initialised uniformly.
    """
    if pe != "zeros":
        raise ValueError(
            f"vendored PatchTST only supports pe='zeros'; got {pe!r}. "
            "Add other variants from upstream PatchTST_layers.py if needed."
        )
    W_pos = torch.empty((q_len, d_model))
    nn.init.uniform_(W_pos, -0.02, 0.02)
    return nn.Parameter(W_pos, requires_grad=learn_pe)


# ============================================================================
# RevIN (adapted from RevIN.py).
# ============================================================================


class RevIN(nn.Module):
    """Reversible Instance Normalization.

    Adapted from https://github.com/ts-kim/RevIN via PatchTST. Per-instance
    statistics (mean, std) are computed at the 'norm' call and reapplied at
    'denorm'. The statistics are detached so they are not part of the
    autograd graph (matches upstream behaviour).
    """

    def __init__(self, num_features: int, eps: float = 1e-5,
                 affine: bool = True, subtract_last: bool = False):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: Tensor, mode: str) -> Tensor:
        if mode == "norm":
            self._get_statistics(x)
            return self._normalize(x)
        if mode == "denorm":
            return self._denormalize(x)
        raise NotImplementedError(f"RevIN mode {mode!r} not supported")

    def _get_statistics(self, x: Tensor) -> None:
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x: Tensor) -> Tensor:
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x: Tensor) -> Tensor:
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x


# ============================================================================
# Attention + Encoder (adapted from PatchTST_backbone.py).
# ============================================================================


class _ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention with optional residual attention from
    a previous layer (Realformer, He et al. 2020) and optional locality
    self-attention (Vision Transformer for Small-Size Datasets).
    """

    def __init__(self, d_model: int, n_heads: int,
                 attn_dropout: float = 0.0, res_attention: bool = False,
                 lsa: bool = False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5),
                                  requires_grad=lsa)
        self.lsa = lsa

    def forward(self, q: Tensor, k: Tensor, v: Tensor,
                prev: Optional[Tensor] = None,
                key_padding_mask: Optional[Tensor] = None,
                attn_mask: Optional[Tensor] = None):
        # q: [bs, n_heads, q_len, d_k]
        # k: [bs, n_heads, d_k, q_len]
        # v: [bs, n_heads, q_len, d_v]
        attn_scores = torch.matmul(q, k) * self.scale
        if prev is not None:
            attn_scores = attn_scores + prev
        if attn_mask is not None:
            # fp16-safe fill; upstream used -np.inf which underflows.
            min_val = torch.finfo(attn_scores.dtype).min
            if attn_mask.dtype == torch.bool:
                attn_scores = attn_scores.masked_fill(attn_mask, min_val)
            else:
                attn_scores = attn_scores + attn_mask
        if key_padding_mask is not None:
            min_val = torch.finfo(attn_scores.dtype).min
            attn_scores = attn_scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), min_val,
            )
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        output = torch.matmul(attn_weights, v)
        if self.res_attention:
            return output, attn_weights, attn_scores
        return output, attn_weights


class _MultiheadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int,
                 d_k: Optional[int] = None, d_v: Optional[int] = None,
                 res_attention: bool = False,
                 attn_dropout: float = 0.0, proj_dropout: float = 0.0,
                 qkv_bias: bool = True, lsa: bool = False):
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v
        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)
        self.res_attention = res_attention
        self.sdp_attn = _ScaledDotProductAttention(
            d_model, n_heads, attn_dropout=attn_dropout,
            res_attention=res_attention, lsa=lsa,
        )
        self.to_out = nn.Sequential(
            nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout),
        )

    def forward(self, Q: Tensor, K: Optional[Tensor] = None,
                V: Optional[Tensor] = None, prev: Optional[Tensor] = None,
                key_padding_mask: Optional[Tensor] = None,
                attn_mask: Optional[Tensor] = None):
        bs = Q.size(0)
        if K is None:
            K = Q
        if V is None:
            V = Q
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0, 2, 3, 1)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1, 2)
        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(
                q_s, k_s, v_s, prev=prev,
                key_padding_mask=key_padding_mask, attn_mask=attn_mask,
            )
        else:
            output, attn_weights = self.sdp_attn(
                q_s, k_s, v_s,
                key_padding_mask=key_padding_mask, attn_mask=attn_mask,
            )
        output = output.transpose(1, 2).contiguous().view(
            bs, -1, self.n_heads * self.d_v,
        )
        output = self.to_out(output)
        if self.res_attention:
            return output, attn_weights, attn_scores
        return output, attn_weights


class TSTEncoderLayer(nn.Module):
    def __init__(self, q_len: int, d_model: int, n_heads: int,
                 d_k: Optional[int] = None, d_v: Optional[int] = None,
                 d_ff: int = 256, store_attn: bool = False,
                 norm: str = "LayerNorm",
                 attn_dropout: float = 0.0, dropout: float = 0.0,
                 bias: bool = True, activation: str = "gelu",
                 res_attention: bool = False, pre_norm: bool = False):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v
        self.res_attention = res_attention
        self.self_attn = _MultiheadAttention(
            d_model, n_heads, d_k, d_v,
            attn_dropout=attn_dropout, proj_dropout=dropout,
            res_attention=res_attention,
        )
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(
                Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2),
            )
        else:
            self.norm_attn = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=bias),
            get_activation_fn(activation),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model, bias=bias),
        )
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(
                Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2),
            )
        else:
            self.norm_ffn = nn.LayerNorm(d_model)
        self.pre_norm = pre_norm
        self.store_attn = store_attn

    def forward(self, src: Tensor, prev: Optional[Tensor] = None,
                key_padding_mask: Optional[Tensor] = None,
                attn_mask: Optional[Tensor] = None):
        if self.pre_norm:
            src = self.norm_attn(src)
        if self.res_attention:
            src2, attn, scores = self.self_attn(
                src, src, src, prev,
                key_padding_mask=key_padding_mask, attn_mask=attn_mask,
            )
        else:
            src2, attn = self.self_attn(
                src, src, src,
                key_padding_mask=key_padding_mask, attn_mask=attn_mask,
            )
        if self.store_attn:
            self.attn = attn
        src = src + self.dropout_attn(src2)
        if not self.pre_norm:
            src = self.norm_attn(src)
        if self.pre_norm:
            src = self.norm_ffn(src)
        src2 = self.ff(src)
        src = src + self.dropout_ffn(src2)
        if not self.pre_norm:
            src = self.norm_ffn(src)
        if self.res_attention:
            return src, scores
        return src


class TSTEncoder(nn.Module):
    def __init__(self, q_len: int, d_model: int, n_heads: int,
                 d_k: Optional[int] = None, d_v: Optional[int] = None,
                 d_ff: int = 256,
                 norm: str = "LayerNorm",
                 attn_dropout: float = 0.0, dropout: float = 0.0,
                 activation: str = "gelu",
                 res_attention: bool = False, n_layers: int = 1,
                 pre_norm: bool = False, store_attn: bool = False):
        super().__init__()
        self.layers = nn.ModuleList([
            TSTEncoderLayer(
                q_len, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff,
                norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                activation=activation, res_attention=res_attention,
                pre_norm=pre_norm, store_attn=store_attn,
            )
            for _ in range(n_layers)
        ])
        self.res_attention = res_attention

    def forward(self, src: Tensor,
                key_padding_mask: Optional[Tensor] = None,
                attn_mask: Optional[Tensor] = None) -> Tensor:
        output = src
        scores: Optional[Tensor] = None
        if self.res_attention:
            for mod in self.layers:
                output, scores = mod(
                    output, prev=scores,
                    key_padding_mask=key_padding_mask, attn_mask=attn_mask,
                )
            return output
        for mod in self.layers:
            output = mod(
                output, key_padding_mask=key_padding_mask, attn_mask=attn_mask,
            )
        return output


class TSTiEncoder(nn.Module):
    """Channel-independent input encoder (the "i" in TSTiEncoder).

    Each channel (panel feature) is patch-embedded and passed through the
    transformer encoder independently; the bs and nvars dimensions are
    flattened together for the encoder pass.
    """

    def __init__(self, c_in: int, patch_num: int, patch_len: int,
                 max_seq_len: int = 1024,
                 n_layers: int = 3, d_model: int = 128, n_heads: int = 16,
                 d_k: Optional[int] = None, d_v: Optional[int] = None,
                 d_ff: int = 256,
                 norm: str = "LayerNorm",
                 attn_dropout: float = 0.0, dropout: float = 0.0,
                 act: str = "gelu", store_attn: bool = False,
                 res_attention: bool = True, pre_norm: bool = False,
                 pe: str = "zeros", learn_pe: bool = True,
                 verbose: bool = False):
        super().__init__()
        self.patch_num = patch_num
        self.patch_len = patch_len
        q_len = patch_num
        self.W_P = nn.Linear(patch_len, d_model)
        self.seq_len = q_len
        self.W_pos = positional_encoding(pe, learn_pe, q_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.encoder = TSTEncoder(
            q_len, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
            attn_dropout=attn_dropout, dropout=dropout,
            pre_norm=pre_norm, activation=act, res_attention=res_attention,
            n_layers=n_layers, store_attn=store_attn,
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: [bs, nvars, patch_len, patch_num]
        n_vars = x.shape[1]
        x = x.permute(0, 1, 3, 2)               # [bs, nvars, patch_num, patch_len]
        x = self.W_P(x)                         # [bs, nvars, patch_num, d_model]
        u = torch.reshape(
            x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        )                                       # [bs*nvars, patch_num, d_model]
        u = self.dropout(u + self.W_pos)
        z = self.encoder(u)                     # [bs*nvars, patch_num, d_model]
        z = torch.reshape(z, (-1, n_vars, z.shape[-2], z.shape[-1]))
        z = z.permute(0, 1, 3, 2)               # [bs, nvars, d_model, patch_num]
        return z


# ============================================================================
# Backbone (adapted from PatchTST_backbone.py, head removed).
# ============================================================================


class PatchTST_backbone(nn.Module):
    """PatchTST encoder without the forecasting head.

    Forward returns the post-encoder tensor of shape
    ``[bs, nvars, d_model, patch_num]``. The adapter is responsible for
    pooling this into a per-channel scalar and reducing across channels
    to produce one score per (day, ticker).

    Defaults match the ICLR 2023 paper's reported architecture for the
    "small" config (ETT/Weather): 3 encoder layers, d_model=128, 16
    heads (we use 8 to match adapter defaults), d_ff=256, RevIN ON with
    learnable affine. The patch_len/stride are set by the caller to
    accommodate the lookback length.
    """

    def __init__(self, c_in: int, context_window: int,
                 patch_len: int, stride: int,
                 max_seq_len: int = 1024,
                 n_layers: int = 3, d_model: int = 128, n_heads: int = 16,
                 d_k: Optional[int] = None, d_v: Optional[int] = None,
                 d_ff: int = 256,
                 norm: str = "LayerNorm",
                 attn_dropout: float = 0.0, dropout: float = 0.0,
                 act: str = "gelu",
                 res_attention: bool = True, pre_norm: bool = False,
                 store_attn: bool = False,
                 pe: str = "zeros", learn_pe: bool = True,
                 padding_patch: Optional[str] = None,
                 revin: bool = True, affine: bool = True,
                 subtract_last: bool = False, verbose: bool = False):
        super().__init__()
        self.revin = revin
        if self.revin:
            self.revin_layer = RevIN(
                c_in, affine=affine, subtract_last=subtract_last,
            )
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        patch_num = int((context_window - patch_len) / stride + 1)
        if padding_patch == "end":
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            patch_num += 1
        self.patch_num = patch_num
        self.d_model = d_model
        self.n_vars = c_in
        self.backbone = TSTiEncoder(
            c_in, patch_num=patch_num, patch_len=patch_len,
            max_seq_len=max_seq_len,
            n_layers=n_layers, d_model=d_model, n_heads=n_heads,
            d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
            attn_dropout=attn_dropout, dropout=dropout, act=act,
            res_attention=res_attention, pre_norm=pre_norm,
            store_attn=store_attn,
            pe=pe, learn_pe=learn_pe, verbose=verbose,
        )

    def forward(self, z: Tensor) -> Tensor:
        # z: [bs, nvars, seq_len]
        if self.revin:
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, "norm")
            z = z.permute(0, 2, 1)
        if self.padding_patch == "end":
            z = self.padding_patch_layer(z)
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # z: [bs, nvars, patch_num, patch_len]
        z = z.permute(0, 1, 3, 2)
        # z: [bs, nvars, patch_len, patch_num]
        z = self.backbone(z)
        # z: [bs, nvars, d_model, patch_num] -- head omitted
        return z


__all__ = [
    "PatchTST_backbone",
    "RevIN",
    "TSTEncoder",
    "TSTEncoderLayer",
    "TSTiEncoder",
]
