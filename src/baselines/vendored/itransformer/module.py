"""Vendored iTransformer architecture (Liu et al., ICLR 2024).

Source: https://github.com/thuml/iTransformer (MIT). Specifically
adapted from
    ``model/iTransformer.py``
    ``layers/Embed.py``           (DataEmbedding_inverted only)
    ``layers/SelfAttention_Family.py`` (FullAttention, AttentionLayer)
    ``layers/Transformer_EncDec.py``   (EncoderLayer, Encoder)

We strip the upstream training loop, dataloaders, all non-iTransformer
model variants (iFlashformer, iReformer, etc.), the decoder path, the
optional ``output_attention``/``class_strategy`` plumbing, and the
unused causal/Prob/Flow attentions. We keep only the inverted-transformer
encoder used in the paper's headline experiments.

Reference:
    Liu, Y., Hu, T., Liu, H., Zhou, J., Li, S., Long, M. (2024).
    "iTransformer: Inverted Transformers Are Effective for Time Series
    Forecasting." ICLR 2024.
"""
from __future__ import annotations

from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Inverted token embedding: each variate's lookback -> one token of d_model.
# (Source: layers/Embed.py :: DataEmbedding_inverted)
# ============================================================================


class DataEmbeddingInverted(nn.Module):
    """Inverted token embedding.

    Input  : x of shape ``(B, L, N)`` with ``L`` = lookback length and
             ``N`` = number of variates.
    Output : ``(B, N, d_model)`` -- each variate becomes one token.
    """

    def __init__(self, c_in: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, N) -> (B, N, L) -> (B, N, d_model)
        x = x.permute(0, 2, 1)
        x = self.value_embedding(x)
        return self.dropout(x)


# ============================================================================
# Standard scaled-dot-product self-attention (no causal mask, no Prob).
# (Source: layers/SelfAttention_Family.py :: FullAttention, AttentionLayer)
# ============================================================================


class FullAttention(nn.Module):
    """Vanilla scaled-dot-product attention without causal masking.

    iTransformer uses ``mask_flag=False`` because attention runs across
    variates, not across time, so there is no causal ordering.
    """

    def __init__(self, attention_dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(attention_dropout)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # queries/keys/values: (B, L, H, E)
        _, _, _, E = queries.shape
        scale = 1.0 / sqrt(E)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, float("-inf"))
        a = self.dropout(torch.softmax(scale * scores, dim=-1))
        v = torch.einsum("bhls,bshd->blhd", a, values)
        return v.contiguous()


class AttentionLayer(nn.Module):
    """Multi-head wrapper around an inner attention module."""

    def __init__(self, inner_attention: nn.Module, d_model: int, n_heads: int):
        super().__init__()
        d_keys = d_model // n_heads
        d_values = d_model // n_heads
        self.inner_attention = inner_attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        q = self.query_projection(queries).view(B, L, H, -1)
        k = self.key_projection(keys).view(B, S, H, -1)
        v = self.value_projection(values).view(B, S, H, -1)
        out = self.inner_attention(q, k, v, attn_mask=attn_mask)
        out = out.view(B, L, -1)
        return self.out_projection(out)


# ============================================================================
# Encoder layer + stack.
# (Source: layers/Transformer_EncDec.py :: EncoderLayer, Encoder)
# ============================================================================


class EncoderLayer(nn.Module):
    def __init__(
        self,
        attention: nn.Module,
        d_model: int,
        d_ff: int | None = None,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        new_x = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y)


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer: nn.Module | None = None):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        for layer in self.attn_layers:
            x = layer(x, attn_mask=attn_mask)
        if self.norm is not None:
            x = self.norm(x)
        return x


# ============================================================================
# iTransformer model: variates-as-tokens encoder + per-variate projector.
# (Source: model/iTransformer.py :: Model.forecast)
# ============================================================================


class ITransformerModel(nn.Module):
    """Inverted transformer for cross-variate self-attention.

    Args:
        seq_len: lookback length L (in our project, set to T*F so each
            variate carries the flattened panel features).
        pred_len: per-variate output dimension (``1`` for our scalar
            cross-sectional ranking head).
        d_model: hidden dimension.
        n_heads: number of attention heads (must divide d_model).
        e_layers: number of encoder layers.
        d_ff: feed-forward width.
        dropout: dropout used in embedding, attention, and FFN.
        activation: ``'gelu'`` (default) or ``'relu'``.
        use_norm: paper's non-stationary normalisation across the
            lookback. Off by default for our cross-sectional setup; the
            v2 protocol already standardises features.

    Forward:
        x_enc shape ``(B, L, N)`` where ``N`` is the number of variates.
        Returns ``(B, pred_len, N)``.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int = 1,
        d_model: int = 128,
        n_heads: int = 4,
        e_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
        activation: str = "gelu",
        use_norm: bool = False,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.use_norm = use_norm

        self.enc_embedding = DataEmbeddingInverted(seq_len, d_model, dropout=dropout)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(attention_dropout=dropout),
                        d_model,
                        n_heads,
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
        )
        self.projector = nn.Linear(d_model, pred_len, bias=True)

    def forward(self, x_enc: torch.Tensor) -> torch.Tensor:
        # x_enc: (B, L, N)
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
            )
            x_enc = x_enc / stdev

        # (B, L, N) -> (B, N, d_model)
        enc_out = self.enc_embedding(x_enc)
        # (B, N, d_model) -> (B, N, d_model)
        enc_out = self.encoder(enc_out, attn_mask=None)
        # (B, N, d_model) -> (B, N, pred_len) -> (B, pred_len, N)
        dec_out = self.projector(enc_out).permute(0, 2, 1)

        if self.use_norm:
            dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)

        return dec_out


__all__ = [
    "ITransformerModel",
    "DataEmbeddingInverted",
    "FullAttention",
    "AttentionLayer",
    "EncoderLayer",
    "Encoder",
]
