"""iTransformer baseline (paper-faithful) for InVAR Phase 3.

Reference:
  Liu et al. (2024). "iTransformer: Inverted Transformers Are Effective for
  Time Series Forecasting." ICLR 2024. https://github.com/thuml/iTransformer

This implementation follows the paper architecture for stock-ranking
benchmarks (where each STOCK is treated as a variate, matching the
paper's Solar / Traffic / Electricity datasets which have one
univariate time series per station). We adapt to our setting where each
ticker has F=26 features over L=60 days by:

  1. RevIN normalisation per (ticker, feature) over the L axis
     (Kim et al. 2022 ICLR). RevIN is the standard normalisation in
     the iTransformer paper for stock data; we keep it.
  2. Plain Linear projection from flattened (L, F) -> d to produce one
     ticker token (the paper's Linear projection from L -> d per variate
     for univariate stations; with multivariate features per ticker we
     project the flattened time-feature vector).
  3. Pre-norm Transformer encoder, 4 layers, d_model=128, 4 heads, FFN
     dim 4*d_model=512, dropout 0.1. Each layer's self-attention runs
     over the N ticker variates.
  4. Linear ranking head replacing the forecasting head per the spec
     ("replace forecasting head with linear ranking head").

This supersedes the prior cross-feature-per-ticker variant (which
treated each FEATURE as a variate and ran attention independently per
ticker) and the prior cross-stock-without-RevIN variant. The paper's
RevIN normalisation is now included.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class ITransformerConfig:
    n_features: int = 26
    lookback: int = 60
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    ffn_hidden: int = 512                  # 4 * d_model per the paper
    dropout: float = 0.10
    head_hidden: int = 64
    revin_eps: float = 1.0e-5
    revin_affine: bool = True


class RevIN(nn.Module):
    """Reversible Instance Normalisation (Kim et al. 2022, ICLR).

    Per-instance normalisation across the lookback axis, with optional
    learnable affine parameters per feature. For ranking we only use
    the forward (normalise) path; no de-normalisation at output.

    Args:
        n_features: F per ticker.
        eps: numerical stability.
        affine: if True, learn per-feature scale (gamma) and shift (beta).
    """

    def __init__(self, n_features: int, eps: float = 1.0e-5,
                 affine: bool = True) -> None:
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.gamma = nn.Parameter(torch.ones(n_features))
            self.beta = nn.Parameter(torch.zeros(n_features))

    def forward(self, x: Tensor) -> Tensor:
        """Normalise x : (N, L, F) along the L axis per (n, f)."""
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True).clamp(min=self.eps)
        x = (x - mean) / std
        if self.affine:
            x = x * self.gamma + self.beta
        return x


class _TickerVariateEmbed(nn.Module):
    """Plain Linear projection of each ticker's flattened panel to one token.

    Each TICKER is a variate (in the iTransformer sense). The (L, F)
    panel history is flattened to (L*F,) and projected to d via a single
    Linear layer (no MLP). This matches the paper's per-variate Linear
    projection for univariate stations adapted to multivariate features.
    """

    def __init__(self, cfg: ITransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        flat_dim = cfg.lookback * cfg.n_features
        self.proj = nn.Linear(flat_dim, cfg.d_model)

    def forward(self, features: Tensor) -> Tensor:
        # features: (N, L, F) -> (N, L*F) -> (N, d)
        N, L, F = features.shape
        return self.proj(features.reshape(N, L * F))


class _ITransformerBlock(nn.Module):
    """Pre-norm Transformer block, attention over the variate axis."""

    def __init__(self, cfg: ITransformerConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.attn = nn.MultiheadAttention(
            cfg.d_model, cfg.n_heads, dropout=cfg.dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.ffn_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.ffn_hidden, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, h: Tensor, key_padding_mask: Tensor | None) -> Tensor:
        # h: (1, N, d); attention runs over N variate tokens.
        h_norm = self.norm1(h)
        a, _ = self.attn(
            h_norm, h_norm, h_norm,
            key_padding_mask=key_padding_mask, need_weights=False,
        )
        h = h + a
        h = h + self.ffn(self.norm2(h))
        return h


class ITransformer(nn.Module):
    """iTransformer with RevIN + plain Linear tokenizer + ranking head."""

    def __init__(self, cfg: ITransformerConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or ITransformerConfig()
        self.revin = RevIN(self.cfg.n_features, eps=self.cfg.revin_eps,
                            affine=self.cfg.revin_affine)
        self.embed = _TickerVariateEmbed(self.cfg)
        self.blocks = nn.ModuleList(
            [_ITransformerBlock(self.cfg) for _ in range(self.cfg.n_layers)]
        )
        self.norm_out = nn.LayerNorm(self.cfg.d_model)
        self.head = nn.Sequential(
            nn.Linear(self.cfg.d_model, self.cfg.head_hidden),
            nn.GELU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(self.cfg.head_hidden, 1),
        )

    def forward(
        self, features: Tensor, macro: Tensor, mask: Tensor,
        return_attn: bool = False,
    ) -> dict[str, Tensor]:
        # features: (N, L, F)
        x = self.revin(features)
        h = self.embed(x)
        h = h.unsqueeze(0)                                  # (1, N, d)
        kpm = (~mask).unsqueeze(0)
        for block in self.blocks:
            h = block(h, key_padding_mask=kpm)
        h = self.norm_out(h)
        y_hat = self.head(h).squeeze(-1).squeeze(0)
        m = mask.float()
        y_hat = y_hat * m
        return {
            "y_hat": y_hat,
            "regime_logits": torch.zeros(8, device=features.device),
            "vol_hat": torch.zeros_like(y_hat),
            "attn_weights": [] if return_attn else None,
        }


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


__all__ = ["ITransformerConfig", "ITransformer", "RevIN", "count_parameters"]
