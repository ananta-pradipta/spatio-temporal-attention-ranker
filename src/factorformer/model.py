"""FactorFormer model: variate-as-token encoder + variational factor bottleneck.

Each foundation contributes one disjoint mechanism:
- iTransformer (Liu et al., 2024): per-variate MLP tokenizer over the
  L-day lookback, then self-attention across the F-axis.
- FactorVAE (Duan et al., 2022): prior network conditioned on
  cross-sectional context, posterior network conditioned on context +
  future-return summary, KL regularization between them, reparameterized
  factor sampling, and stock-specific factor-loading decoders.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn, Tensor


@dataclass
class FactorFormerConfig:
    n_features: int = 26
    lookback: int = 60
    d_model: int = 128
    n_heads: int = 4
    ffn_dim: int = 256
    n_layers: int = 4
    dropout: float = 0.1
    n_factors: int = 8
    tokenizer_hidden: int = 64
    kl_weight: float = 1.0e-3
    log_sigma_min: float = -5.0
    log_sigma_max: float = 2.0


class VariateTokenizer(nn.Module):
    """Per-variate MLP over the L-day lookback. Shared across variates."""

    def __init__(self, lookback: int, d_model: int, hidden: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(lookback, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: (N, L, F) -> (N, F, L) -> (N, F, D)
        return self.proj(x.transpose(1, 2))


class ITransformerBlock(nn.Module):
    """Pre-norm self-attention over the F-axis, then position-wise FFN."""

    def __init__(self, d_model: int, n_heads: int,
                 ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, v: Tensor) -> Tensor:
        h = self.norm1(v)
        a, _ = self.attn(h, h, h, need_weights=False)
        v = v + self.drop(a)
        h = self.norm2(v)
        v = v + self.drop(self.ffn(h))
        return v


class FactorFormer(nn.Module):
    def __init__(self, cfg: FactorFormerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tokenizer = VariateTokenizer(
            cfg.lookback, cfg.d_model, cfg.tokenizer_hidden,
        )
        self.blocks = nn.ModuleList([
            ITransformerBlock(cfg.d_model, cfg.n_heads, cfg.ffn_dim, cfg.dropout)
            for _ in range(cfg.n_layers)
        ])
        # Cross-sectional context = [mean, std] of stock embeddings -> R^{2D}
        self.prior = nn.Sequential(
            nn.Linear(2 * cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, 2 * cfg.n_factors),
        )
        # Posterior fuses per-stock embedding with per-stock future return,
        # then pools cross-sectionally. Mirrors FactorVAE's posterior design,
        # which is allowed to see future per-stock returns at training time.
        self.posterior_per_stock = nn.Sequential(
            nn.Linear(cfg.d_model + 1, cfg.d_model),
            nn.GELU(),
        )
        self.posterior = nn.Sequential(
            nn.Linear(2 * cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, 2 * cfg.n_factors),
        )
        # Stock-specific factor loadings: H_i -> (w_i in R^k, b_i in R)
        self.loadings = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.n_factors + 1),
        )

    def encode(self, x: Tensor) -> Tensor:
        v = self.tokenizer(x)
        for blk in self.blocks:
            v = blk(v)
        return v.mean(dim=1)  # pool over F-axis -> (N, D)

    def _gauss_params(self, raw: Tensor) -> tuple[Tensor, Tensor]:
        mu, log_sigma = raw.chunk(2, dim=-1)
        log_sigma = log_sigma.clamp(self.cfg.log_sigma_min, self.cfg.log_sigma_max)
        return mu, log_sigma

    def forward(self, x: Tensor, y_future: Tensor | None = None) -> dict:
        h_stock = self.encode(x)            # (N, D)
        c_mean = h_stock.mean(dim=0)        # (D,)
        c_std = h_stock.std(dim=0, unbiased=False) + 1.0e-6  # (D,)
        c = torch.cat([c_mean, c_std], dim=0)  # (2D,)

        mu_p, log_sigma_p = self._gauss_params(self.prior(c))

        if self.training and y_future is not None:
            y_aug = y_future.unsqueeze(-1)
            per_stock_post = self.posterior_per_stock(
                torch.cat([h_stock, y_aug], dim=-1),
            )
            cp_mean = per_stock_post.mean(dim=0)
            cp_std = per_stock_post.std(dim=0, unbiased=False) + 1.0e-6
            c_post = torch.cat([cp_mean, cp_std], dim=0)
            mu_q, log_sigma_q = self._gauss_params(self.posterior(c_post))
            eps = torch.randn_like(mu_q)
            z = mu_q + log_sigma_q.exp() * eps
        else:
            mu_q, log_sigma_q = None, None
            z = mu_p

        loadings = self.loadings(h_stock)
        w = loadings[:, : self.cfg.n_factors]   # (N, k)
        b = loadings[:, -1]                     # (N,)
        y_hat = (w * z.unsqueeze(0)).sum(dim=-1) + b  # (N,)

        return dict(
            y_hat=y_hat, mu_p=mu_p, log_sigma_p=log_sigma_p,
            mu_q=mu_q, log_sigma_q=log_sigma_q, z=z,
        )


def factor_kl(mu_q: Tensor, log_sigma_q: Tensor,
              mu_p: Tensor, log_sigma_p: Tensor) -> Tensor:
    """KL(q || p) for diagonal Gaussians, summed over factors."""
    var_q = (2.0 * log_sigma_q).exp()
    var_p = (2.0 * log_sigma_p).exp()
    kl = (log_sigma_p - log_sigma_q
          + (var_q + (mu_q - mu_p) ** 2) / (2.0 * var_p) - 0.5)
    return kl.sum()


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
