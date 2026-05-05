"""Correlation-aware attention for R-STAR fold-2 robustness (Proposal A).

Motivation (diagnostics in docs/fold2_*):
  - Fold 2 has the largest correlation-structure shift on every metric
    (Frobenius 15.5 vs 9.2/13.8; spectral 19.6 vs 11.9/12.0).
  - No edge sign-flips; correlations rise uniformly (+0.097 mean Δ).
  - PC1 variance share jumps to 40.2% on fold 2 (vs 27-32% elsewhere).

Mechanism of the fix:
  During the correlation-spike regime, high-correlation neighbors carry
  redundant information (everyone moves together with the common factor).
  We add a per-pair attention-logit penalty proportional to the rolling
  pairwise correlation, so the transformer automatically diffuses
  attention to less-correlated tokens when correlations rise.

  bias[a, p, q] = -alpha * |C_t[idx_a[p], idx_a[q]]|
  where C_t is the causal rolling-W correlation matrix at day t,
  idx_a[0] is the ego ticker of batch element a, and idx_a[1..N] are its
  graph neighbors (per-cluster top-K union). The bias is broadcast
  across the W-day time dimension.

Design constraints satisfied:
  - Strictly causal (correlation uses returns from t-W to t-1).
  - No graph rebuild (graph remains static, only attention weights
    adjust; prior monthly-rebuild failed at fold-2 IC -0.0045).
  - Orthogonal to iter 10's robust loss (no loss change here).
  - Adds zero parameters.
"""
from __future__ import annotations

import copy

import torch
from torch import Tensor, nn


class CorrBiasedEncoderLayer(nn.Module):
    """nn.TransformerEncoderLayer variant that accepts per-batch attn bias.

    Matches PyTorch's default post-norm ordering (x + attn -> LN -> FF ->
    x + FF -> LN). GELU activation and dropout on attention + FF outputs.
    """

    def __init__(self, d_model: int, nhead: int, ff_dim: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.linear1 = nn.Linear(d_model, ff_dim)
        self.linear2 = nn.Linear(ff_dim, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout_ff = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(
        self,
        src: Tensor,
        attn_bias: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        attn_out, _ = self.self_attn(
            src, src, src,
            attn_mask=attn_bias,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        src = self.norm1(src + self.dropout1(attn_out))
        ff = self.linear2(self.dropout_ff(self.activation(self.linear1(src))))
        src = self.norm2(src + self.dropout2(ff))
        return src


class CorrBiasedEncoder(nn.Module):
    def __init__(self, base_layer: CorrBiasedEncoderLayer, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(base_layer) for _ in range(num_layers)]
        )

    def forward(
        self,
        src: Tensor,
        attn_bias: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        for layer in self.layers:
            src = layer(src, attn_bias=attn_bias, key_padding_mask=key_padding_mask)
        return src


def precompute_rolling_corr(
    log_returns: Tensor,
    mask: Tensor,
    window: int,
) -> Tensor:
    """Causal rolling pairwise correlation of log returns across tickers.

    log_returns: [T, N] raw log returns (NaN or zero outside active window)
    mask:        [T, N] bool active mask
    window:      W, rolling correlation length

    Returns corr: [T, N, N] float32 tensor of pairwise correlations.
    corr[t, i, j] uses log_returns[t-window:t, i] and [t-window:t, j]
    masked to days where both tickers are active in the window. Days
    without enough overlap (<3 points) get correlation 0.

    Causality: day t uses window days ENDING AT t-1 (strictly prior).
    """
    T, N = log_returns.shape
    device = log_returns.device
    corr = torch.zeros(T, N, N, device=device, dtype=torch.float32)
    # Replace NaNs with zeros for the math but use mask to zero out
    lr = torch.where(torch.isfinite(log_returns), log_returns, torch.zeros_like(log_returns))
    lr = lr * mask.to(lr.dtype)

    for t in range(window + 1, T):
        win = lr[t - window: t]  # [W, N]  (days t-W .. t-1 inclusive)
        m = mask[t - window: t].to(lr.dtype)  # [W, N]
        # Per-ticker mean and std over the window (only over active days)
        count = m.sum(dim=0).clamp(min=1.0)  # [N]
        mu = (win * m).sum(dim=0) / count  # [N]
        centered = (win - mu.view(1, N)) * m  # zero out inactive
        var = (centered ** 2).sum(dim=0) / count  # [N]
        sd = var.clamp(min=1e-8).sqrt()  # [N]
        # Pairwise overlap count for normalization
        overlap = m.transpose(0, 1) @ m  # [N, N]
        cov = centered.transpose(0, 1) @ centered  # [N, N]
        cov = cov / overlap.clamp(min=1.0)
        denom = sd.view(N, 1) * sd.view(1, N)
        c = cov / denom.clamp(min=1e-8)
        # Zero out pairs with insufficient overlap (< 3 common active days)
        c = torch.where(overlap >= 3.0, c, torch.zeros_like(c))
        corr[t] = c
    return corr


def build_attn_bias(
    corr_full: Tensor,
    ego_idx: Tensor,
    neighbor_idx: Tensor,
    W: int,
    alpha: float,
    memory_prefix_len: int,
    suffix_len: int,
) -> Tensor:
    """Build per-batch attention bias for the transformer.

    corr_full:     [N, N] day-t correlation matrix (on-device)
    ego_idx:       [A] active ticker indices
    neighbor_idx:  [A, N_nbr] top-N neighbors per active ticker (-1 = missing)
    W:             temporal window
    alpha:         bias strength (nonneg)
    memory_prefix_len: number of memory/summary tokens prepended (0 or 1 or M)
    suffix_len:    number of regime/summary tokens appended (0 or 1)

    Returns bias: [A, S, S] float32 additive attn bias where
    S = memory_prefix_len + (N_nbr+1)*W + suffix_len.
    Prefix and suffix positions have 0 bias.
    """
    A = ego_idx.shape[0]
    N_nbr = neighbor_idx.shape[1]
    NP1 = N_nbr + 1
    device = corr_full.device

    # Build [A, NP1] position->ticker mapping (ego first, neighbors after).
    # Replace -1 neighbors with their ego index (bias then becomes |C[ego,ego]|=1,
    # but these slots are padding-masked in key_padding_mask so the value is
    # never read; use ego to avoid negative index errors).
    nbr_safe = torch.where(neighbor_idx >= 0, neighbor_idx, ego_idx.view(A, 1))
    pos_to_ticker = torch.cat([ego_idx.view(A, 1), nbr_safe], dim=1)  # [A, NP1]

    # Gather pairwise correlation submatrix: [A, NP1, NP1]
    # corr_full[pos_to_ticker[a, p], pos_to_ticker[a, q]]
    c_sub = corr_full[pos_to_ticker.unsqueeze(2), pos_to_ticker.unsqueeze(1)]

    # Take absolute value (both positive and negative strong correlations are
    # redundant with respect to ranking information).
    c_sub = c_sub.abs()

    # Expand across time: bias_patch[a, p*W+w, q*W+w'] = -alpha * c_sub[a, p, q]
    # Use broadcasting: repeat_interleave on both axes by W.
    bias_patch = c_sub.repeat_interleave(W, dim=1).repeat_interleave(W, dim=2)
    bias_patch = -alpha * bias_patch  # [A, NP1*W, NP1*W]

    # Pad with zeros for memory prefix and regime/summary suffix tokens.
    S_patch = NP1 * W
    S = memory_prefix_len + S_patch + suffix_len
    bias = torch.zeros(A, S, S, device=device, dtype=bias_patch.dtype)
    s0 = memory_prefix_len
    s1 = s0 + S_patch
    bias[:, s0:s1, s0:s1] = bias_patch
    return bias


def expand_bias_for_heads(bias: Tensor, num_heads: int) -> Tensor:
    """Convert [A, S, S] bias to [A*num_heads, S, S] for nn.MultiheadAttention."""
    return bias.repeat_interleave(num_heads, dim=0)


__all__ = [
    "CorrBiasedEncoderLayer",
    "CorrBiasedEncoder",
    "precompute_rolling_corr",
    "build_attn_bias",
    "expand_bias_for_heads",
]
