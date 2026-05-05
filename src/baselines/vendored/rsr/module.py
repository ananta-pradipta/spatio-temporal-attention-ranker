"""PyTorch port of the RSR architecture (Feng et al., TOIS 2019).

Two modules:

    StockLSTM
        Per-ticker LSTM. Input ``(B, T, F)``; output ``(B, hidden)``
        from the final timestep.

    TemporalGraphAttention
        For each query node i with embedding e_i, aggregate neighbour
        embeddings e_j weighted by attention scores alpha_ij computed
        from a learned bilinear scoring rule and gated by the static
        adjacency mask. This implements the "Explicit relation rank"
        variant the paper reports as the strongest configuration.

Original paper formulation (with our notation):

    s_ij     = LeakyReLU( w^T [W e_i ; W e_j] )
    alpha_ij = softmax_j( s_ij ) over neighbours j with A[i,j] = 1
    g_i      = sum_j alpha_ij * (W_v e_j)

Implementation notes:

    - We softmax over all j with A[i,j] = 1; nodes whose mask is 0
      are pushed to -inf before the softmax. Self-loop is excluded by
      the build_rsr_relation_graph script.
    - We add identity self-aggregation via a residual concat in the
      adapter (see ``rsr_adapter.RSRAdapter``), matching the paper's
      practice of using ``[e_i ; g_i]`` as the head input.
    - We use a single-head attention; the paper does not stack multiple
      heads.

The original repo uses TensorFlow 1.x and a custom ``leaky_relu`` slope
of 0.2; we keep that here.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class StockLSTM(nn.Module):
    """Per-ticker LSTM encoder.

    Args:
        d_feat: number of input features per timestep (F).
        hidden_size: LSTM hidden width (paper default 64).
        num_layers: LSTM depth (paper default 1).
        dropout: dropout between LSTM layers (0 if num_layers == 1).
    """

    def __init__(
        self,
        d_feat: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, F) per-ticker lookback windows.
        Returns:
            (B, hidden_size) final-step hidden state.
        """
        out, _ = self.lstm(x)
        return out[:, -1, :]


class TemporalGraphAttention(nn.Module):
    """Static-graph relation-aware attention over per-ticker embeddings.

    Args:
        hidden_size: dimensionality of the per-ticker embedding e_i.
        attn_hidden: scoring projection width. Paper uses the same
            hidden_size; we expose it for flexibility.
        leaky_slope: LeakyReLU slope used in the additive scoring.
            Paper uses 0.2 (TF1 default).
    """

    def __init__(
        self,
        hidden_size: int,
        attn_hidden: int | None = None,
        leaky_slope: float = 0.2,
    ) -> None:
        super().__init__()
        d_attn = attn_hidden if attn_hidden is not None else hidden_size
        # W in the paper. Project both i and j embeddings.
        self.proj_q = nn.Linear(hidden_size, d_attn, bias=False)
        self.proj_k = nn.Linear(hidden_size, d_attn, bias=False)
        # w in the paper: 1-d attention vector applied to [Wq e_i ; Wk e_j].
        self.attn_vec = nn.Linear(2 * d_attn, 1, bias=False)
        # Value projection W_v.
        self.proj_v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.leaky_slope = leaky_slope

    def forward(
        self,
        e: torch.Tensor,
        adj: torch.Tensor,
    ) -> torch.Tensor:
        """Aggregate neighbour embeddings under the static adjacency.

        Args:
            e:   (N, H) per-ticker LSTM embedding for the active set.
            adj: (N, N) {0, 1} float tensor; ``adj[i, j] = 1`` iff j is
                 a neighbour of i. Should already be restricted to the
                 active subset and self-loop-free.

        Returns:
            g: (N, H) neighbour-aggregated embedding for each ticker.
        """
        N = e.size(0)
        H = e.size(1)
        if N == 0:
            return e

        q = self.proj_q(e)        # (N, d_attn)
        k = self.proj_k(e)        # (N, d_attn)
        v = self.proj_v(e)        # (N, H)

        # Pairwise additive scores: (N, N, 2 * d_attn) -> (N, N).
        # Memory check: with N ~ 200 and d_attn = 64 this is ~6 MB; fine.
        q_e = q.unsqueeze(1).expand(N, N, q.size(1))
        k_e = k.unsqueeze(0).expand(N, N, k.size(1))
        s = self.attn_vec(torch.cat([q_e, k_e], dim=-1)).squeeze(-1)
        s = F.leaky_relu(s, negative_slope=self.leaky_slope)

        # Mask out non-edges before softmax over j.
        mask = adj > 0
        s = s.masked_fill(~mask, float("-inf"))

        # If a row has zero neighbours, softmax of all -inf is NaN; route
        # those rows to a uniform-zero attention by replacing the row with
        # zeros explicitly.
        has_neigh = mask.any(dim=-1)             # (N,)
        if not has_neigh.all():
            # Substitute a single self attention so the softmax is finite.
            self_idx = torch.arange(N, device=e.device)
            s[~has_neigh, :] = float("-inf")
            s[~has_neigh, self_idx[~has_neigh]] = 0.0

        alpha = F.softmax(s, dim=-1)              # (N, N)

        # Aggregate.
        g = alpha @ v                             # (N, H)
        # For genuinely-isolated rows the aggregation collapses to v_i,
        # which is fine: the adapter still concatenates the raw e_i, so
        # information is preserved.
        return g


__all__ = ["StockLSTM", "TemporalGraphAttention"]
