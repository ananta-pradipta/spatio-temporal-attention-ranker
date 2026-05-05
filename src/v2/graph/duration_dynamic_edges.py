"""Duration-aware dynamic graph for DOW-epiSTAR v2 (spec Section G).

v2.2 fallback per spec G end ("If full residual graph is expensive,
first implement only A_duration and add it to existing A_corr"):

    score_edge[t, i, j] = w_corr[t] * A_corr[t, i, j]
                        + w_duration[t] * A_duration[t, i, j]

with the gate weights drawn from a softmax over a small MLP that
takes ``macro_gate_state[t]`` as input. A_corr is the existing
reliability-shrunk rolling correlation matrix; A_duration is the
cosine similarity between per-(day, ticker) duration_exposure
embeddings.

The full three-source variant (A_corr + A_resid + A_duration) is
deferred to v2.3 because A_resid requires per-ticker residual-return
computation against XBI, QQQ, rate_shock, credit_shock. The fallback
captures the highest-leverage piece of the spec (the duration-aware
graph) without the residual-return engineering.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn


@dataclass
class DurationGraphConfig:
    """Hyperparameters for the duration-aware dynamic graph."""

    top_k: int = 8
    min_overlap_absolute: int = 5
    # v2.3 patch E: default weights now favour A_corr more strongly
    # (0.8 / 0.2) so the gate must earn the right to lean on the
    # duration graph. Spec section E suggested log(0.8/0.2).
    init_w_corr: float = 0.8
    init_w_duration: float = 0.2
    # v2.3: pick which duration-similarity source to use.
    #   "deterministic_features": cosine over fixed 10-dim hand-engineered
    #     features (default; auditable, doesn't depend on learned encoder).
    #   "learned_encoder": cosine over DurationExposureEncoder output
    #     (the v2.2 default; can be unstable before the encoder warms up).
    duration_graph_source: str = "deterministic_features"


# v2.3 patch E: deterministic duration-graph feature columns.
DURATION_GRAPH_FEATURE_COLS = [
    "cash_runway_q",
    "cash_to_mc",
    "rd_intensity",
    "log_market_cap",
    "realized_vol_60d",
    "rolling_rate_beta_60d",
    "rolling_credit_beta_60d",
    "rolling_xbi_beta_60d",
    "age_trading_days",
    "history_valid_ratio_60d",
]


class GraphSourceGate(nn.Module):
    """Softmax gate over (w_corr, w_duration) from macro_gate_state[t].

    Single-day input (16-d gate state) -> 2-d softmax weights.
    """

    def __init__(
        self, macro_gate_state_dim: int = 16, hidden_dim: int = 32,
        init_corr: float = 0.8, init_duration: float = 0.2,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(macro_gate_state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )
        # v2.3: initialise bias so softmax starts at (0.8, 0.2),
        # forcing the duration graph to earn weight from the gate.
        with torch.no_grad():
            self.net[-1].bias.data = torch.tensor(
                [float(np.log(init_corr / max(init_duration, 1e-6))), 0.0],
                dtype=torch.float32,
            )

    def forward(self, macro_gate_state: Tensor) -> Tensor:
        """Return [..., 2] softmax weights (w_corr, w_duration)."""
        return torch.softmax(self.net(macro_gate_state), dim=-1)


def build_duration_similarity(
    duration_exposure: Tensor, active_mask: Tensor,
) -> Tensor:
    """[N, N] cosine similarity over per-active-ticker duration_exposure.

    Args:
        duration_exposure: [A, d_dur] embedding for active tickers
            on this day.
        active_mask: [N] bool active mask (used to scatter back).

    Returns:
        sim: [N, N] cosine similarity, with rows/cols of inactive
            tickers set to -inf for the diagonal and to 0 elsewhere.
    """
    n = active_mask.shape[0]
    a = duration_exposure.shape[0]
    sim_full = torch.full((n, n), 0.0, device=duration_exposure.device,
                          dtype=duration_exposure.dtype)
    if a < 2:
        return sim_full
    norm = duration_exposure.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-8)
    de_norm = duration_exposure / norm
    sim_active = de_norm @ de_norm.T               # [A, A]
    active_idx = active_mask.nonzero(as_tuple=False).squeeze(-1)
    # Scatter [A, A] block into [N, N].
    sim_full[active_idx[:, None], active_idx[None, :]] = sim_active
    sim_full.fill_diagonal_(0.0)
    return sim_full


def merge_corr_and_duration(
    a_corr: Tensor, a_duration: Tensor, w_corr: Tensor, w_duration: Tensor,
    top_k: int, active_mask: Tensor,
) -> Tensor:
    """Combine A_corr and A_duration with gate weights, return top-K.

    Args:
        a_corr: [N, N] reliability-shrunk correlation (numpy in;
            convert to tensor in caller).
        a_duration: [N, N] duration similarity (already matched dtype).
        w_corr, w_duration: scalar gate weights.
        top_k: number of neighbours per node.
        active_mask: [N] bool.

    Returns:
        top: [N, K] long tensor of neighbour indices, -1 padding.
    """
    score = w_corr * a_corr + w_duration * a_duration
    # Mask inactive rows/cols and the diagonal.
    n = active_mask.shape[0]
    inactive = ~active_mask
    score = score.clone()
    score[inactive, :] = -float("inf")
    score[:, inactive] = -float("inf")
    score.fill_diagonal_(-float("inf"))
    # Top-K per row.
    top = torch.full((n, top_k), -1, dtype=torch.long, device=score.device)
    if active_mask.sum() < 2:
        return top
    valid_rows = active_mask.nonzero(as_tuple=False).squeeze(-1)
    row_scores = score[valid_rows]                   # [A, N]
    k_eff = min(top_k, row_scores.shape[1])
    vals, idx = torch.topk(row_scores, k=k_eff, dim=-1, largest=True)
    # Drop neighbours scored -inf (i.e., no valid neighbour).
    valid_neigh = torch.isfinite(vals)
    idx = torch.where(valid_neigh, idx, torch.full_like(idx, -1))
    top[valid_rows, : k_eff] = idx
    return top


__all__ = [
    "DurationGraphConfig",
    "DURATION_GRAPH_FEATURE_COLS",
    "GraphSourceGate",
    "build_duration_similarity",
    "merge_corr_and_duration",
]
