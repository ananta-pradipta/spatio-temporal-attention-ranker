"""Dynamic neighbor selection for DyReg-STAR.

For each (day t, ticker i), compute the top-K=8 graph neighbors using a
rolling-window correlation graph constructed only from data up to and
including day t. Optional mixture with the static mechanistic graph is
supported via a configurable weight.

Leakage rule: at every day t the rolling window covers days
[t - window + 1, t]. Inactive tickers on day t are excluded from neighbor
candidates and from neighbor-of lists. No future returns are used.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DynamicGraphConfig:
    """Hyperparameters for dynamic neighbor selection.

    Attributes:
        window_days: rolling correlation window length.
        top_k: number of neighbors per ticker per day.
        static_mix_weight: blend weight on the static mechanistic graph.
            Final neighbor score = (1 - w) * dynamic_corr + w * static_score.
            Set to 0.0 for pure dynamic, 1.0 for pure static (sanity).
        residualize_xbi: if True, regress each ticker's returns on XBI
            biotech ETF return before computing the correlation graph.
            Not implemented in Stage 1; reserved for ablations.
    """

    window_days: int = 60
    top_k: int = 8
    static_mix_weight: float = 0.0
    residualize_xbi: bool = False


def _correlation_matrix(window_returns: np.ndarray) -> np.ndarray:
    """Compute the column-wise Pearson correlation of a [W, N] window.

    Returns a [N, N] matrix with the diagonal set to -inf so that a
    ticker is never its own top neighbor.
    """
    x = window_returns - window_returns.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-8, 1e-8, sd)
    x_norm = x / sd
    corr = (x_norm.T @ x_norm) / window_returns.shape[0]
    np.fill_diagonal(corr, -np.inf)
    return corr


def build_dynamic_neighbors(
    returns: np.ndarray,
    mask: np.ndarray,
    cfg: DynamicGraphConfig | None = None,
    static_score: np.ndarray | None = None,
) -> np.ndarray:
    """Build [T, N, K] int64 array of top-K neighbors per (day, ticker).

    Args:
        returns: [T, N] daily log returns of the panel.
        mask: [T, N] active mask. Inactive cells are excluded from
            neighbor candidates and from being a candidate's neighbor.
        cfg: hyperparameters.
        static_score: optional [N, N] static mechanistic edge-weight
            matrix. Only used if cfg.static_mix_weight > 0.

    Returns:
        neighbors: [T, N, K] int64. Days t < window_days have all -1
        entries (insufficient history); inactive (day, ticker) cells
        have all -1; missing slots are -1. Trainer must mask these.
    """
    cfg = cfg or DynamicGraphConfig()
    t_total, n = returns.shape
    k = cfg.top_k
    w = cfg.window_days
    out = np.full((t_total, n, k), -1, dtype=np.int64)

    # Pre-normalize the static score (or zero out if absent).
    if cfg.static_mix_weight > 0.0 and static_score is not None:
        s_static = static_score.astype(np.float32).copy()
        np.fill_diagonal(s_static, -np.inf)
    else:
        s_static = None

    for t in range(w, t_total):
        win = returns[t - w + 1 : t + 1]  # [w, n]
        # Skip if too many NaNs (can happen for tickers with sparse history).
        nan_mask = np.isnan(win)
        if nan_mask.any():
            win = np.where(nan_mask, 0.0, win)
        corr = _correlation_matrix(win)

        if s_static is not None:
            score = (1.0 - cfg.static_mix_weight) * corr + cfg.static_mix_weight * s_static
        else:
            score = corr

        # Mask out inactive tickers on day t.
        active_t = mask[t]
        score[~active_t, :] = -np.inf
        score[:, ~active_t] = -np.inf

        # Top-K per row using argpartition; partitioned order is fine for
        # STAR because the patch construction does not weight neighbors.
        # Use -score so that argpartition picks largest values.
        top_k_part = np.argpartition(-score, kth=min(k, n - 1), axis=1)[:, :k]
        # Filter out -inf-scored neighbors (e.g., when a ticker has fewer
        # than K active candidates).
        for i in range(n):
            if not active_t[i]:
                continue
            row_scores = score[i, top_k_part[i]]
            valid = np.isfinite(row_scores)
            chosen = top_k_part[i][valid]
            out[t, i, : len(chosen)] = chosen

    return out


def static_score_from_correlation_edges(
    edge_index: np.ndarray, edge_weight: np.ndarray, num_nodes: int
) -> np.ndarray:
    """Convert sparse static edges into a dense [N, N] score matrix.

    Used only when cfg.static_mix_weight > 0 (mixed dynamic/static).
    """
    s = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for k in range(edge_index.shape[1]):
        u = int(edge_index[0, k])
        v = int(edge_index[1, k])
        w = float(edge_weight[k]) if edge_weight is not None and k < len(edge_weight) else 1.0
        s[u, v] += w
        s[v, u] += w
    return s


__all__ = [
    "DynamicGraphConfig",
    "build_dynamic_neighbors",
    "static_score_from_correlation_edges",
]
