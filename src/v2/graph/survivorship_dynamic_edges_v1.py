"""Reliability-shrunk rolling-correlation dynamic graph for OW-epiSTAR v1.

Soft replacement for the hard age cutoff used in epiDyReg-STAR. Builds
per-day top-K neighbour lists from a rolling Pearson correlation that
is shrunk toward zero by the count of overlapping tradable days:

    rho_shrunk[t, i, j] = rho_raw[t, i, j] * n_overlap / (n_overlap + tau)

with tau = 30, corr_window = 60, top_k = 8, and a hard floor of
``min_overlap_absolute = 5`` below which the edge is dropped.

The graph uses ``tradable_mask`` (from ``minimal_masks.py``) to count
overlap, so young tickers can participate in the graph as soon as they
have any common tradable history with another ticker. The hard age
cutoff that hurt fold-1 in earlier experiments is replaced by this
soft reliability correction.

Output is a [T, N, K] long tensor of neighbour ticker indices, with -1
padding when fewer than K reliable neighbours exist for a (day, ticker).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SurvivorshipGraphConfig:
    """Hyperparameters for the reliability-shrunk dynamic graph."""

    corr_window: int = 60
    tau: float = 30.0
    top_k: int = 8
    min_overlap_absolute: int = 5


def shrunk_correlation_neighbors_one_day(
    returns: np.ndarray,
    tradable_mask: np.ndarray,
    t: int,
    cfg: SurvivorshipGraphConfig,
) -> tuple[np.ndarray, dict]:
    """Per-day reliability-shrunk neighbour list.

    Args:
        returns: [T, N] daily log returns of the panel.
        tradable_mask: [T, N] bool tradable indicator.
        t: query day index (must be >= cfg.corr_window).
        cfg: graph config.

    Returns:
        top: [N, K] long array of neighbour ticker indices for each
            ticker; -1 padding when fewer than K reliable neighbours.
        diag: dict of per-day diagnostic stats (avg n_overlap by age
            cohort can be added by the caller; here we report the
            cross-sectional summary of overlap and shrinkage).
    """
    n = returns.shape[1]
    w = cfg.corr_window
    win_returns = returns[t - w + 1 : t + 1]
    win_mask = tradable_mask[t - w + 1 : t + 1]
    nan_filled = np.where(win_mask, np.where(np.isnan(win_returns), 0.0, win_returns), 0.0)
    valid_count = win_mask.sum(axis=0).astype(np.float32)

    x = nan_filled - nan_filled.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-8, 1e-8, sd)
    x_norm = x / sd
    rho = (x_norm.T @ x_norm) / float(w)

    overlap = np.minimum(valid_count[:, None], valid_count[None, :])
    shrinkage = overlap / (overlap + cfg.tau)
    rho_shrunk = rho * shrinkage
    rho_shrunk[overlap < cfg.min_overlap_absolute] = -np.inf
    np.fill_diagonal(rho_shrunk, -np.inf)

    rho_shrunk[~tradable_mask[t], :] = -np.inf
    rho_shrunk[:, ~tradable_mask[t]] = -np.inf

    top = np.full((n, cfg.top_k), -1, dtype=np.int64)
    if tradable_mask[t].sum() < 2:
        diag = {
            "n_active": int(tradable_mask[t].sum()),
            "mean_overlap": float(overlap[overlap >= cfg.min_overlap_absolute].mean()
                                  if (overlap >= cfg.min_overlap_absolute).any() else 0.0),
            "mean_shrinkage": float(shrinkage[overlap >= cfg.min_overlap_absolute].mean()
                                    if (overlap >= cfg.min_overlap_absolute).any() else 0.0),
            "shrunk_corr_mean": float(0.0),
            "shrunk_corr_std": float(0.0),
        }
        return top, diag
    part = np.argpartition(-rho_shrunk, kth=min(cfg.top_k, n - 1), axis=1)[:, : cfg.top_k]
    for i in range(n):
        if not tradable_mask[t, i]:
            continue
        row_scores = rho_shrunk[i, part[i]]
        valid = np.isfinite(row_scores)
        chosen_local = np.where(valid)[0]
        if chosen_local.size == 0:
            continue
        # Sort the valid neighbours by descending shrunk correlation so
        # the model sees them ordered.
        sorted_local = chosen_local[np.argsort(-row_scores[chosen_local])]
        chosen = part[i][sorted_local]
        top[i, : len(chosen)] = chosen

    finite_mask = np.isfinite(rho_shrunk)
    diag = {
        "n_active": int(tradable_mask[t].sum()),
        "mean_overlap": float(overlap[finite_mask].mean()) if finite_mask.any() else 0.0,
        "mean_shrinkage": float(shrinkage[finite_mask].mean()) if finite_mask.any() else 0.0,
        "shrunk_corr_mean": float(rho_shrunk[finite_mask].mean()) if finite_mask.any() else 0.0,
        "shrunk_corr_std": float(rho_shrunk[finite_mask].std()) if finite_mask.any() else 0.0,
    }
    return top, diag


def build_survivorship_neighbors(
    returns: np.ndarray,
    tradable_mask: np.ndarray,
    cfg: SurvivorshipGraphConfig | None = None,
) -> tuple[np.ndarray, list[dict]]:
    """Per-day [T, N, K] neighbour list using reliability shrinkage.

    Days before ``cfg.corr_window`` get an all -1 neighbour list (the
    trainer should skip these with the existing temporal-window guard).

    Returns:
        top: [T, N, K] long array.
        per_day_diag: list of length T with per-day diagnostic dicts.
    """
    cfg = cfg or SurvivorshipGraphConfig()
    t_total, n = returns.shape
    out = np.full((t_total, n, cfg.top_k), -1, dtype=np.int64)
    diag: list[dict] = []
    for t in range(t_total):
        if t < cfg.corr_window:
            diag.append({"n_active": 0})
            continue
        top, d = shrunk_correlation_neighbors_one_day(returns, tradable_mask, t, cfg)
        out[t] = top
        diag.append(d)
    return out, diag


__all__ = [
    "SurvivorshipGraphConfig",
    "shrunk_correlation_neighbors_one_day",
    "build_survivorship_neighbors",
]
