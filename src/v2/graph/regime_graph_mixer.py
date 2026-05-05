"""Regime-scheduled graph mixer for FC-DGraph-epiSTAR.

Per spec Part D. Computes a per-day "stress" scalar from a small set
of macro and cross-sectional features, then maps it to graph-blend
weights `w_corr` and `w_duration` via a clipped sigmoid schedule.
The blended graph is `A_blend = w_corr * A_corr + w_duration * A_duration`,
and top-K is selected from `A_blend`.

This replaces the learned GraphSourceGate from DOW v2.3 with a
deterministic, auditable schedule. The motivation: in small-N
small-T panels the learned gate had little gradient signal.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RegimeMixerConfig:
    """Hyperparameters for the regime-scheduled mixer."""

    sigmoid_slope: float = 0.5
    sigmoid_intercept: float = 1.0   # subtract before sigmoid
    w_duration_min: float = 0.15
    w_duration_max: float = 0.45


def compute_stress_scalar(
    macro_arr: np.ndarray, macro_cols: list[str],
    avg_corr_z: np.ndarray, cs_disp_z: np.ndarray,
    use_xbi_ret: bool = False,
) -> np.ndarray:
    """Per-day stress scalar [T] from macro + cs scalars.

    All inputs assumed already train-fold standardised. The scalar
    is the SUM of:
        |delta_10y_20d|_z + |delta_hy_spread_20d|_z + xbi_rv_20d_z
        + avg_pairwise_corr_60d_z + cross_sectional_dispersion_z
        + (optional) |xbi_ret_20d|_z

    Returns:
        [T] float32 stress per day.
    """
    t = macro_arr.shape[0]
    out = np.zeros(t, dtype=np.float32)

    def col(name: str) -> np.ndarray:
        if name in macro_cols:
            return macro_arr[:, macro_cols.index(name)].astype(np.float32)
        return np.zeros(t, dtype=np.float32)

    out += np.abs(col("delta_10y_20d"))
    out += np.abs(col("delta_hy_spread_20d"))
    out += col("xbi_rv_20d")
    out += avg_corr_z.astype(np.float32)
    out += cs_disp_z.astype(np.float32)
    if use_xbi_ret:
        out += np.abs(col("xbi_ret_20d"))
    return out


def stress_to_weights(
    stress: np.ndarray, cfg: RegimeMixerConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """[T] stress -> (w_corr [T], w_duration [T]).

    `w_duration = sigmoid(slope * stress - intercept)` clipped to
    [w_duration_min, w_duration_max]. `w_corr = 1 - w_duration`.
    """
    cfg = cfg or RegimeMixerConfig()
    raw = cfg.sigmoid_slope * stress - cfg.sigmoid_intercept
    sig = 1.0 / (1.0 + np.exp(-raw))
    w_dur = np.clip(sig, cfg.w_duration_min, cfg.w_duration_max).astype(np.float32)
    w_corr = (1.0 - w_dur).astype(np.float32)
    return w_corr, w_dur


def merge_topk(
    a_corr: np.ndarray, a_duration: np.ndarray,
    w_corr: float, w_duration: float,
    top_k: int, active_mask: np.ndarray,
) -> np.ndarray:
    """Top-K of `w_corr * A_corr + w_duration * A_duration`.

    Args:
        a_corr: [N, N]. Inactive rows/cols already set to -inf.
        a_duration: [N, N]. Inactive rows/cols already 0 or -inf.
        w_corr, w_duration: scalars (per-day blend weights).
        top_k: K neighbours per ticker.
        active_mask: [N] bool.

    Returns:
        top: [N, K] long, -1 padded.
    """
    n = active_mask.shape[0]
    score = w_corr * a_corr + w_duration * a_duration
    # Mask inactive rows/cols and the diagonal.
    inactive = ~active_mask
    score = np.where(np.isfinite(score), score, -np.inf)
    score[inactive, :] = -np.inf
    score[:, inactive] = -np.inf
    np.fill_diagonal(score, -np.inf)
    top = np.full((n, top_k), -1, dtype=np.int64)
    if active_mask.sum() < 2:
        return top
    valid_rows = np.where(active_mask)[0]
    for i in valid_rows:
        row = score[i]
        valid = np.isfinite(row)
        if valid.sum() == 0:
            continue
        idx = np.argpartition(-row, kth=min(top_k, n - 1))[:top_k]
        idx = idx[np.argsort(-row[idx])]
        for j, k_idx in enumerate(idx):
            if np.isfinite(row[k_idx]):
                top[i, j] = int(k_idx)
    return top


__all__ = [
    "RegimeMixerConfig",
    "compute_stress_scalar",
    "stress_to_weights",
    "merge_topk",
]
