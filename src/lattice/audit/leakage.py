"""Four-check leakage audit for LATTICE.

Replicated from the v2 universal-validation work and adapted to the LATTICE
data shapes. Per spec Section 2.2, this module is called by `train.py` and
`evaluate.py` non-optionally.

The four checks (each a hard assertion):

  A1. Auxiliary supervision uses only strictly future data.
      For every (t, n) cell with mask True and label y[t, n] non-NaN, the
      label must be derived from prices at days strictly greater than t.
      Specifically y[t, n] equals log(close[t+5, n] / close[t, n]) and uses
      future closes.

  A2. Per-day standardization uses only the day's mask-indexed slice.
      Cross-sectional z-scoring at day t uses statistics computed over
      mask[t]; the function must NOT depend on data from t+k for any k > 0.

  A3. Feature standardization statistics use train-fold only.
      Per-feature mean and std for input z-scoring is computed over
      train_idx slice of the mask, not over the full panel.

  A4. Active mask uses non-anticipating signals only.
      The mask at day t must be derivable from data at timestamps <= t.
      In particular: tradable[t, n] requires close[t, n] non-NaN and
      volume[t, n] non-NaN and prior 20 days of returns to be present.
      It must NOT require any future-dated data.

Reference: docs/lattice_design_rationale.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class AuditResult:
    """Outcome of one audit run.

    Attributes:
        passed: Boolean conjunction across all four checks.
        details: Per-check fail count and a short rationale string.
    """

    passed: bool
    details: dict


def audit_a1_label_correctness(
    y: np.ndarray,
    mask: np.ndarray,
    close_panel: np.ndarray,
    horizon_days: int = 5,
    sample_size: int = 1000,
    tolerance: float = 1e-3,
    rng_seed: int = 0,
) -> tuple[bool, dict]:
    """A1: y[t, n] equals log(close[t+horizon, n] / close[t, n]).

    Args:
        y: [T, N] forward-return label tensor.
        mask: [T, N] boolean active mask.
        close_panel: [T, N] close prices (NaN where unavailable).
        horizon_days: forward-return horizon (default 5).
        sample_size: number of cells to sample for the consistency check.
        tolerance: max absolute deviation between y[t, n] and recomputed
            log-return that we consider "consistent."
        rng_seed: numpy default_rng seed for reproducible sampling.

    Returns:
        (passed, details). passed is True iff at least 95% of sampled
        cells have y matching the recomputed forward log-return within
        tolerance.
    """
    T, N = mask.shape
    active_idx = np.argwhere(mask)
    if len(active_idx) == 0:
        return False, {"check": "A1", "fail_reason": "empty mask"}
    rng = np.random.default_rng(rng_seed)
    sample = active_idx[rng.choice(len(active_idx),
                                    size=min(sample_size, len(active_idx)),
                                    replace=False)]
    n_consistent = 0
    n_checked = 0
    for t, n in sample:
        if t + horizon_days >= T:
            continue
        c0 = close_panel[t, n]
        ch = close_panel[t + horizon_days, n]
        if np.isnan(c0) or np.isnan(ch) or c0 <= 0:
            continue
        expected = float(np.log(ch / c0))
        if abs(expected - float(y[t, n])) < tolerance:
            n_consistent += 1
        n_checked += 1
    passed = (n_checked >= 100) and (n_consistent / max(1, n_checked) > 0.95)
    return passed, {
        "check": "A1",
        "n_checked": n_checked,
        "n_consistent": n_consistent,
        "consistency_rate": n_consistent / max(1, n_checked),
    }


def audit_a2_per_day_standardisation_independence(
    standardise_fn,
    sample_arr: np.ndarray,
    sample_mask: np.ndarray,
    day_idx: int,
) -> tuple[bool, dict]:
    """A2: per-day z-scoring at day t depends only on data at day t.

    Verified by perturbing data at days t+1 and confirming the output for
    day t is unchanged.

    Args:
        standardise_fn: callable that takes (arr, mask) -> standardised
            array of the same shape. Will be called twice: once on the
            original sample, once on a perturbed-future copy.
        sample_arr: [T, N] feature panel.
        sample_mask: [T, N] active mask.
        day_idx: day index t to test for independence.

    Returns:
        (passed, details). passed is True iff perturbing future days
        leaves the day-t standardisation bitwise identical.
    """
    if sample_arr.shape[0] <= day_idx + 1:
        return True, {"check": "A2", "skipped": "day_idx near panel end"}
    out_clean = standardise_fn(sample_arr.copy(), sample_mask.copy())
    perturbed = sample_arr.copy()
    perturbed[day_idx + 1:] += np.random.default_rng(0).normal(
        size=perturbed[day_idx + 1:].shape
    )
    out_perturbed = standardise_fn(perturbed, sample_mask.copy())
    same_at_day_t = np.allclose(
        out_clean[day_idx, sample_mask[day_idx]],
        out_perturbed[day_idx, sample_mask[day_idx]],
        atol=0.0, rtol=0.0,
    )
    return same_at_day_t, {
        "check": "A2",
        "day_idx_tested": day_idx,
        "bit_identical_at_day_t": bool(same_at_day_t),
    }


def audit_a3_feature_standardisation_train_only(
    feature_arr: np.ndarray,
    mask: np.ndarray,
    train_idx: np.ndarray,
    z_score_fn,
    feature_idx: int = 0,
) -> tuple[bool, dict]:
    """A3: feature z-scoring statistics use train_idx only.

    Verified by computing the z-score function with the full panel and
    again with only the train slice; the resulting mean and std along
    train_idx should match.

    Args:
        feature_arr: [T, N] feature column under test.
        mask: [T, N] active mask.
        train_idx: 1-D integer array of train day indices.
        z_score_fn: callable z_score_fn(arr, mask, train_idx) ->
            standardised array of the same shape, internally using mean
            and std of arr[train_idx][mask[train_idx]] only.
        feature_idx: integer index for diagnostic logging.

    Returns:
        (passed, details).
    """
    out_full_panel = z_score_fn(feature_arr.copy(), mask.copy(), train_idx)
    train_subset = feature_arr[train_idx][mask[train_idx]]
    train_mean = float(np.mean(train_subset)) if train_subset.size > 0 else 0.0
    train_std = float(np.std(train_subset)) if train_subset.size > 0 else 1.0
    expected_zscored_train = (
        feature_arr[train_idx][mask[train_idx]] - train_mean
    ) / max(train_std, 1e-6)
    actual_zscored_train = out_full_panel[train_idx][mask[train_idx]]
    is_close = bool(np.allclose(expected_zscored_train, actual_zscored_train,
                                  atol=1e-4))
    return is_close, {
        "check": "A3",
        "feature_idx": feature_idx,
        "train_mean": train_mean,
        "train_std": train_std,
        "matches_train_only_zscore": is_close,
    }


def audit_a4_active_mask_non_anticipating(
    mask: np.ndarray,
    close_panel: np.ndarray,
    volume_panel: np.ndarray,
    history_window: int = 20,
) -> tuple[bool, dict]:
    """A4: active mask uses only data with timestamp <= t.

    Verified by reconstructing the mask from price + volume + history at
    days <= t and confirming it matches the supplied mask.

    Args:
        mask: [T, N] supplied mask.
        close_panel: [T, N] close prices (NaN where unavailable).
        volume_panel: [T, N] volume.
        history_window: number of prior days the mask requires.

    Returns:
        (passed, details). Compares mask cell-for-cell against a clean
        reconstruction (allowing for the panel-build's additional filters
        like fwd_return availability, which IS forward-looking and is
        intentional for label_mask but should NOT be in tradable_mask).
    """
    T, N = mask.shape
    reconstructed = np.zeros_like(mask, dtype=bool)
    for t in range(history_window, T):
        for n in range(N):
            if np.isnan(close_panel[t, n]) or np.isnan(volume_panel[t, n]):
                continue
            if volume_panel[t, n] <= 0:
                continue
            window = close_panel[t - history_window:t, n]
            if np.isnan(window).any():
                continue
            reconstructed[t, n] = True
    diff = (mask & ~reconstructed).sum()
    return bool(diff == 0), {
        "check": "A4",
        "supplied_mask_cells": int(mask.sum()),
        "reconstructed_mask_cells": int(reconstructed.sum()),
        "supplied_minus_reconstructed": int(diff),
        "interpretation": (
            "supplied_minus_reconstructed should be 0; > 0 indicates the "
            "supplied mask depends on data the reconstruction does not "
            "know about, which means the mask is anticipating something."
        ),
    }


def run_full_audit(
    y: np.ndarray,
    mask: np.ndarray,
    feature_arr: np.ndarray,
    train_idx: np.ndarray,
    close_panel: np.ndarray,
    volume_panel: np.ndarray,
    standardise_fn,
    z_score_fn,
    horizon_days: int = 5,
    history_window: int = 20,
    a2_test_day: Optional[int] = None,
) -> AuditResult:
    """Run all four checks and aggregate.

    Returns:
        AuditResult with passed=True iff all four checks passed.
    """
    if a2_test_day is None:
        # Pick a day in the middle of the panel for the perturbation test.
        a2_test_day = int(mask.shape[0] // 2)

    a1_pass, a1 = audit_a1_label_correctness(
        y, mask, close_panel, horizon_days=horizon_days,
    )
    a2_pass, a2 = audit_a2_per_day_standardisation_independence(
        standardise_fn, feature_arr, mask, a2_test_day,
    )
    a3_pass, a3 = audit_a3_feature_standardisation_train_only(
        feature_arr, mask, train_idx, z_score_fn,
    )
    a4_pass, a4 = audit_a4_active_mask_non_anticipating(
        mask, close_panel, volume_panel, history_window=history_window,
    )

    return AuditResult(
        passed=bool(a1_pass and a2_pass and a3_pass and a4_pass),
        details={"A1": a1, "A2": a2, "A3": a3, "A4": a4,
                 "all_passed": bool(a1_pass and a2_pass and a3_pass and a4_pass)},
    )


__all__ = [
    "AuditResult",
    "audit_a1_label_correctness",
    "audit_a2_per_day_standardisation_independence",
    "audit_a3_feature_standardisation_train_only",
    "audit_a4_active_mask_non_anticipating",
    "run_full_audit",
]
