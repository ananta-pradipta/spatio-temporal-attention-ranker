"""Per-(day, ticker) age and history-validity features for OW-epiSTAR.

Computed from the existing panel active mask, no new data feeds needed.
The OW-epiSTAR spec Section 6.1 lists these features; this module emits
the subset we use for the IPO analogue retrieval keys, the dual-gate
inputs, and the cohort-level evaluation reports.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


AGE_COLS = [
    "age_trading_days",
    "log1p_age_trading_days",
    "age_bucket_0_20",
    "age_bucket_21_60",
    "age_bucket_61_252",
    "age_bucket_253_plus",
    "history_valid_ratio_20d",
    "history_valid_ratio_60d",
]


@dataclass
class AgeFeatureConfig:
    """Hyperparameters for age-feature computation."""

    fresh_ipo_days: int = 60
    young_public_days: int = 252
    history_window_short: int = 20
    history_window_long: int = 60


def compute_age_days(mask: np.ndarray) -> np.ndarray:
    """Per-(day, ticker) age in trading days since first active day.

    Tickers that have never been active have age 0 throughout. The
    first active day has age 0 and the count increases monotonically
    while the ticker remains in the panel (whether active or not on
    any given subsequent day).
    """
    t_total, n = mask.shape
    out = np.zeros((t_total, n), dtype=np.int64)
    first_active = np.full(n, -1, dtype=np.int64)
    for t in range(t_total):
        active = mask[t]
        new_active = np.where(active & (first_active == -1))[0]
        first_active[new_active] = t
        age = np.where(first_active >= 0, t - first_active, 0)
        out[t] = np.maximum(age, 0)
    return out


def compute_history_valid_ratio(mask: np.ndarray, window: int) -> np.ndarray:
    """Per-(day, ticker) fraction of the trailing `window` days that
    were active. Useful for the IPO retrieval key and the dual gate."""
    t_total, n = mask.shape
    out = np.zeros((t_total, n), dtype=np.float32)
    if window <= 0:
        return out
    for t in range(t_total):
        if t < window:
            sub = mask[: t + 1]
            denom = float(t + 1)
        else:
            sub = mask[t - window + 1 : t + 1]
            denom = float(window)
        out[t] = sub.sum(axis=0) / denom
    return out


def build_age_feature_tensor(
    mask: np.ndarray, cfg: AgeFeatureConfig | None = None
) -> tuple[np.ndarray, list[str]]:
    """Return [T, N, 8] age-feature tensor and column names."""
    cfg = cfg or AgeFeatureConfig()
    t_total, n = mask.shape
    out = np.zeros((t_total, n, len(AGE_COLS)), dtype=np.float32)
    age = compute_age_days(mask)
    out[:, :, 0] = age.astype(np.float32)
    out[:, :, 1] = np.log1p(age).astype(np.float32)
    out[:, :, 2] = ((age >= 0) & (age <= 20)).astype(np.float32)
    out[:, :, 3] = ((age > 20) & (age <= cfg.fresh_ipo_days)).astype(np.float32)
    out[:, :, 4] = ((age > cfg.fresh_ipo_days) & (age <= cfg.young_public_days)).astype(np.float32)
    out[:, :, 5] = (age > cfg.young_public_days).astype(np.float32)
    out[:, :, 6] = compute_history_valid_ratio(mask, cfg.history_window_short)
    out[:, :, 7] = compute_history_valid_ratio(mask, cfg.history_window_long)
    return out, list(AGE_COLS)


def cohort_label(age_days: int, cfg: AgeFeatureConfig | None = None) -> str:
    """Map an age-in-trading-days to a cohort label for evaluation."""
    cfg = cfg or AgeFeatureConfig()
    if age_days <= cfg.fresh_ipo_days:
        return "fresh_ipo"
    if age_days <= cfg.young_public_days:
        return "young_public"
    return "seasoned"


__all__ = [
    "AGE_COLS",
    "AgeFeatureConfig",
    "compute_age_days",
    "compute_history_valid_ratio",
    "build_age_feature_tensor",
    "cohort_label",
]
