"""Cohort fractions per day: 4-dim sub-key for the SBP regime key.

For each panel day t, compute:
    1. fraction of active universe in their first 21 trading days post-IPO
    2. fraction in days 22-126
    3. fraction in days 127-252
    4. mean log(1 + age_in_trading_days), capped at log(2520)

Per spec Section 5.1. The 4 dimensions are appended to the existing
14-dim regime key (8-dim risk + 6-dim cs diagnostics) to produce the
18-dim cohort-augmented key used by dual-pool retrieval.

Also exposes per-(day, ticker) cohort bucket labels used by the IRF
reweighting and V-REx penalty in the loss.
"""
from __future__ import annotations

import numpy as np


COHORT_KEY_COLS = [
    "frac_age_0_21d",
    "frac_age_22_126d",
    "frac_age_127_252d",
    "mean_log_age_capped",
]


def build_cohort_subkey(age_days: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per-day 4-dim cohort sub-key from age and active mask.

    Args:
        age_days: [T, N] integer age in trading days (from
            src.v2.data.age_features.compute_age_days).
        mask: [T, N] active mask.

    Returns:
        [T, 4] float32 cohort sub-key.
    """
    t_total = age_days.shape[0]
    out = np.zeros((t_total, 4), dtype=np.float32)
    log_cap = np.log(2520.0)
    for t in range(t_total):
        m = mask[t]
        if m.sum() < 1:
            continue
        ages = age_days[t, m]
        denom = float(ages.size)
        out[t, 0] = float(((ages >= 0) & (ages < 22)).sum()) / denom
        out[t, 1] = float(((ages >= 22) & (ages < 127)).sum()) / denom
        out[t, 2] = float(((ages >= 127) & (ages < 253)).sum()) / denom
        log_age = np.log1p(ages.astype(np.float32))
        log_age = np.minimum(log_age, log_cap)
        out[t, 3] = float(np.mean(log_age))
    return out


COHORT_BUCKET_LABELS = [
    "fresh_ipo_0_21",
    "young_22_126",
    "young_127_252",
    "seasoned_253_plus",
]


def cohort_bucket_index(age: int) -> int:
    """Return 0-3 cohort bucket index for a single age in trading days."""
    if age < 22:
        return 0
    if age < 127:
        return 1
    if age < 253:
        return 2
    return 3


def cohort_bucket_per_cell(age_days: np.ndarray) -> np.ndarray:
    """Per-(day, ticker) cohort bucket index in {0, 1, 2, 3}."""
    out = np.zeros_like(age_days, dtype=np.int8)
    out[(age_days >= 22) & (age_days < 127)] = 1
    out[(age_days >= 127) & (age_days < 253)] = 2
    out[age_days >= 253] = 3
    return out


__all__ = [
    "COHORT_KEY_COLS",
    "COHORT_BUCKET_LABELS",
    "build_cohort_subkey",
    "cohort_bucket_index",
    "cohort_bucket_per_cell",
]
