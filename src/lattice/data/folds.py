"""Walk-forward fold definitions for LATTICE.

Inherited verbatim from the v2/biotech protocol so LATTICE's S&P 500 numbers
are paired-t-testable seed-for-seed against biotech RAG-STAR locked numbers.

5-day embargo at every train/val and val/test boundary.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class FoldDef:
    """One walk-forward fold's date ranges."""

    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


FOLDS: dict[int, FoldDef] = {
    1: FoldDef(
        fold=1,
        train_start=pd.Timestamp("2015-01-09"),
        train_end=pd.Timestamp("2018-12-21"),
        val_start=pd.Timestamp("2019-01-02"),
        val_end=pd.Timestamp("2019-12-23"),
        test_start=pd.Timestamp("2020-01-02"),
        test_end=pd.Timestamp("2020-12-31"),
    ),
    2: FoldDef(
        fold=2,
        train_start=pd.Timestamp("2015-01-09"),
        train_end=pd.Timestamp("2020-12-23"),
        val_start=pd.Timestamp("2021-01-04"),
        val_end=pd.Timestamp("2021-06-23"),
        test_start=pd.Timestamp("2021-07-01"),
        test_end=pd.Timestamp("2022-06-22"),
    ),
    # F3 test extended from 2022-12-22 to 2023-06-22 (2026-05-11 Scenario A
    # 5F-balanced-shift) so F3 has ~248 test days matching F1/F2. F3 train
    # and val unchanged so the F3 scaler can be reused as-is.
    3: FoldDef(
        fold=3,
        train_start=pd.Timestamp("2015-01-09"),
        train_end=pd.Timestamp("2021-12-23"),
        val_start=pd.Timestamp("2022-01-03"),
        val_end=pd.Timestamp("2022-06-23"),
        test_start=pd.Timestamp("2022-07-01"),
        test_end=pd.Timestamp("2023-06-22"),
    ),
    # F4 and F5 added 2026-05-11 per Scenario A 5F-balanced-shift. F4 test
    # spans the 2024 AI rally + Fed-pause anticipation (full calendar year);
    # F5 test spans the 2025 Fed-cut cycle and post-election repositioning
    # through 2026 H1 to give F5 a balanced ~210-day test window.
    4: FoldDef(
        fold=4,
        train_start=pd.Timestamp("2015-01-09"),
        train_end=pd.Timestamp("2023-06-23"),
        val_start=pd.Timestamp("2023-07-03"),
        val_end=pd.Timestamp("2023-12-22"),
        test_start=pd.Timestamp("2024-01-02"),
        test_end=pd.Timestamp("2024-12-31"),
    ),
    5: FoldDef(
        fold=5,
        train_start=pd.Timestamp("2015-01-09"),
        train_end=pd.Timestamp("2024-12-23"),
        val_start=pd.Timestamp("2025-01-02"),
        val_end=pd.Timestamp("2025-06-23"),
        test_start=pd.Timestamp("2025-07-01"),
        test_end=pd.Timestamp("2026-04-30"),
    ),
}

EMBARGO_DAYS = 5


def fold_indices(
    fold: int, dates: list[pd.Timestamp]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (train_idx, val_idx, test_idx) integer indices into the dates list.

    Applies the 5-day embargo at the train-val and val-test boundaries by
    truncating the trailing days of each window.
    """
    fd = FOLDS[fold]
    dates_arr = np.asarray(dates)

    def in_range(start: pd.Timestamp, end: pd.Timestamp) -> np.ndarray:
        m = (dates_arr >= start) & (dates_arr <= end)
        return np.where(m)[0]

    train = in_range(fd.train_start, fd.train_end)
    val = in_range(fd.val_start, fd.val_end)
    test = in_range(fd.test_start, fd.test_end)

    if EMBARGO_DAYS > 0:
        train = train[: max(0, len(train) - EMBARGO_DAYS)]
        val = val[: max(0, len(val) - EMBARGO_DAYS)]

    return train.astype(np.int64), val.astype(np.int64), test.astype(np.int64)


# Reverse-time validation: fixed historical val window for ALL folds.
# Diagnostic for the val-regime overfit hypothesis (2026-05-13). Train is the
# union of pre-val and post-val segments (excluding val and test); val is
# always 2017 (calm-bull, no major regime break, low VIX). This breaks the
# val/test regime correlation by construction since val sits 3-9 years before
# any test window.
REVERSE_VAL_START = pd.Timestamp("2017-01-03")
REVERSE_VAL_END = pd.Timestamp("2017-12-29")


def fold_indices_reverse_val(
    fold: int, dates: list[pd.Timestamp]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reverse-time-val variant of fold_indices.

    Train is two-segment: pre-2017 + post-2017 up to fd.train_end.
    Val is fixed at calendar 2017 across all folds.
    Test is identical to the standard fold_indices test window.
    5-day embargo applied at every boundary between segments.
    """
    fd = FOLDS[fold]
    dates_arr = np.asarray(dates)

    def in_range(start: pd.Timestamp, end: pd.Timestamp) -> np.ndarray:
        m = (dates_arr >= start) & (dates_arr <= end)
        return np.where(m)[0]

    val = in_range(REVERSE_VAL_START, REVERSE_VAL_END)
    test = in_range(fd.test_start, fd.test_end)

    pre_val_end = REVERSE_VAL_START - pd.Timedelta(days=1)
    post_val_start = REVERSE_VAL_END + pd.Timedelta(days=1)
    train_pre = in_range(fd.train_start, pre_val_end)
    train_post = in_range(post_val_start, fd.train_end)

    if EMBARGO_DAYS > 0:
        train_pre = train_pre[: max(0, len(train_pre) - EMBARGO_DAYS)]
        train_post = train_post[EMBARGO_DAYS:] if len(train_post) > EMBARGO_DAYS else train_post[:0]
        train_post = train_post[: max(0, len(train_post) - EMBARGO_DAYS)]
        val = val[: max(0, len(val) - EMBARGO_DAYS)]

    train = np.concatenate([train_pre, train_post])
    return train.astype(np.int64), val.astype(np.int64), test.astype(np.int64)


# Two-regime val: 2017 H2 (calm late-bull, VIX ~10) + 2018 H2 (vol spike +
# Q4 sell-off, VIX peaks 37). Captures both calm and stress in val IC, no
# retrospective regime matching. Total ~252 days, same size as canonical
# F1 val and ~2x canonical F2-F5 val. Train is two-segment (pre-2017 H2 +
# post-2018 H2, up to fd.train_end).
TWO_REGIME_VAL_RANGES = [
    (pd.Timestamp("2017-07-03"), pd.Timestamp("2017-12-29")),
    (pd.Timestamp("2018-07-02"), pd.Timestamp("2018-12-31")),
]


def fold_indices_two_regime_val(
    fold: int, dates: list[pd.Timestamp]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two-regime-val variant of fold_indices.

    Val = 2017 H2 + 2018 H2 across all folds. Train is segmented to exclude
    val ranges. For F1, the 2018 H2 val segment is naturally capped at
    fd.train_end (2018-12-21). 5-day embargo applied at every boundary.
    """
    fd = FOLDS[fold]
    dates_arr = np.asarray(dates)

    def in_range(start: pd.Timestamp, end: pd.Timestamp) -> np.ndarray:
        m = (dates_arr >= start) & (dates_arr <= end)
        return np.where(m)[0]

    test = in_range(fd.test_start, fd.test_end)

    # Apply val ranges (clipped to fd.train_end for F1).
    val_parts = []
    train_segments = []
    seg_start = fd.train_start
    for vs, ve in TWO_REGIME_VAL_RANGES:
        ve_clipped = min(ve, fd.train_end)
        if vs > fd.train_end:
            break
        # Train segment before this val range.
        seg_end = vs - pd.Timedelta(days=1)
        if seg_end > seg_start:
            train_segments.append((seg_start, seg_end))
        val_parts.append(in_range(vs, ve_clipped))
        seg_start = ve_clipped + pd.Timedelta(days=1)
    # Trailing train segment up to fd.train_end.
    if seg_start <= fd.train_end:
        train_segments.append((seg_start, fd.train_end))

    val = np.concatenate(val_parts) if val_parts else np.zeros((0,), dtype=np.int64)

    # Apply embargo: trim EMBARGO_DAYS from each segment's edges adjacent to a
    # val/test boundary. Leading embargo removes from the start; trailing
    # removes from the end.
    train_pieces = []
    for i, (s, e) in enumerate(train_segments):
        seg = in_range(s, e)
        if EMBARGO_DAYS > 0:
            leading_embargo = i > 0
            trailing_embargo = i < len(train_segments) - 1 or True
            if leading_embargo:
                seg = seg[EMBARGO_DAYS:] if len(seg) > EMBARGO_DAYS else seg[:0]
            if trailing_embargo:
                seg = seg[: max(0, len(seg) - EMBARGO_DAYS)]
        train_pieces.append(seg)
    train = np.concatenate(train_pieces) if train_pieces else np.zeros((0,), dtype=np.int64)

    if EMBARGO_DAYS > 0 and len(val) > 0:
        val = val[: max(0, len(val) - EMBARGO_DAYS)]

    return train.astype(np.int64), val.astype(np.int64), test.astype(np.int64)


__all__ = [
    "FoldDef", "FOLDS", "EMBARGO_DAYS", "fold_indices",
    "fold_indices_reverse_val", "REVERSE_VAL_START", "REVERSE_VAL_END",
    "fold_indices_two_regime_val", "TWO_REGIME_VAL_RANGES",
]
