"""Walk-forward fold definitions for the v2 panel.

Three expanding-window folds with a 5-day embargo at every train-validation
and validation-test boundary. Identical to the v1 / qualifying exam paper
protocol; preserved across the v2 restart.

Fold-3 reservation: during the v1 13-iteration investigation, fold 3 was
treated as a reserved set. v2 uses the same folds but does not enforce the
reservation protocol; v2 is a single new architecture, not an iterative
investigation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class FoldDef:
    """One walk-forward fold's date ranges.

    Attributes:
        fold: integer fold id (1, 2, or 3).
        train_start: first training date (inclusive).
        train_end: last training date (inclusive, before embargo).
        val_start: first validation date.
        val_end: last validation date (inclusive, before embargo).
        test_start: first test date.
        test_end: last test date.
    """

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
    3: FoldDef(
        fold=3,
        train_start=pd.Timestamp("2015-01-09"),
        train_end=pd.Timestamp("2021-12-23"),
        val_start=pd.Timestamp("2022-01-03"),
        val_end=pd.Timestamp("2022-06-23"),
        test_start=pd.Timestamp("2022-07-01"),
        test_end=pd.Timestamp("2022-12-22"),
    ),
}

EMBARGO_DAYS = 5


def fold_indices(
    fold: int, dates: list[pd.Timestamp]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return train, validation, test integer indices into the dates list.

    Args:
        fold: fold id (1, 2, 3).
        dates: full list of trading days for the panel.

    Returns:
        (train_idx, val_idx, test_idx): three int64 numpy arrays.
    """
    fd = FOLDS[fold]
    dates_arr = np.array(dates)

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


__all__ = ["FoldDef", "FOLDS", "EMBARGO_DAYS", "fold_indices"]
