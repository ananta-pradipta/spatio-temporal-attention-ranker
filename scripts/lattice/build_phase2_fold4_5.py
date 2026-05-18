"""Fit fold scalers for the NEW F4 and F5 walk-forward folds.

Companion to scripts/lattice/build_phase2.py which fits F1, F2, F3. Added
2026-05-11 with the Scenario A dataset extension (panel 2015-01 to
2025-12). Reads the extended panel + cohorts, fits a FoldScaler for each
new fold, writes to experiments/lattice/fold{4,5}/scalers.pkl.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.lattice.data.folds import fold_indices
from src.lattice.training.standardise import (
    fit_fold_scaler, save_fold_scaler,
)


def main() -> None:
    panel = pd.read_parquet("data/lattice/processed/panel_features.parquet")
    cohorts = pd.read_parquet("data/lattice/processed/cohorts.parquet")
    panel["date"] = pd.to_datetime(panel["date"])
    cohorts["date"] = pd.to_datetime(cohorts["date"])
    dates = sorted(panel["date"].unique())
    print(f"[fold4_5] panel: {len(panel):,} rows, "
          f"{panel.ticker.nunique()} tickers, {len(dates)} dates",
          flush=True)

    for fold in (4, 5):
        train_idx, val_idx, test_idx = fold_indices(fold, dates)
        scaler = fit_fold_scaler(panel, train_idx, cohorts, fold)
        save_fold_scaler(
            scaler, Path(f"experiments/lattice/fold{fold}/scalers.pkl")
        )
        print(f"  fold {fold}: train_days={len(train_idx)} "
              f"val_days={len(val_idx)} test_days={len(test_idx)}",
              flush=True)
    print("[fold4_5] DONE", flush=True)


if __name__ == "__main__":
    main()
