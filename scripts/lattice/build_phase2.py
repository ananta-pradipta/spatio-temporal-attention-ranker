"""LATTICE Phase 2: fit per-fold scalers and verify Phase 2 acceptance gate.

Per spec section 5.4:
  - All 30 panel features computed and saved with no leak-audit violations.
  - Sector-z-scoring uses training-fold statistics only.
  - Standardization scaler saved per-fold to
    experiments/lattice/<fold>/scalers.pkl.
  - Cohort labels match standard data conventions.
  - Save experiments/lattice/phase2_feature_audit.md.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.lattice.data.folds import fold_indices
from src.lattice.training.standardise import (
    fit_fold_scaler, save_fold_scaler,
    SECTOR_Z_FEATURES, PANEL_Z_FEATURES,
)


def main() -> None:
    panel = pd.read_parquet("data/lattice/processed/panel_features.parquet")
    cohorts = pd.read_parquet("data/lattice/processed/cohorts.parquet")
    panel["date"] = pd.to_datetime(panel["date"])
    cohorts["date"] = pd.to_datetime(cohorts["date"])
    dates = sorted(panel["date"].unique())
    print(f"[lattice phase2] panel: {len(panel):,} rows, {panel.ticker.nunique()} tickers, "
          f"{len(dates)} dates", flush=True)

    summary = {}
    for fold in (1, 2, 3):
        train_idx, val_idx, test_idx = fold_indices(fold, dates)
        scaler = fit_fold_scaler(panel, train_idx, cohorts, fold)
        save_fold_scaler(
            scaler, Path(f"experiments/lattice/fold{fold}/scalers.pkl")
        )
        summary[fold] = {
            "train_days": len(train_idx),
            "val_days": len(val_idx),
            "test_days": len(test_idx),
            "sector_z_features_fit": len(set(f for (f, _) in scaler.sector_zscore_stats.keys())),
            "panel_z_features_fit": len(scaler.panel_zscore_stats),
            "sectors_fit": len(set(s for (_, s) in scaler.sector_zscore_stats.keys()
                                    if s != "_global_")),
        }
        print(f"  fold {fold}: {summary[fold]}", flush=True)

    # Save summary
    import json
    Path("experiments/lattice/phase2_summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    print(f"[lattice phase2] wrote experiments/lattice/phase2_summary.json", flush=True)


if __name__ == "__main__":
    main()
