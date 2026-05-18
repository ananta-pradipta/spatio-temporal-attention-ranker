"""Rebuild the LATTICE panel + macro + scalers into the AUGMENTED 37-feature
form at a sibling directory (data/lattice_aug/processed and
experiments/lattice_aug/), keeping the canonical 26-feature outputs intact.

Used by the 2026-05-12 Tier A1 + A2 panel improvement work (see
docs/panel_feature_improvement.md). The augmented panel adds 11 columns:

    K-line (5):              kmid, klen, kup, klow, ksft
    Multi-horizon mom (2):   log_return_60d, log_return_12m_minus_1m
    Vol + PV extra (4):      max20, ivol_21d, corr20, cord20

After this script finishes the augmented training run is selected by:

    export LATTICE_PROCESSED_DIR=data/lattice_aug/processed
    export LATTICE_SCALER_DIR=experiments/lattice_aug
    python -m src.invar.training.train --fold ... --seed ... ...

All raw inputs are read from the canonical data/lattice/raw/ directory, so
extending the raw side once feeds both panel versions.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.lattice.data.build_panel import LatticePhase1Config, build_phase1
from src.lattice.data.build_macro import build_macro_state
from src.lattice.data.folds import fold_indices
from src.lattice.training.standardise import fit_fold_scaler, save_fold_scaler


AUG_PROCESSED = Path("data/lattice_aug/processed")
AUG_SCALER_ROOT = Path("experiments/lattice_aug")


def _has_extended_raw() -> bool:
    """Return True if the extended raw (2015-2025) is available."""
    return (Path("data/lattice/raw/prices_sp500.parquet").exists()
            and Path("data/lattice/raw/sp500_constituents_pit.parquet").exists())


def main(panel_end: str = "2025-12-31") -> None:
    AUG_PROCESSED.mkdir(parents=True, exist_ok=True)
    AUG_SCALER_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"[panel_aug] rebuilding panel into {AUG_PROCESSED}", flush=True)
    cfg = LatticePhase1Config(
        raw_dir=Path("data/lattice/raw"),
        out_dir=AUG_PROCESSED,
        panel_start="2015-01-09",
        panel_end=panel_end,
    )
    summary = build_phase1(cfg)
    print(f"[panel_aug] phase1 summary: rows={summary['panel_rows']:,} "
          f"tickers={summary['tickers']} dates={summary['dates']} "
          f"features={summary['n_features']}", flush=True)

    print(f"[panel_aug] rebuilding macro state at {AUG_PROCESSED}", flush=True)
    build_macro_state(
        panel_start="2015-01-09",
        panel_end=panel_end,
        out_path=AUG_PROCESSED / "macro_state.parquet",
    )

    panel = pd.read_parquet(AUG_PROCESSED / "panel_features.parquet")
    cohorts = pd.read_parquet(AUG_PROCESSED / "cohorts.parquet")
    panel["date"] = pd.to_datetime(panel["date"])
    cohorts["date"] = pd.to_datetime(cohorts["date"])
    dates = sorted(panel["date"].unique())
    print(f"[panel_aug] panel: {len(panel):,} rows, "
          f"{panel.ticker.nunique()} tickers, {len(dates)} dates", flush=True)

    target_folds = (1, 2, 3, 4, 5) if panel_end >= "2024-01-01" else (1, 2, 3)
    for fold in target_folds:
        train_idx, val_idx, test_idx = fold_indices(fold, dates)
        scaler = fit_fold_scaler(panel, train_idx, cohorts, fold)
        save_fold_scaler(
            scaler, AUG_SCALER_ROOT / f"fold{fold}/scalers.pkl",
        )
        print(f"  fold {fold}: train_days={len(train_idx)} "
              f"val_days={len(val_idx)} test_days={len(test_idx)} "
              f"panel_z={len(scaler.panel_zscore_stats)} "
              f"sector_z={len(scaler.sector_zscore_stats)}", flush=True)
    print(f"[panel_aug] DONE; augmented panel at {AUG_PROCESSED}", flush=True)


if __name__ == "__main__":
    main()
