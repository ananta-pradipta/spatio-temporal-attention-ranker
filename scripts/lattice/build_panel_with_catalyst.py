"""Build a Universal Ticker panel with populated catalyst features at a
sibling directory, leaving the canonical panel and the current model code
untouched.

Solution G from the 2026-05-12 F2 deep-dive: the 3 catalyst columns
(days_to_next_catalyst_sin / cos / catalyst_type_id) in the canonical
panel are zero-filled. Populating them from the earnings calendar is a
data-only improvement; the architecture reads those columns identically.

This script:
  1. Creates `data/lattice_catalyst/processed/`.
  2. Copies the canonical panel + cohorts + active_mask + macro_state.
  3. Recomputes the catalyst features against the canonical panel's grid
     using the existing build_catalyst_features.py logic, with
     --update-panel pointed at the new copy so only the new panel gets
     the catalyst columns populated.
  4. Fits per-fold scalers (F1-F5) for the new panel at
     `experiments/lattice_catalyst/foldX/scalers.pkl`.

After this script finishes the catalyst-populated sweep is selected by:

    export LATTICE_PROCESSED_DIR=data/lattice_catalyst/processed
    export LATTICE_SCALER_DIR=experiments/lattice_catalyst
    python -m src.v2.training.train_dow_epistar \
        --config configs/rag_star_universe_v2.yaml --fold ... --seed ...

The canonical `data/lattice/processed/` panel is not modified.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pandas as pd

from src.lattice.data.folds import fold_indices
from src.lattice.training.standardise import fit_fold_scaler, save_fold_scaler


SRC_DIR = Path("data/lattice/processed")
DST_DIR = Path("data/lattice_catalyst/processed")
SCALER_ROOT = Path("experiments/lattice_catalyst")


def main() -> None:
    DST_DIR.mkdir(parents=True, exist_ok=True)
    SCALER_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"[catalyst panel] copying canonical artifacts to {DST_DIR}", flush=True)
    for name in ("panel_features.parquet", "cohorts.parquet",
                 "active_mask.parquet", "macro_state.parquet"):
        src = SRC_DIR / name
        dst = DST_DIR / name
        if not src.exists():
            raise FileNotFoundError(f"missing canonical artifact: {src}")
        shutil.copyfile(src, dst)
        print(f"  {src} -> {dst}", flush=True)

    print(f"[catalyst panel] running build_catalyst_features with "
          f"--update-panel on the catalyst copy", flush=True)
    subprocess.run(
        [
            "python3", "-u", "-m",
            "scripts.lattice.build_catalyst_features",
            "--calendar", "data/lattice/raw/earnings_calendar.parquet",
            "--panel", str(DST_DIR / "panel_features.parquet"),
            "--out", str(DST_DIR / "catalyst_features.parquet"),
            "--coverage-out",
            str(SCALER_ROOT / "phase_c_earnings_coverage.md"),
            "--update-panel",
        ],
        check=True,
    )

    print(f"[catalyst panel] fitting per-fold scalers to {SCALER_ROOT}",
          flush=True)
    panel = pd.read_parquet(DST_DIR / "panel_features.parquet")
    cohorts = pd.read_parquet(DST_DIR / "cohorts.parquet")
    panel["date"] = pd.to_datetime(panel["date"])
    cohorts["date"] = pd.to_datetime(cohorts["date"])
    dates = sorted(panel["date"].unique())
    target_folds = (1, 2, 3, 4, 5) if max(dates) >= pd.Timestamp("2024-01-01") else (1, 2, 3)
    for fold in target_folds:
        train_idx, val_idx, test_idx = fold_indices(fold, dates)
        scaler = fit_fold_scaler(panel, train_idx, cohorts, fold)
        save_fold_scaler(scaler, SCALER_ROOT / f"fold{fold}/scalers.pkl")
        print(f"  fold {fold}: train={len(train_idx)} val={len(val_idx)} "
              f"test={len(test_idx)}", flush=True)

    print(f"[catalyst panel] DONE; new panel at {DST_DIR}", flush=True)


if __name__ == "__main__":
    main()
