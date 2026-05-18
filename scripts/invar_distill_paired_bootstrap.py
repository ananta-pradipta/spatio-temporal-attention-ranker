"""Paired bootstrap: DISTILL-InVAR vs MASTER and vs SWA-InVAR baseline.

Reuses the bootstrap machinery from
`scripts.invar_swa_paired_bootstrap_extended` but adds DISTILL as the
treatment arm. Reports per-day rank IC, NDCG@10, and LS decile return,
each with paired-by-date bootstrap p-values for both comparisons.

Usage:

    PYTHONPATH=$PWD python3 -m scripts.invar_distill_paired_bootstrap
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.invar_swa_paired_bootstrap_extended import (
    SEEDS, FOLDS, N_BOOTSTRAP, BLOCK_SIZE,
    per_day_ndcg, per_day_ls_return, load_per_day_metric,
    paired_block_bootstrap_p,
)
import scipy.stats as st


def per_day_rank_ic_df(df: pd.DataFrame) -> pd.Series:
    rows = []
    for date, sub in df.groupby("date"):
        sub = sub[sub["y_true"] != 0]
        if len(sub) < 5:
            continue
        rho, _ = st.spearmanr(sub["y_hat"], sub["y_true"])
        rows.append({"date": date, "rho": float(rho)})
    return pd.DataFrame(rows).set_index("date")["rho"]


def run_pair(label: str, treat_root: Path, ctrl_root: Path,
              treat_dirs: dict, ctrl_dirs: dict, metric_fn, metric_col: str) -> None:
    print("=" * 78)
    print(f"{label}")
    print(f"  N_bootstrap = {N_BOOTSTRAP}, block_size = {BLOCK_SIZE} days")
    print("=" * 78)
    pooled_diffs = []
    for fold in FOLDS:
        treat = load_per_day_metric(treat_root, fold, treat_dirs, metric_fn, metric_col)
        ctrl = load_per_day_metric(ctrl_root, fold, ctrl_dirs, metric_fn, metric_col)
        if treat is None or ctrl is None:
            print(f"Fold {fold}: missing predictions")
            continue
        td = treat.groupby("date")[metric_col].mean()
        cd = ctrl.groupby("date")[metric_col].mean()
        common = td.index.intersection(cd.index)
        if len(common) < 30:
            continue
        diffs = (td.loc[common] - cd.loc[common]).to_numpy()
        mean_obs, p, sd = paired_block_bootstrap_p(diffs, BLOCK_SIZE, N_BOOTSTRAP)
        pooled_diffs.extend(diffs.tolist())
        print(
            f"\nFold {fold}: treat {td.loc[common].mean():+.4f} | "
            f"ctrl {cd.loc[common].mean():+.4f} | "
            f"diff {mean_obs:+.4f} | p={p:.4f}"
        )
    if pooled_diffs:
        pooled = np.asarray(pooled_diffs)
        m, p, sd = paired_block_bootstrap_p(pooled, BLOCK_SIZE, N_BOOTSTRAP)
        print(f"\nPooled ({len(pooled)} d): diff {m:+.4f}  p={p:.4f}  sd={sd:.4f}\n")


def main() -> None:
    seed_designR = {s: f"seed{s}_designR" for s in SEEDS}
    seed_master = {s: f"seed{s}" for s in SEEDS}

    distill_root = Path("experiments/invar/distill")
    swa_root = Path("experiments/invar/swa")
    master_root = Path("experiments/invar/baselines/master")

    for label, fn, col in [
        ("Per-day rank IC", per_day_rank_ic_df, "rho"),
        ("Per-day NDCG@10", per_day_ndcg, "ndcg"),
        ("Per-day LS return", per_day_ls_return, "ls"),
    ]:
        run_pair(
            f"{label}: DISTILL vs MASTER",
            distill_root, master_root, seed_designR, seed_master, fn, col,
        )
        run_pair(
            f"{label}: DISTILL vs SWA-baseline",
            distill_root, swa_root, seed_designR, seed_designR, fn, col,
        )


if __name__ == "__main__":
    main()
