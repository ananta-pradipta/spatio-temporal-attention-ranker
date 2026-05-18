"""Paired bootstrap on per-day rank IC: SWA-InVAR vs MASTER.

For each fold and seed, builds the per-day rank IC time series for
both models on the same eval days (paired). Then runs a paired
block bootstrap with replacement over the time series to estimate
the distribution of the (SWA - MASTER) per-day rank IC mean. The
output is a one-sided p-value for "SWA-InVAR significantly above
MASTER on pooled rank IC".

Usage:

    PYTHONPATH=$PWD python3 -m scripts.invar_swa_paired_bootstrap
"""
from __future__ import annotations

from pathlib import Path
import statistics

import numpy as np
import pandas as pd
import scipy.stats as st


SEEDS = [42, 43, 44, 45, 46]
FOLDS = [1, 2]
N_BOOTSTRAP = 5000
BLOCK_SIZE = 10                          # days; ~2 trading weeks


def load_per_seed_daily_ric(
    base_dir: Path, fold: int, seed_dirs: dict,
) -> pd.DataFrame | None:
    """Load and concat per-day rank IC for a (model, fold) across seeds."""
    out = []
    for seed in SEEDS:
        sub = seed_dirs.get(seed)
        if sub is None:
            return None
        path = base_dir / f"fold{fold}/{sub}/predictions.parquet"
        if not path.exists():
            return None
        df = pd.read_parquet(path)
        df = df[df["y_true"] != 0]                       # drop inactive (z-score = 0)
        rows = []
        for date, sub_df in df.groupby("date"):
            if len(sub_df) < 5:
                continue
            rho, _ = st.spearmanr(sub_df["y_hat"], sub_df["y_true"])
            rows.append({"date": date, "seed": seed, "rho": float(rho)})
        out.append(pd.DataFrame(rows))
    return pd.concat(out, ignore_index=True)


def paired_block_bootstrap_p(
    diffs: np.ndarray, block_size: int = 10, n_bootstrap: int = 5000,
    rng_seed: int = 0,
) -> tuple[float, float, float]:
    """One-sided paired block bootstrap p-value for H1: mean(diffs) > 0.

    Returns (mean_obs, p_one_sided, std_bootstrap).
    """
    rng = np.random.default_rng(rng_seed)
    T = len(diffs)
    n_blocks = (T + block_size - 1) // block_size
    obs_mean = float(np.mean(diffs))
    boot_means = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        starts = rng.integers(0, max(T - block_size + 1, 1), size=n_blocks)
        sample_idx = np.concatenate(
            [np.arange(s, min(s + block_size, T)) for s in starts]
        )[:T]
        boot_means[b] = float(np.mean(diffs[sample_idx]))
    # Center the bootstrap distribution at 0 to test H0: mean = 0.
    boot_centered = boot_means - obs_mean
    p_one_sided = float(np.mean(boot_centered >= obs_mean))
    return obs_mean, p_one_sided, float(np.std(boot_means))


def main() -> None:
    swa_seed_dirs = {s: f"seed{s}_designR" for s in SEEDS}
    master_seed_dirs = {s: f"seed{s}" for s in SEEDS}

    swa_root = Path("experiments/invar/swa")
    mast_root = Path("experiments/invar/baselines/master")

    print("=" * 70)
    print("Paired block bootstrap: SWA-InVAR vs MASTER on per-day rank IC")
    print(f"  N_bootstrap = {N_BOOTSTRAP}, block_size = {BLOCK_SIZE} days")
    print("=" * 70)

    pooled_diffs = []

    for fold in FOLDS:
        swa = load_per_seed_daily_ric(swa_root, fold, swa_seed_dirs)
        mast = load_per_seed_daily_ric(mast_root, fold, master_seed_dirs)
        if swa is None or mast is None:
            print(f"Fold {fold}: missing predictions, skipping")
            continue
        # Aggregate per-day rank IC across seeds (mean per (date, seed) -> mean per date)
        swa_daily = swa.groupby("date")["rho"].mean()
        mast_daily = mast.groupby("date")["rho"].mean()
        common = swa_daily.index.intersection(mast_daily.index)
        if len(common) < 30:
            print(f"Fold {fold}: only {len(common)} common days, skipping")
            continue
        diffs = (swa_daily.loc[common] - mast_daily.loc[common]).to_numpy()
        mean_obs, p, sd = paired_block_bootstrap_p(
            diffs, BLOCK_SIZE, N_BOOTSTRAP,
        )
        pooled_diffs.extend(diffs.tolist())
        print(
            f"\n=== Fold {fold} ({len(common)} eval days, paired-by-date) ===\n"
            f"  Mean SWA daily rank IC:    {swa_daily.loc[common].mean():+.4f}\n"
            f"  Mean MASTER daily rank IC: {mast_daily.loc[common].mean():+.4f}\n"
            f"  Mean diff (SWA - MASTER):  {mean_obs:+.4f}\n"
            f"  Bootstrap std of diff:     {sd:.4f}\n"
            f"  One-sided p-value (H1: SWA > MASTER): {p:.4f}\n"
        )

    if len(pooled_diffs) > 0:
        pooled = np.asarray(pooled_diffs)
        mean_obs, p, sd = paired_block_bootstrap_p(
            pooled, BLOCK_SIZE, N_BOOTSTRAP,
        )
        print(
            f"\n=== Pooled (F1 + F2, {len(pooled_diffs)} days total) ===\n"
            f"  Mean diff (SWA - MASTER):  {mean_obs:+.4f}\n"
            f"  Bootstrap std of diff:     {sd:.4f}\n"
            f"  One-sided p-value:         {p:.4f}\n"
        )


if __name__ == "__main__":
    main()
