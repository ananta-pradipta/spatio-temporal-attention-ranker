"""Extended paired bootstrap: SWA-InVAR vs MASTER on multiple metrics.

Per-day rank IC was already covered by `invar_swa_paired_bootstrap.py`.
This script extends to:
  - Per-day NDCG@10 (top-of-list ranking quality)
  - Per-day long-short decile return (financial utility)

Aggregation is per-day, paired-by-date, mean across seeds. Bootstrap is
moving-block bootstrap with replacement (block_size=10 days).

Usage:

    PYTHONPATH=$PWD python3 -m scripts.invar_swa_paired_bootstrap_extended
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st


SEEDS = [42, 43, 44, 45, 46]
FOLDS = [1, 2]
N_BOOTSTRAP = 5000
BLOCK_SIZE = 10


def per_day_ndcg(df: pd.DataFrame, k: int = 10) -> pd.Series:
    """Per-day NDCG@k. df has columns: date, y_hat, y_true (active rows only).

    Inactive rows (y_true == 0) are dropped before scoring.
    """
    rows = []
    for date, sub in df.groupby("date"):
        sub = sub[sub["y_true"] != 0].copy()
        if len(sub) < k:
            continue
        sub = sub.sort_values("y_hat", ascending=False)
        gains = np.maximum(sub["y_true"].to_numpy(), 0.0)  # only positive returns count
        idx = np.arange(len(gains))
        discount = 1.0 / np.log2(idx + 2.0)
        dcg = float((gains[:k] * discount[:k]).sum())
        # ideal: sort gains desc
        ideal_gains = np.sort(gains)[::-1]
        idcg = float((ideal_gains[:k] * discount[:k]).sum())
        ndcg = dcg / idcg if idcg > 0 else 0.0
        rows.append({"date": date, "ndcg": ndcg})
    return pd.DataFrame(rows).set_index("date")["ndcg"]


def per_day_ls_return(df: pd.DataFrame, q: int = 10) -> pd.Series:
    """Per-day long-short return: long top-decile, short bottom-decile, equal-weight.

    df has columns: date, y_hat, y_true. Returns daily LS return as
    mean(y_true | top-decile by y_hat) - mean(y_true | bottom-decile by y_hat).
    """
    rows = []
    for date, sub in df.groupby("date"):
        sub = sub[sub["y_true"] != 0].copy()
        if len(sub) < q * 2:
            continue
        sub = sub.sort_values("y_hat")
        n_dec = max(len(sub) // q, 1)
        bot = sub.iloc[:n_dec]
        top = sub.iloc[-n_dec:]
        ret = float(top["y_true"].mean() - bot["y_true"].mean())
        rows.append({"date": date, "ls": ret})
    return pd.DataFrame(rows).set_index("date")["ls"]


def load_per_day_metric(
    base_dir: Path, fold: int, seed_dirs: dict, metric_fn, metric_col: str,
) -> pd.DataFrame | None:
    """Load and concat per-day metric for a (model, fold) across seeds."""
    out = []
    for seed in SEEDS:
        sub = seed_dirs.get(seed)
        if sub is None:
            return None
        path = base_dir / f"fold{fold}/{sub}/predictions.parquet"
        if not path.exists():
            return None
        df = pd.read_parquet(path)
        ser = metric_fn(df).rename(metric_col)
        out.append(ser.reset_index().assign(seed=seed))
    return pd.concat(out, ignore_index=True)


def paired_block_bootstrap_p(
    diffs: np.ndarray, block_size: int = 10, n_bootstrap: int = 5000,
    rng_seed: int = 0,
) -> tuple[float, float, float]:
    """One-sided paired block bootstrap p-value for H1: mean(diffs) > 0."""
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
    boot_centered = boot_means - obs_mean
    p_one_sided = float(np.mean(boot_centered >= obs_mean))
    return obs_mean, p_one_sided, float(np.std(boot_means))


def run_metric(
    label: str, swa_root: Path, mast_root: Path,
    swa_seed_dirs: dict, master_seed_dirs: dict, metric_fn,
    metric_col: str,
) -> None:
    print("=" * 70)
    print(f"Metric: {label}")
    print(f"  N_bootstrap = {N_BOOTSTRAP}, block_size = {BLOCK_SIZE} days")
    print("=" * 70)

    pooled_diffs = []
    for fold in FOLDS:
        swa = load_per_day_metric(swa_root, fold, swa_seed_dirs, metric_fn, metric_col)
        mast = load_per_day_metric(mast_root, fold, master_seed_dirs, metric_fn, metric_col)
        if swa is None or mast is None:
            print(f"Fold {fold}: missing predictions, skipping")
            continue
        swa_daily = swa.groupby("date")[metric_col].mean()
        mast_daily = mast.groupby("date")[metric_col].mean()
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
            f"\n=== Fold {fold} ({len(common)} eval days) ===\n"
            f"  Mean SWA {metric_col}:    {swa_daily.loc[common].mean():+.4f}\n"
            f"  Mean MASTER {metric_col}: {mast_daily.loc[common].mean():+.4f}\n"
            f"  Mean diff:                {mean_obs:+.4f}\n"
            f"  Bootstrap std of diff:    {sd:.4f}\n"
            f"  One-sided p-value (H1: SWA > MASTER): {p:.4f}"
        )

    if pooled_diffs:
        pooled = np.asarray(pooled_diffs)
        mean_obs, p, sd = paired_block_bootstrap_p(
            pooled, BLOCK_SIZE, N_BOOTSTRAP,
        )
        print(
            f"\n=== Pooled (F1+F2, {len(pooled_diffs)} days) ===\n"
            f"  Mean diff:                {mean_obs:+.4f}\n"
            f"  Bootstrap std of diff:    {sd:.4f}\n"
            f"  One-sided p-value:        {p:.4f}\n"
        )


def main() -> None:
    swa_seed_dirs = {s: f"seed{s}_designR" for s in SEEDS}
    master_seed_dirs = {s: f"seed{s}" for s in SEEDS}
    swa_root = Path("experiments/invar/swa")
    mast_root = Path("experiments/invar/baselines/master")

    run_metric(
        "Per-day NDCG@10",
        swa_root, mast_root, swa_seed_dirs, master_seed_dirs,
        per_day_ndcg, "ndcg",
    )

    run_metric(
        "Per-day long-short decile return",
        swa_root, mast_root, swa_seed_dirs, master_seed_dirs,
        per_day_ls_return, "ls",
    )


if __name__ == "__main__":
    main()
