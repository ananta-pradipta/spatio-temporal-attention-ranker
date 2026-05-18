"""Cell E (Solution C): InVAR-v4 + MASTER per-day rank-average ensemble.

Reads predictions parquet from v4 and MASTER, rank-normalises each
model's y_hat per day, averages, and reports IC/rank-IC/NDCG10/Sharpe
on F1 + F2.

Inference-only; no retraining.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.invar.evaluation.metrics import (
    daily_ic, daily_rank_ic, ndcg_at_k, long_short_sharpe,
)


def load_preds(path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def per_day_rank_norm(df: pd.DataFrame, col: str) -> pd.Series:
    return df.groupby("date")[col].rank(pct=True)


def run() -> None:
    out_dir = Path("experiments/invar/v4_phase4/cellE_ensemble")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows_per_seed: list[dict] = []
    for fold in (1, 2):
        for s in (42, 43, 44, 45, 46):
            v4 = Path(f"experiments/invar/v4/fold{fold}/seed{s}_designR/predictions.parquet")
            ms = Path(f"experiments/invar/baselines/master/fold{fold}/seed{s}/predictions.parquet")
            if not (v4.exists() and ms.exists()):
                continue
            a = load_preds(v4).rename(columns={"y_hat": "yh_v4"})
            b = load_preds(ms).rename(columns={"y_hat": "yh_master"})
            merged = pd.merge(
                a[["date", "ticker", "yh_v4", "y_true", "sector_id",
                   "size_decile", "age_bucket"]],
                b[["date", "ticker", "yh_master"]],
                on=["date", "ticker"], how="inner",
            )
            merged["rank_v4"] = per_day_rank_norm(merged, "yh_v4")
            merged["rank_master"] = per_day_rank_norm(merged, "yh_master")
            merged["y_hat_ens"] = 0.5 * merged["rank_v4"] + 0.5 * merged["rank_master"]
            for label, col in (("v4", "yh_v4"),
                                ("master", "yh_master"),
                                ("ensemble", "y_hat_ens")):
                df = merged.rename(columns={col: "y_hat"})
                ic = daily_ic(df)
                rk = daily_rank_ic(df)
                ndcg = ndcg_at_k(df, k=10)
                sh = long_short_sharpe(df)
                rows_per_seed.append({
                    "fold": fold, "seed": s, "model": label,
                    "ic": ic["mean"], "rank_ic": rk["mean"],
                    "ndcg10": ndcg["mean"], "sharpe": sh["sharpe"],
                    "n_days": ic["n_days"],
                })

    df = pd.DataFrame(rows_per_seed)
    df.to_csv(out_dir / "ensemble_per_seed.csv", index=False)
    print("\n=== Cell E (v4 + MASTER ensemble) per seed ===\n")
    print(df.to_string(index=False))

    print("\n=== Aggregate (5-seed per fold) ===\n")
    agg = df.groupby(["fold", "model"]).agg(
        ic_mean=("ic", "mean"), ic_std=("ic", "std"),
        rank_mean=("rank_ic", "mean"), rank_std=("rank_ic", "std"),
        ndcg_mean=("ndcg10", "mean"),
        sharpe_mean=("sharpe", "mean"),
    ).reset_index()
    agg.to_csv(out_dir / "ensemble_aggregate.csv", index=False)
    print(agg.to_string(index=False))


if __name__ == "__main__":
    run()
