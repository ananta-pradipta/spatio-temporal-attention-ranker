"""Phase 3 diagnostics (a) ensemble check and (b) bank-usage audit.

(a) Ensemble: average y_hat across InVAR-C v2 and iTransformer per (date,
    ticker) on F1 test. Report ensemble Pearson IC, rank IC, and NDCG@10
    vs each individual model.

(b) Bank usage: load each v2 ckpt, run forward on F1 test split with the
    K used at best epoch (best_k from ckpt), capture
    model.regime_axis.last_top_idx per day, compute the fraction of the
    1024 bank entries that have nonzero usage. If that fraction is below
    50/1024 the bank is collapsed.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.invar.data.dataset import InvarDataset, PANEL_FEATURE_DIM, MACRO_FEATURE_DIM
from src.invar.model.invar import Invar, InvarConfig
from src.invar.evaluation.metrics import daily_ic, daily_rank_ic, ndcg_at_k


# ---------- (a) ensemble ----------

def diagnostic_ensemble(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_per_seed = []
    rows_ensemble = []
    for s in (42, 43, 44, 45, 46):
        invar_path = Path(f"experiments/invar/headline_v2/fold1/seed{s}_designR/predictions.parquet")
        itrans_path = Path(f"experiments/invar/baselines/itransformer/fold1/seed{s}/predictions.parquet")
        if not invar_path.exists() or not itrans_path.exists():
            print(f"missing predictions for seed {s}; skipping")
            continue
        a = pd.read_parquet(invar_path).rename(columns={"y_hat": "y_hat_invar"})
        b = pd.read_parquet(itrans_path).rename(columns={"y_hat": "y_hat_itrans"})
        merged = pd.merge(
            a[["date", "ticker", "y_hat_invar", "y_true"]],
            b[["date", "ticker", "y_hat_itrans"]],
            on=["date", "ticker"], how="inner",
        )
        # Per-day rank-normalise each model's y_hat before averaging so the
        # different output scales (NDCG-strong InVAR vs IC-strong iTrans)
        # do not give one model dominant weight in the ensemble.
        def per_day_rank_norm(col_in: str) -> pd.Series:
            return (merged.groupby("date")[col_in]
                          .rank(pct=True))
        merged["rank_invar"] = per_day_rank_norm("y_hat_invar")
        merged["rank_itrans"] = per_day_rank_norm("y_hat_itrans")
        merged["y_hat_ens"] = 0.5 * merged["rank_invar"] + 0.5 * merged["rank_itrans"]

        for label, col in (("invar", "y_hat_invar"),
                            ("itrans", "y_hat_itrans"),
                            ("ensemble", "y_hat_ens")):
            df = merged.rename(columns={col: "y_hat"})
            ic = daily_ic(df)
            rank = daily_rank_ic(df)
            ndcg = ndcg_at_k(df, k=10)
            rows_per_seed.append({
                "seed": s, "model": label,
                "ic": ic["mean"], "rank_ic": rank["mean"],
                "ndcg10": ndcg["mean"], "n_days": ic["n_days"],
            })

    df = pd.DataFrame(rows_per_seed)
    print("\n=== (a) Ensemble F1 test per seed ===\n")
    print(df.to_string(index=False))
    print()
    by_model = df.groupby("model").agg(
        ic_mean=("ic", "mean"),
        ic_std=("ic", "std"),
        rank_mean=("rank_ic", "mean"),
        rank_std=("rank_ic", "std"),
        ndcg10_mean=("ndcg10", "mean"),
    ).reset_index()
    print("\n=== Aggregate (5 seeds) ===\n")
    print(by_model.to_string(index=False))

    df.to_csv(out_dir / "ensemble_per_seed.csv", index=False)
    by_model.to_csv(out_dir / "ensemble_aggregate.csv", index=False)


# ---------- (b) bank usage audit ----------

def diagnostic_bank_usage(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    test_ds = InvarDataset(fold=1, split="test")
    rows = []
    for s in (42, 43, 44, 45, 46):
        ckpt_path = Path(f"experiments/invar/headline_v2/fold1/seed{s}_designR/ckpt.pt")
        if not ckpt_path.exists():
            print(f"missing ckpt for seed {s}")
            continue
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            state = ckpt["model_state"]
            best_k = ckpt.get("best_k", 32) or 32
        else:
            state = ckpt
            best_k = 32
        cfg = InvarConfig(
            n_features=PANEL_FEATURE_DIM, macro_dim=MACRO_FEATURE_DIM,
            regime_axis="retrieval", bank_size=1024,
            use_scalar_gate=False, zero_init_cross_attn=False,
        )
        model = Invar(cfg)
        model.load_state_dict(state)
        model.eval()
        model.regime_axis.set_top_k(int(best_k))
        usage = np.zeros(cfg.bank_size, dtype=np.int64)
        n_days = 0
        with torch.no_grad():
            for t in test_ds._eligible_idx:
                batch = test_ds.get(int(t))
                _ = model(batch.features, batch.macro, batch.mask, return_attn=False)
                idx = model.regime_axis.last_top_idx
                if idx is None:
                    continue
                for i in idx.cpu().numpy():
                    usage[int(i)] += 1
                n_days += 1
        unique_used = int((usage > 0).sum())
        max_used = int(usage.max())
        top10 = sorted(usage.tolist(), reverse=True)[:10]
        rows.append({
            "seed": s, "best_k": best_k, "n_test_days": n_days,
            "unique_bank_entries_used": unique_used,
            "fraction_of_1024_used": float(unique_used / cfg.bank_size),
            "max_per_entry_count": max_used,
            "top10_counts": top10,
        })
    df = pd.DataFrame(rows)
    print("\n=== (b) F1 test bank usage (per seed) ===\n")
    print(df.to_string(index=False))
    if not df.empty:
        df.to_csv(out_dir / "bank_usage.csv", index=False)


def main() -> None:
    out_dir = Path("experiments/invar/diagnostics_phase3")
    diagnostic_ensemble(out_dir)
    diagnostic_bank_usage(out_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
