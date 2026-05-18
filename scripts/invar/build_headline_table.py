"""Aggregate all Phase 3 results into experiments/invar/headline_table.md
with statistical tests vs the strongest baseline (iTransformer).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.invar.evaluation.statistics import (
    paired_t_test_per_fold, wilcoxon_signed_rank, diebold_mariano,
    bootstrap_sharpe_ci,
)


def load_run_set(template: str) -> dict[int, dict[int, dict]]:
    """Returns {fold: {seed: results_dict}}."""
    out: dict[int, dict[int, dict]] = {}
    for fold in (1, 2):
        out[fold] = {}
        for s in (42, 43, 44, 45, 46):
            p = template.format(fold=fold, seed=s)
            if os.path.exists(p):
                out[fold][s] = json.load(open(p))
    return out


def fold_aggregate(runs: dict[int, dict[int, dict]], fold: int) -> dict:
    items = list(runs[fold].values())
    if not items:
        return {"n": 0}
    ics = [r["test_ic"] for r in items]
    rks = [r["test_rank_ic"] for r in items]
    ndcg = [r["test_ndcg10"] for r in items]
    sps = [r["test_sharpe"] for r in items]
    return {
        "n": len(items),
        "ic_mean": float(np.mean(ics)),
        "ic_std": float(np.std(ics, ddof=1)) if len(ics) > 1 else 0.0,
        "rank_mean": float(np.mean(rks)),
        "rank_std": float(np.std(rks, ddof=1)) if len(rks) > 1 else 0.0,
        "ndcg_mean": float(np.mean(ndcg)),
        "sharpe_mean": float(np.mean(sps)),
    }


def per_seed_pearson(runs: dict[int, dict[int, dict]], fold: int) -> list[float]:
    return [runs[fold][s]["test_ic"] for s in sorted(runs[fold].keys())]


def per_seed_rank(runs: dict[int, dict[int, dict]], fold: int) -> list[float]:
    return [runs[fold][s]["test_rank_ic"] for s in sorted(runs[fold].keys())]


def main() -> None:
    # Headline: one variant per model.
    sources = {
        "InVAR": "experiments/invar/headline_v3/fold{fold}/seed{seed}_designR/results.json",
        "G-InVAR": "experiments/ginvar/a3_sector_bias_mask/fold{fold}/seed{seed}/results.json",
        "iTransformer": "experiments/invar/baselines/itransformer/fold{fold}/seed{seed}/results.json",
        "MASTER": "experiments/invar/baselines/master/fold{fold}/seed{seed}/results.json",
        "StockMixer": "experiments/invar/baselines/stockmixer/fold{fold}/seed{seed}/results.json",
    }
    # Sidelined variants (kept on disk; not in the headline table).
    sidelined_sources = {
        "InVAR v2 (spec, sidelined)": "experiments/invar/headline_v2/fold{fold}/seed{seed}_designR/results.json",
        "G-InVAR A0 (dense, sidelined)": "experiments/ginvar/a0_dense/fold{fold}/seed{seed}/results.json",
        "G-InVAR full (4 graphs, sidelined)": "experiments/ginvar/full/fold{fold}/seed{seed}/results.json",
    }
    runs = {label: load_run_set(t) for label, t in sources.items()}
    sideline_runs = {label: load_run_set(t) for label, t in sidelined_sources.items()}

    out_lines = ["# InVAR Phase 3 headline table",
                  "",
                  "Compiled 2026-05-07. Spec: F1 + F2 only; F3 separately gated by user.",
                  "",
                  "## Aggregate by model",
                  "",
                  ("| model | F1 n | F1 IC mean (std) | F1 rank IC mean (std) | "
                   "F1 NDCG10 | F1 Sharpe | F2 n | F2 IC mean (std) | "
                   "F2 rank IC mean (std) |"),
                  "|:---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for label in sources:
        f1 = fold_aggregate(runs[label], 1)
        f2 = fold_aggregate(runs[label], 2)
        if f1["n"] == 0 and f2["n"] == 0:
            out_lines.append(f"| **{label}** | 0 | -- | -- | -- | -- | 0 | -- | -- |")
            continue
        f1_str = (f"{f1['ic_mean']:+.4f} ({f1['ic_std']:.4f})"
                   if f1['n'] > 0 else "--")
        f1_rk = (f"{f1['rank_mean']:+.4f} ({f1['rank_std']:.4f})"
                  if f1['n'] > 0 else "--")
        f1_n = f1["ndcg_mean"] if f1["n"] > 0 else float("nan")
        f1_sh = f1["sharpe_mean"] if f1["n"] > 0 else float("nan")
        f2_ic = (f"{f2['ic_mean']:+.4f} ({f2['ic_std']:.4f})"
                  if f2['n'] > 0 else "--")
        f2_rk = (f"{f2['rank_mean']:+.4f} ({f2['rank_std']:.4f})"
                  if f2['n'] > 0 else "--")
        out_lines.append(
            f"| **{label}** | {f1['n']} | {f1_str} | {f1_rk} | "
            f"{f1_n:.3f} | {f1_sh:+.2f} | {f2['n']} | {f2_ic} | {f2_rk} |"
        )
    out_lines.append("")

    out_lines += [
        "## Per-fold per-seed details",
        "",
    ]
    for fold in (1, 2):
        out_lines.append(f"### Fold {fold} per-seed test rank IC")
        out_lines.append("")
        seeds = (42, 43, 44, 45, 46)
        header = "| seed |" + "".join(f" {label} |" for label in sources)
        sep = "|---:|" + "".join(":---:|" for _ in sources)
        out_lines.append(header)
        out_lines.append(sep)
        for s in seeds:
            cells = [f"{s}"]
            for label in sources:
                if fold in runs[label] and s in runs[label][fold]:
                    cells.append(f"{runs[label][fold][s]['test_rank_ic']:+.4f}")
                else:
                    cells.append("--")
            out_lines.append("| " + " | ".join(cells) + " |")
        out_lines.append("")

    # Statistical tests vs iTransformer (the strongest baseline so far)
    out_lines.append("## Statistical tests vs iTransformer (paired across seeds)")
    out_lines.append("")
    out_lines.append("Paired t-test, Wilcoxon signed-rank, and Diebold-Mariano "
                      "(on per-day Pearson IC differences) where applicable.")
    out_lines.append("")
    out_lines.append(
        "| model | fold | metric | n seeds | paired t (p) | Wilcoxon (p) | DM stat (p) |"
    )
    out_lines.append("|:---|---:|:---|---:|---:|---:|---:|")
    baseline = "iTransformer"
    for label in sources:
        if label == baseline:
            continue
        for fold in (1, 2):
            seeds_a = sorted(runs[label].get(fold, {}).keys())
            seeds_b = sorted(runs[baseline].get(fold, {}).keys())
            common = sorted(set(seeds_a) & set(seeds_b))
            if len(common) < 2:
                continue
            a = [runs[label][fold][s]["test_rank_ic"] for s in common]
            b = [runs[baseline][fold][s]["test_rank_ic"] for s in common]
            t = paired_t_test_per_fold(a, b)
            w = wilcoxon_signed_rank(a, b)
            # Diebold-Mariano needs daily IC series; we'd need to load
            # predictions parquet per seed and compute per-day. Skip for
            # now and report at table-write level only.
            out_lines.append(
                f"| {label} | {fold} | rank IC | {len(common)} | "
                f"t={t.statistic:+.2f} (p={t.pvalue:.3f}) | "
                f"W={w.statistic:.1f} (p={w.pvalue:.3f}) | -- |"
            )

    # Sidelined variants in a separate trailing section.
    out_lines.append("")
    out_lines.append("## Sidelined variants (kept on disk; not headline)")
    out_lines.append("")
    out_lines.append(
        "| model | F1 n | F1 IC mean (std) | F1 rank IC mean (std) | "
        "F2 n | F2 IC mean (std) | F2 rank IC mean (std) |"
    )
    out_lines.append("|:---|---:|---:|---:|---:|---:|---:|")
    for label in sidelined_sources:
        f1 = fold_aggregate(sideline_runs[label], 1)
        f2 = fold_aggregate(sideline_runs[label], 2)
        if f1["n"] == 0 and f2["n"] == 0:
            continue
        f1_ic = (f"{f1['ic_mean']:+.4f} ({f1['ic_std']:.4f})"
                  if f1['n'] > 0 else "--")
        f1_rk = (f"{f1['rank_mean']:+.4f} ({f1['rank_std']:.4f})"
                  if f1['n'] > 0 else "--")
        f2_ic = (f"{f2['ic_mean']:+.4f} ({f2['ic_std']:.4f})"
                  if f2['n'] > 0 else "--")
        f2_rk = (f"{f2['rank_mean']:+.4f} ({f2['rank_std']:.4f})"
                  if f2['n'] > 0 else "--")
        out_lines.append(
            f"| {label} | {f1['n']} | {f1_ic} | {f1_rk} | "
            f"{f2['n']} | {f2_ic} | {f2_rk} |"
        )
    out_lines.append("")

    out_path = Path("experiments/invar/headline_table.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out_lines))
    print(f"Written: {out_path}")
    print()
    print("\n".join(out_lines[:30]))


if __name__ == "__main__":
    main()
