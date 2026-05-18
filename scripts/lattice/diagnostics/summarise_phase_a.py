"""Phase 5b A: summarise gate trajectories, router utilisation, within-sector IC.

Reads the per-(fold, seed) JSON and parquet artifacts produced by
``extract_phase_a.py`` and writes four markdown documents:

  - ``experiments/lattice/diagnostics/phase_a_gate_summary.md``
  - ``experiments/lattice/diagnostics/phase_a_router_summary.md``
  - ``experiments/lattice/diagnostics/phase_a_within_sector_ic.md``
  - ``experiments/lattice/diagnostics/phase_a_verdict.md``

The acceptance gates (per Phase 5b spec section 4.4) drive the verdict
classification: Proceed, Proceed with caveat, or Stop and consult.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# ---- Verdict thresholds (from spec) ----
GATE_LOCKED_STD = 0.01
GATE_LOCKED_DRIFT = 0.02
GATE_MOVING_STD = 0.05
GATE_MOVING_DRIFT = 0.05

ROUTER_HEALTHY_ENTROPY = 1.0
ROUTER_HEALTHY_MAX_UTIL = 0.50
ROUTER_HEALTHY_MAX_DAY = 0.95

ROUTER_CONCENTRATED_UTIL_LO = 0.50
ROUTER_CONCENTRATED_UTIL_HI = 0.80

ROUTER_COLLAPSED_UTIL = 0.80
ROUTER_COLLAPSED_ENTROPY = 0.7
ROUTER_COLLAPSED_DAY_FRAC = 0.30

WITHIN_SECTOR_AGNOSTIC = 9
WITHIN_SECTOR_BYSTANDER = 4

INIT = {"alpha_blend": 0.5, "alpha_regime": 0.27, "alpha_novelty": 0.27, "lambda_macro": 0.05}


def gate_verdict(gate_name: str, mean: float, std: float, init_val: float) -> str:
    """Return Locked / Moving / Saturated / Mixed verdict for a gate."""
    if not np.isfinite(mean) or not np.isfinite(std):
        return "Unknown"
    if mean <= 0.001 or mean >= 0.999:
        return "Saturated"
    drift = abs(mean - init_val)
    if std < GATE_LOCKED_STD and drift < GATE_LOCKED_DRIFT:
        return "Locked"
    if std > GATE_MOVING_STD or drift > GATE_MOVING_DRIFT:
        return "Moving"
    return "Mixed"


def router_verdict(util: list[float], entropy: float, max_day_frac: float) -> str:
    """Return Healthy / Concentrated / Collapsed for one (fold, seed)."""
    if not util:
        return "Unknown"
    max_util = max(util)
    if max_util > ROUTER_COLLAPSED_UTIL or entropy < ROUTER_COLLAPSED_ENTROPY \
            or max_day_frac > ROUTER_COLLAPSED_DAY_FRAC:
        return "Collapsed"
    if max_util > ROUTER_CONCENTRATED_UTIL_LO:
        return "Concentrated"
    if entropy >= ROUTER_HEALTHY_ENTROPY and max_util <= ROUTER_HEALTHY_MAX_UTIL:
        return "Healthy"
    return "Mildly concentrated"


def write_gate_summary(diag_dir: Path, folds: list[int], seeds: list[int]) -> dict:
    """Aggregate gate trajectory JSONs and write phase_a_gate_summary.md."""
    rows = []
    raw: dict[tuple[int, int], dict] = {}
    for fold in folds:
        for seed in seeds:
            p = diag_dir / "gate_trajectories" / f"fold{fold}_seed{seed}.json"
            if not p.exists():
                continue
            data = json.loads(p.read_text())
            raw[(fold, seed)] = data
            for split in ("val", "test"):
                summary = data[split]["summary"]
                if not summary:
                    continue
                for gate_name in ("alpha_blend", "alpha_regime",
                                   "alpha_novelty_per_day_mean"):
                    s = summary[gate_name]
                    init_key = ("alpha_novelty" if "novelty" in gate_name
                                else gate_name)
                    init_val = INIT.get(init_key, float("nan"))
                    rows.append({
                        "fold": fold, "seed": seed, "split": split,
                        "gate": gate_name,
                        "init": init_val,
                        "mean": s["mean"], "std": s["std"],
                        "min": s["min"], "max": s["max"],
                        "verdict": gate_verdict(gate_name, s["mean"], s["std"], init_val),
                    })
                rows.append({
                    "fold": fold, "seed": seed, "split": split,
                    "gate": "lambda_macro",
                    "init": INIT["lambda_macro"],
                    "mean": data["lambda_macro"], "std": 0.0,
                    "min": data["lambda_macro"], "max": data["lambda_macro"],
                    "verdict": gate_verdict(
                        "lambda_macro", data["lambda_macro"], 0.0, INIT["lambda_macro"]),
                })
    df = pd.DataFrame(rows)

    out_path = diag_dir / "phase_a_gate_summary.md"
    lines = ["# Phase 5b A: gate trajectory summary", ""]
    lines.append("Init values: alpha_blend = 0.5, alpha_regime = 0.27, "
                  "alpha_novelty = 0.27, lambda_macro = 0.05.")
    lines.append("")
    lines.append("Verdicts: Locked = std < 0.01 AND |mean - init| < 0.02; "
                  "Moving = std > 0.05 OR |mean - init| > 0.05; "
                  "Saturated = mean approximately 0 or 1; otherwise Mixed.")
    lines.append("")
    if df.empty:
        lines.append("(no checkpoints found)")
    else:
        for fold in sorted(df["fold"].unique()):
            df_f = df[df["fold"] == fold]
            lines.append(f"## Fold {fold}")
            lines.append("")
            lines.append("| seed | split | gate | init | mean | std | min | max | verdict |")
            lines.append("|---:|:---|:---|---:|---:|---:|---:|---:|:---|")
            for _, r in df_f.iterrows():
                lines.append(
                    f"| {int(r['seed'])} | {r['split']} | {r['gate']} | "
                    f"{r['init']:.3f} | {r['mean']:+.4f} | {r['std']:.4f} | "
                    f"{r['min']:+.4f} | {r['max']:+.4f} | {r['verdict']} |"
                )
            lines.append("")

        agg = (df[df["split"] == "test"]
               .groupby(["fold", "gate"])
               .agg(mean_pool=("mean", "mean"),
                    std_pool=("std", "mean"),
                    seeds=("seed", "nunique"))
               .reset_index())
        lines.append("## Cross-seed test-split aggregate")
        lines.append("")
        lines.append("| fold | gate | seeds | mean (avg) | std (avg) |")
        lines.append("|---:|:---|---:|---:|---:|")
        for _, r in agg.iterrows():
            lines.append(
                f"| {int(r['fold'])} | {r['gate']} | {int(r['seeds'])} | "
                f"{r['mean_pool']:+.4f} | {r['std_pool']:.4f} |"
            )
        lines.append("")

        summary_per_gate_fold = {}
        for fold in sorted(df["fold"].unique()):
            for gate_name in ("alpha_blend", "alpha_regime",
                               "alpha_novelty_per_day_mean", "lambda_macro"):
                d_test = df[(df["fold"] == fold) & (df["gate"] == gate_name)
                              & (df["split"] == "test")]
                if d_test.empty:
                    continue
                verdicts = list(d_test["verdict"])
                from collections import Counter
                ctr = Counter(verdicts)
                top, _ = ctr.most_common(1)[0]
                summary_per_gate_fold[(fold, gate_name)] = {
                    "verdict_majority": top,
                    "verdict_counts": dict(ctr),
                }

        lines.append("## Verdict majority per (fold, gate) on test split")
        lines.append("")
        lines.append("| fold | gate | majority verdict | counts |")
        lines.append("|---:|:---|:---|:---|")
        for (fold, gate_name), v in summary_per_gate_fold.items():
            counts_str = ", ".join(f"{k}={v_}" for k, v_ in v["verdict_counts"].items())
            lines.append(
                f"| {fold} | {gate_name} | {v['verdict_majority']} | {counts_str} |"
            )
        lines.append("")

    out_path.write_text("\n".join(lines))
    return {"df": df, "raw": raw}


def write_router_summary(diag_dir: Path, folds: list[int], seeds: list[int]) -> dict:
    """Aggregate router utilisation JSONs and write phase_a_router_summary.md."""
    rows = []
    raw: dict[tuple[int, int], dict] = {}
    for fold in folds:
        for seed in seeds:
            p = diag_dir / "router_utilisation" / f"fold{fold}_seed{seed}.json"
            if not p.exists():
                continue
            data = json.loads(p.read_text())
            raw[(fold, seed)] = data
            for split in ("val", "test"):
                s = data[split]
                rows.append({
                    "fold": fold, "seed": seed, "split": split,
                    "n_days": s["n_days"],
                    "expert_util": s["expert_mean_utilisation"],
                    "max_util": (max(s["expert_mean_utilisation"])
                                  if s["expert_mean_utilisation"] else float("nan")),
                    "entropy": s["mean_routing_entropy"],
                    "max_day_frac": s.get("frac_days_argmax_top1", float("nan")),
                    "verdict": router_verdict(
                        s["expert_mean_utilisation"],
                        s["mean_routing_entropy"],
                        s.get("frac_days_argmax_top1", 0.0),
                    ),
                })
    df = pd.DataFrame(rows)

    out_path = diag_dir / "phase_a_router_summary.md"
    lines = ["# Phase 5b A: MoE router utilisation summary", ""]
    lines.append("4 experts, max entropy = log(4) approximately 1.386. "
                  "Init: balance loss weight 0.01, near-uniform routing at iter 0.")
    lines.append("")
    lines.append("Verdicts: Healthy = entropy >= 1.0 AND max util <= 0.50; "
                  "Concentrated = max util in (0.50, 0.80]; "
                  "Collapsed = max util > 0.80 OR entropy < 0.7 OR > 30% of days top-expert > 0.95.")
    lines.append("")
    if df.empty:
        lines.append("(no checkpoints found)")
    else:
        for fold in sorted(df["fold"].unique()):
            df_f = df[df["fold"] == fold]
            lines.append(f"## Fold {fold}")
            lines.append("")
            lines.append("| seed | split | n_days | exp0 | exp1 | exp2 | exp3 | "
                          "max_util | entropy | top1>.95 frac | verdict |")
            lines.append("|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|:---|")
            for _, r in df_f.iterrows():
                eu = r["expert_util"] + [float("nan")] * (4 - len(r["expert_util"]))
                lines.append(
                    f"| {int(r['seed'])} | {r['split']} | {int(r['n_days'])} | "
                    f"{eu[0]:.3f} | {eu[1]:.3f} | {eu[2]:.3f} | {eu[3]:.3f} | "
                    f"{r['max_util']:.3f} | {r['entropy']:.3f} | "
                    f"{r['max_day_frac']:.3f} | {r['verdict']} |"
                )
            lines.append("")

    out_path.write_text("\n".join(lines))
    return {"df": df, "raw": raw}


def write_within_sector(diag_dir: Path, folds: list[int], seeds: list[int]) -> dict:
    """Compute per-sector IC and write phase_a_within_sector_ic.md."""
    pred_dir = diag_dir / "predictions"
    frames = []
    for fold in folds:
        for seed in seeds:
            p = pred_dir / f"fold{fold}_seed{seed}.parquet"
            if p.exists():
                frames.append(pd.read_parquet(p))
    if not frames:
        out_path = diag_dir / "phase_a_within_sector_ic.md"
        out_path.write_text("# Phase 5b A: within-sector IC\n\n(no predictions found)\n")
        return {"df": pd.DataFrame()}
    preds = pd.concat(frames, ignore_index=True)

    rows = []
    rows_per_seed = []
    for fold in sorted(preds["fold"].unique()):
        df_f = preds[preds["fold"] == fold]
        for sector in sorted(df_f["sector_name"].unique()):
            d_sec = df_f[df_f["sector_name"] == sector]
            ics_per_seed = []
            for seed in sorted(d_sec["seed"].unique()):
                d_ss = d_sec[d_sec["seed"] == seed]
                ics_day = []
                for date in sorted(d_ss["date"].unique()):
                    d_day = d_ss[d_ss["date"] == date]
                    if len(d_day) < 5:
                        continue
                    if d_day["y_hat"].std() < 1e-9 or d_day["y_true"].std() < 1e-9:
                        continue
                    rho, _ = spearmanr(d_day["y_hat"], d_day["y_true"])
                    if np.isfinite(rho):
                        ics_day.append(float(rho))
                if ics_day:
                    seed_mean = float(np.mean(ics_day))
                    ics_per_seed.append(seed_mean)
                    rows_per_seed.append({
                        "fold": fold, "sector": sector, "seed": seed,
                        "n_days": len(ics_day),
                        "n_tickers_avg": int(d_ss.groupby("date").size().mean()),
                        "mean_rank_ic": seed_mean,
                    })
            if ics_per_seed:
                rows.append({
                    "fold": fold, "sector": sector,
                    "n_seeds": len(ics_per_seed),
                    "n_tickers_avg": int(d_sec.groupby(["seed", "date"])
                                           .size().groupby(level=1).mean().mean()),
                    "mean_rank_ic_pool": float(np.mean(ics_per_seed)),
                    "std_rank_ic_pool": (float(np.std(ics_per_seed))
                                          if len(ics_per_seed) > 1 else 0.0),
                })
    sec_df = pd.DataFrame(rows)
    seed_df = pd.DataFrame(rows_per_seed)

    out_path = diag_dir / "phase_a_within_sector_ic.md"
    lines = ["# Phase 5b A: within-sector IC", ""]
    lines.append("Per-sector mean Spearman rank IC computed across test days, "
                  "averaged across seeds. Pre-registered target: at least 9 of 11 "
                  "sectors with positive mean rank IC on Fold 1.")
    lines.append("")
    if sec_df.empty:
        lines.append("(no per-sector data)")
    else:
        for fold in sorted(sec_df["fold"].unique()):
            df_f = sec_df[sec_df["fold"] == fold].sort_values("mean_rank_ic_pool",
                                                                  ascending=False)
            n_pos = int((df_f["mean_rank_ic_pool"] > 0).sum())
            n_total = len(df_f)
            lines.append(f"## Fold {fold}: {n_pos}/{n_total} sectors positive")
            lines.append("")
            if fold == 1:
                if n_pos >= WITHIN_SECTOR_AGNOSTIC:
                    verdict = "Sector-agnostic on F1 (target met)"
                elif n_pos >= WITHIN_SECTOR_BYSTANDER:
                    verdict = (f"Concentrated ({n_pos}/{n_total} positive); "
                                "F1 success is partly cross-sector level dispersion")
                else:
                    verdict = (f"Bystander ({n_pos}/{n_total} positive); "
                                "F1 success is a level-dispersion artefact")
                lines.append(f"**F1 sector-agnostic verdict**: {verdict}.")
                lines.append("")
            lines.append("| sector | n seeds | tickers/day (avg) | mean rank IC (5-seed) | std (across seeds) |")
            lines.append("|:---|---:|---:|---:|---:|")
            for _, r in df_f.iterrows():
                lines.append(
                    f"| {r['sector']} | {int(r['n_seeds'])} | {int(r['n_tickers_avg'])} | "
                    f"{r['mean_rank_ic_pool']:+.4f} | {r['std_rank_ic_pool']:.4f} |"
                )
            lines.append("")
    out_path.write_text("\n".join(lines))
    return {"sector_df": sec_df, "seed_df": seed_df}


def write_verdict(diag_dir: Path, gate_out: dict, router_out: dict, sector_out: dict) -> str:
    """Roll-up verdict per Phase 5b spec section 4.4."""
    df_gate = gate_out["df"]
    df_rt = router_out["df"]
    sec_df = sector_out.get("sector_df", pd.DataFrame())

    out_path = diag_dir / "phase_a_verdict.md"
    lines = ["# Phase 5b A: roll-up verdict", ""]

    # 1. Gate-health paragraph
    if df_gate.empty:
        gate_para = "No gate trajectory data."
    else:
        d_test = df_gate[df_gate["split"] == "test"]
        verdict_counts = (d_test.groupby(["gate", "verdict"]).size()
                          .unstack(fill_value=0))
        moving = (
            int(verdict_counts.get("Moving", pd.Series()).sum())
            if "Moving" in verdict_counts.columns else 0
        )
        locked = (
            int(verdict_counts.get("Locked", pd.Series()).sum())
            if "Locked" in verdict_counts.columns else 0
        )
        saturated = (
            int(verdict_counts.get("Saturated", pd.Series()).sum())
            if "Saturated" in verdict_counts.columns else 0
        )
        mixed = (
            int(verdict_counts.get("Mixed", pd.Series()).sum())
            if "Mixed" in verdict_counts.columns else 0
        )
        gate_para = (f"Across {len(d_test)} (fold, seed, gate) cells on the test split: "
                     f"{moving} Moving, {locked} Locked, {mixed} Mixed, {saturated} Saturated.")

    lines.append("## Gate health")
    lines.append("")
    lines.append(gate_para)
    lines.append("")

    # 2. Router paragraph
    if df_rt.empty:
        router_para = "No router data."
    else:
        d_test = df_rt[df_rt["split"] == "test"]
        v_counts = d_test["verdict"].value_counts().to_dict()
        router_para = (f"Across {len(d_test)} (fold, seed) test-split cells: "
                       + ", ".join(f"{k}={v}" for k, v in v_counts.items()) + ".")
    lines.append("## Router health")
    lines.append("")
    lines.append(router_para)
    lines.append("")

    # 3. Within-sector paragraph
    if sec_df.empty:
        sector_para = "No within-sector data."
        f1_n_pos, f1_n_total = -1, -1
    else:
        f1 = sec_df[sec_df["fold"] == 1]
        f1_n_pos = int((f1["mean_rank_ic_pool"] > 0).sum()) if not f1.empty else 0
        f1_n_total = len(f1) if not f1.empty else 0
        sector_para = (f"Fold 1 within-sector IC: {f1_n_pos}/{f1_n_total} sectors positive. "
                       "Target for sector-agnostic claim: at least 9 of 11.")
    lines.append("## Within-sector decomposition (F1)")
    lines.append("")
    lines.append(sector_para)
    lines.append("")

    # 4. go/no-go recommendation
    p0_flag = False
    p1_flag = False
    reasons = []

    if not df_rt.empty:
        d_test = df_rt[df_rt["split"] == "test"]
        n_collapsed = int((d_test["verdict"] == "Collapsed").sum())
        n_concentrated = int((d_test["verdict"] == "Concentrated").sum())
        if n_collapsed == len(d_test) and len(d_test) > 0:
            p0_flag = True
            reasons.append("Router collapsed across all (fold, seed) test cells.")
        elif n_collapsed > 0:
            p1_flag = True
            reasons.append(f"Router collapsed on {n_collapsed} of {len(d_test)} cells.")
        elif n_concentrated > 0:
            p1_flag = True
            reasons.append(f"Router concentrated on {n_concentrated} of {len(d_test)} cells.")

    if not df_gate.empty:
        d_test = df_gate[df_gate["split"] == "test"]
        n_saturated_per_gate = (
            d_test[d_test["verdict"] == "Saturated"]
            .groupby("gate").size()
        )
        n_locked_per_gate = (
            d_test[d_test["verdict"] == "Locked"].groupby("gate").size()
        )
        for gate in d_test["gate"].unique():
            n_total = (d_test["gate"] == gate).sum()
            n_sat = int(n_saturated_per_gate.get(gate, 0))
            n_loc = int(n_locked_per_gate.get(gate, 0))
            if n_sat == n_total and n_total > 0:
                p0_flag = True
                reasons.append(f"Gate {gate} saturated on every (fold, seed) cell.")
            elif n_loc == n_total and n_total > 0:
                p1_flag = True
                reasons.append(f"Gate {gate} locked on every (fold, seed) cell.")

    if f1_n_total > 0:
        if f1_n_pos < WITHIN_SECTOR_BYSTANDER:
            p0_flag = True
            reasons.append(
                f"F1 within-sector positive in only {f1_n_pos}/{f1_n_total} sectors "
                "(fewer than 4); F1 headline is a level-dispersion artefact.")
        elif f1_n_pos < WITHIN_SECTOR_AGNOSTIC:
            p1_flag = True
            reasons.append(
                f"F1 within-sector positive in {f1_n_pos}/{f1_n_total} sectors "
                "(below 9-of-11 target); sector-agnostic claim is partial.")

    if p0_flag:
        recommendation = "Stop and consult"
    elif p1_flag:
        recommendation = "Proceed with caveat"
    else:
        recommendation = "Proceed"

    lines.append("## Go / no-go recommendation for Phase B")
    lines.append("")
    lines.append(f"**{recommendation}**.")
    lines.append("")
    if reasons:
        lines.append("Reasons:")
        for r in reasons:
            lines.append(f"- {r}")
        lines.append("")

    out_path.write_text("\n".join(lines))
    return recommendation


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--diag-dir", type=str, default="experiments/lattice/diagnostics")
    p.add_argument("--folds", type=str, default="1,2,3")
    p.add_argument("--seeds", type=str, default="42,43,44,45,46")
    args = p.parse_args()

    diag_dir = Path(args.diag_dir)
    folds = [int(x) for x in args.folds.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]

    print("[phase_a summarise] gate trajectories", flush=True)
    gate_out = write_gate_summary(diag_dir, folds, seeds)
    print("[phase_a summarise] router utilisation", flush=True)
    router_out = write_router_summary(diag_dir, folds, seeds)
    print("[phase_a summarise] within-sector IC", flush=True)
    sector_out = write_within_sector(diag_dir, folds, seeds)
    print("[phase_a summarise] roll-up verdict", flush=True)
    verdict = write_verdict(diag_dir, gate_out, router_out, sector_out)
    print(f"[phase_a summarise] recommendation: {verdict}", flush=True)


if __name__ == "__main__":
    main()
