"""Stratified Information Coefficient (IC) analysis over walk-forward results.

Reads every `results/walkforward/<model>_fold<F>_seed<S>.npz` companion file,
joins per-day predictions against:
  - CBOE Volatility Index (VIX) buckets (calm / volatile / crash)
  - Catalyst-window proximity (any ticker within +/- 3 trading days of
    an earnings / FDA / trial-readout event)

and reports IC + Rank IC per bucket per model, averaged over seeds within
each fold and then aggregated across folds.

Usage:
    python3 scripts/analyze_regimes.py
        --walkforward-dir results/walkforward
        --output docs/regime_stratified_analysis.md
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def vix_bucket(vix: float) -> str:
    """CBOE Volatility Index level classification.

    Thresholds follow standard market-stress convention:
      calm      VIX < 20   (low realized vol regime)
      volatile  20 <= VIX < 30
      crash     VIX >= 30  (crisis regime; COVID-19 peak ~85, 2008 peak ~80)
    """
    if np.isnan(vix):
        return "unknown"
    if vix < 20:
        return "calm"
    if vix < 30:
        return "volatile"
    return "crash"


def load_catalyst_days(path: Path) -> set[pd.Timestamp]:
    """Return set of dates within +/- 3 trading days of any catalyst event."""
    if not path.exists():
        return set()
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    expanded: set[pd.Timestamp] = set()
    for d in df["date"].unique():
        ts = pd.Timestamp(d)
        for offset in range(-3, 4):
            expanded.add(ts + pd.Timedelta(days=offset))
    return expanded


def daily_corr(preds: np.ndarray, y: np.ndarray, mask: np.ndarray,
               corr_fn) -> float:
    """Average of per-day cross-sectional correlations, weighted by day."""
    out = []
    for t in range(preds.shape[0]):
        m = mask[t]
        if m.sum() < 5:
            continue
        p = preds[t][m]; yv = y[t][m]
        if np.std(p) < 1e-10 or np.std(yv) < 1e-10:
            continue
        c = corr_fn(p, yv)[0]
        if not np.isnan(c):
            out.append(c)
    return float(np.mean(out)) if out else float("nan")


def analyze_bucket(files: Iterable[Path], keep_days_mask, vix_lookup, catalyst_days):
    """Aggregate per-(model,fold) stratified IC across seed files."""
    rows = []
    for f in files:
        try:
            z = np.load(f, allow_pickle=True)
        except Exception:
            continue
        preds = z["preds"]; y = z["y"]; mask = z["mask"]
        dates = [pd.Timestamp(d) for d in z["test_dates"]]
        name = f.stem.rsplit(".", 1)[0]
        parts = name.split("_")
        model = parts[0]
        fold  = int(parts[1].replace("fold", ""))
        seed  = int(parts[2].replace("seed", ""))

        vix_vals = np.array([vix_lookup.get(d.normalize(), np.nan) for d in dates])
        buckets  = np.array([vix_bucket(v) for v in vix_vals])
        cat_flag = np.array([d.normalize() in catalyst_days for d in dates])

        for bucket in ["calm", "volatile", "crash", "all"]:
            sel = np.ones(len(dates), dtype=bool) if bucket == "all" else (buckets == bucket)
            if sel.sum() < 5:
                continue
            ic = daily_corr(preds[sel], y[sel], mask[sel], pearsonr)
            ri = daily_corr(preds[sel], y[sel], mask[sel], spearmanr)
            rows.append({"model": model, "fold": fold, "seed": seed,
                         "strat": "vix", "bucket": bucket, "n_days": int(sel.sum()),
                         "ic": ic, "rank_ic": ri})
        for bucket in ["catalyst", "non_catalyst"]:
            sel = cat_flag if bucket == "catalyst" else ~cat_flag
            if sel.sum() < 5:
                continue
            ic = daily_corr(preds[sel], y[sel], mask[sel], pearsonr)
            ri = daily_corr(preds[sel], y[sel], mask[sel], spearmanr)
            rows.append({"model": model, "fold": fold, "seed": seed,
                         "strat": "catalyst", "bucket": bucket, "n_days": int(sel.sum()),
                         "ic": ic, "rank_ic": ri})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--walkforward-dir", type=Path, default=Path("results/walkforward"))
    ap.add_argument("--vix-parquet", type=Path, default=Path("data/raw/volatility_indices.parquet"))
    ap.add_argument("--catalyst-parquet", type=Path,
                    default=Path("data/processed/catalyst_days_full.parquet"))
    ap.add_argument("--output", type=Path, default=Path("docs/regime_stratified_analysis.md"))
    args = ap.parse_args()

    files = sorted(args.walkforward_dir.glob("*.npz"))
    if not files:
        raise SystemExit(f"No .npz files in {args.walkforward_dir}")
    print(f"found {len(files)} per-seed .npz files")

    vix = pd.read_parquet(args.vix_parquet)
    vix_lookup = {pd.Timestamp(idx).normalize(): float(row["VIX"])
                  for idx, row in vix.iterrows()}
    catalyst_days = load_catalyst_days(args.catalyst_parquet)
    print(f"VIX days: {len(vix_lookup)}  catalyst +/-3 day union: {len(catalyst_days)}")

    df = analyze_bucket(files, None, vix_lookup, catalyst_days)
    if df.empty:
        raise SystemExit("no stratified rows produced")

    out_lines: list[str] = ["# Regime-stratified IC analysis\n"]
    for strat in ["vix", "catalyst"]:
        sub = df[df["strat"] == strat]
        if sub.empty:
            continue
        out_lines.append(f"## Stratification: {strat}\n")
        agg = (sub.groupby(["model", "bucket", "fold"])[["ic", "rank_ic", "n_days"]]
                  .agg({"ic": ["mean", "std"], "rank_ic": ["mean", "std"], "n_days": "first"}))
        agg.columns = ["ic_mean", "ic_std", "rank_mean", "rank_std", "n_days"]
        agg = agg.reset_index()
        for fold in sorted(sub["fold"].unique()):
            out_lines.append(f"### Fold {fold}\n")
            out_lines.append("| model | bucket | n_days | IC (mean +- std) | RankIC (mean +- std) |")
            out_lines.append("|---|---|---:|---:|---:|")
            for _, r in agg[agg["fold"] == fold].iterrows():
                out_lines.append(f"| {r['model']} | {r['bucket']} | {r['n_days']} | "
                                 f"{r['ic_mean']:+.4f} +- {r['ic_std']:.4f} | "
                                 f"{r['rank_mean']:+.4f} +- {r['rank_std']:.4f} |")
            out_lines.append("")
        # cross-fold aggregate
        cross = (sub.groupby(["model", "bucket"])[["ic", "rank_ic"]]
                    .agg(["mean", "std"]))
        cross.columns = ["ic_mean", "ic_std", "rank_mean", "rank_std"]
        cross = cross.reset_index()
        out_lines.append(f"### Cross-fold aggregate ({strat})\n")
        out_lines.append("| model | bucket | IC (mean +- std) | RankIC (mean +- std) |")
        out_lines.append("|---|---|---:|---:|")
        for _, r in cross.iterrows():
            out_lines.append(f"| {r['model']} | {r['bucket']} | "
                             f"{r['ic_mean']:+.4f} +- {r['ic_std']:.4f} | "
                             f"{r['rank_mean']:+.4f} +- {r['rank_std']:.4f} |")
        out_lines.append("")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(out_lines) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
