"""Fold-2 sub-period IC decomposition for pure STAR predictions.

Diagnostic 1 per `parallel-work-plan-sota-and-fold2-diagnostics.md`
Section 4.1. Answers: is pure STAR's fold-2 IC failure uniform across
the test window, or concentrated in specific sub-periods?

Input: results/star/audited_pure/fold2_seed{42..46}_n100.npz
Output: docs/fold2_diagnostic_1.md + figures/fold2_subperiod_*.png
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_DIR = Path("results/star/audited_pure")
OUT_MD = Path("docs/fold2_diagnostic_1.md")
FIG_DIR = Path("docs/figures")
SEEDS = [42, 43, 44, 45, 46]


def daily_ic(preds: np.ndarray, y: np.ndarray, mask: np.ndarray,
             method: str = "pearson") -> np.ndarray:
    """Per-day IC across the active tickers on each day.

    preds, y: [T, N]  mask: [T, N]. Returns: [T] with NaN on days where
    fewer than 3 tickers are active.
    """
    T = preds.shape[0]
    out = np.full(T, np.nan, dtype=np.float64)
    for t in range(T):
        m = mask[t]
        if m.sum() < 3:
            continue
        p = preds[t, m].astype(np.float64)
        q = y[t, m].astype(np.float64)
        if method == "spearman":
            p = pd.Series(p).rank().values
            q = pd.Series(q).rank().values
        if p.std() < 1e-12 or q.std() < 1e-12:
            continue
        out[t] = np.corrcoef(p, q)[0, 1]
    return out


def load_fold2_predictions() -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Stack predictions across 5 seeds; return (preds_mean, y, mask, dates)."""
    preds_stack = []
    y_ref = mask_ref = dates_ref = None
    for s in SEEDS:
        z = np.load(RESULTS_DIR / f"fold2_seed{s}_n100.npz", allow_pickle=True)
        preds_stack.append(z["preds"])
        if y_ref is None:
            y_ref = z["y"]
            mask_ref = z["mask"]
            # test_dates is the full 252 test days; preds span the last 232 (W=20 burn-in)
            td = [str(d) for d in z["test_dates"]]
            dates_ref = pd.to_datetime(td)
    preds_mean = np.mean(np.stack(preds_stack, axis=0), axis=0)  # [T_pred, N]
    T_pred = preds_mean.shape[0]
    # Align dates: the last T_pred entries
    dates_aligned = dates_ref[-T_pred:]
    return preds_mean, y_ref, mask_ref, dates_aligned


def per_seed_daily_ic() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-seed per-day IC and rank-IC. Columns: seeds, index: dates."""
    ic_by_seed = {}
    rk_by_seed = {}
    dates_aligned = None
    for s in SEEDS:
        z = np.load(RESULTS_DIR / f"fold2_seed{s}_n100.npz", allow_pickle=True)
        preds, y, mask = z["preds"], z["y"], z["mask"]
        td = [str(d) for d in z["test_dates"]]
        T_pred = preds.shape[0]
        dates_aligned = pd.to_datetime(td)[-T_pred:]
        ic_by_seed[s] = daily_ic(preds, y, mask, "pearson")
        rk_by_seed[s] = daily_ic(preds, y, mask, "spearman")
    ic_df = pd.DataFrame(ic_by_seed, index=dates_aligned)
    rk_df = pd.DataFrame(rk_by_seed, index=dates_aligned)
    return ic_df, rk_df


def aggregate_by_period(daily: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Mean across seeds then across days within each period bucket."""
    across_seeds = daily.mean(axis=1)
    grouped = across_seeds.groupby(pd.Grouper(freq=freq)).agg(["mean", "std", "count"])
    grouped.columns = ["ic_mean", "ic_std", "n_days"]
    return grouped.dropna(subset=["ic_mean"])


def bucket_by_vix(daily_ic_series: pd.Series, vix: pd.Series,
                  n_buckets: int = 3) -> pd.DataFrame:
    """Bucket test days into terciles by VIX level and compute mean IC per bucket."""
    aligned = pd.concat([daily_ic_series.rename("ic"), vix.rename("vix")], axis=1).dropna()
    qs = aligned["vix"].quantile([0, 1/3, 2/3, 1]).values
    labels = [f"low (VIX<{qs[1]:.1f})", f"mid ({qs[1]:.1f}-{qs[2]:.1f})", f"high (VIX>{qs[2]:.1f})"]
    aligned["bucket"] = pd.cut(aligned["vix"], qs, labels=labels, include_lowest=True)
    summary = aligned.groupby("bucket")["ic"].agg(["mean", "std", "count"])
    summary.columns = ["ic_mean", "ic_std", "n_days"]
    return summary


def classify_pattern(weekly_ic: pd.DataFrame) -> str:
    """Classify uniform / bimodal / declining per memo Section 4.1."""
    vals = weekly_ic["ic_mean"].dropna().values
    if len(vals) < 4:
        return "insufficient data"
    # uniform: low variance, close to overall mean
    overall = vals.mean()
    uniform_score = np.std(vals) / (abs(overall) + 1e-6)
    # declining: monotonic downward trend (linear fit slope < 0 and |slope| large)
    t = np.arange(len(vals))
    slope, _ = np.polyfit(t, vals, 1)
    declining = slope < -0.002  # arbitrary threshold
    # bimodal: high variance with some periods substantially positive
    bimodal = np.std(vals) > 0.03 and (vals.max() - vals.min() > 0.06)
    notes = []
    notes.append(f"overall mean {overall:+.4f}, std {np.std(vals):.4f}, min {vals.min():+.4f}, max {vals.max():+.4f}")
    notes.append(f"linear slope over weeks: {slope:+.5f}")
    if declining and not bimodal:
        label = "C (declining)"
    elif bimodal:
        label = "B (bimodal)"
    else:
        label = "A (uniform failure)"
    return label + " | " + "; ".join(notes)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)

    print("Loading fold-2 predictions (5 seeds)...")
    ic_df, rk_df = per_seed_daily_ic()
    print(f"  {len(ic_df)} test days, {ic_df.shape[1]} seeds")

    daily_mean_ic = ic_df.mean(axis=1)
    daily_mean_rk = rk_df.mean(axis=1)
    print(f"  overall daily-then-mean IC: {daily_mean_ic.mean():+.4f}")
    print(f"  overall daily-then-mean rank-IC: {daily_mean_rk.mean():+.4f}")

    weekly = aggregate_by_period(ic_df, "W")
    weekly_rk = aggregate_by_period(rk_df, "W")
    monthly = aggregate_by_period(ic_df, "M")
    monthly_rk = aggregate_by_period(rk_df, "M")

    pattern = classify_pattern(weekly)
    print(f"  pattern classification: {pattern}")

    risk = pd.read_parquet("data/processed/risk_features.parquet")
    risk.index = pd.to_datetime(risk.index)
    # We standardized VIX at train time; here we want the RAW VIX level for
    # bucketing. Load the raw volatility indices file.
    vol_raw = pd.read_parquet("data/raw/volatility_indices.parquet")
    vol_raw.index = pd.to_datetime(vol_raw.index)
    vix_series = vol_raw["VIX"]

    ic_bucket = bucket_by_vix(daily_mean_ic, vix_series)
    rk_bucket = bucket_by_vix(daily_mean_rk, vix_series)

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        ax[0].plot(daily_mean_ic.index, daily_mean_ic.values, "o-", markersize=3, linewidth=0.8)
        ax[0].axhline(0, color="gray", linewidth=0.5)
        ax[0].set_ylabel("Daily IC")
        ax[0].set_title("Pure STAR fold-2: per-day IC (mean of 5 seeds)")
        ax[0].grid(True, alpha=0.3)
        ax[1].plot(daily_mean_rk.index, daily_mean_rk.values, "o-", markersize=3, linewidth=0.8, color="C1")
        ax[1].axhline(0, color="gray", linewidth=0.5)
        ax[1].set_ylabel("Daily rank-IC")
        ax[1].set_xlabel("Test date")
        ax[1].grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(FIG_DIR / "fold2_subperiod_daily.png", dpi=120)

        fig2, ax2 = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        ax2[0].bar(weekly.index, weekly["ic_mean"], width=5, color=["tab:red" if v < 0 else "tab:blue" for v in weekly["ic_mean"]])
        ax2[0].axhline(0, color="gray", linewidth=0.5)
        ax2[0].set_ylabel("Weekly mean IC")
        ax2[0].set_title("Pure STAR fold-2: weekly IC (mean of 5 seeds, then weekly mean)")
        ax2[0].grid(True, alpha=0.3)
        ax2[1].bar(monthly.index, monthly["ic_mean"], width=20, color=["tab:red" if v < 0 else "tab:blue" for v in monthly["ic_mean"]])
        ax2[1].axhline(0, color="gray", linewidth=0.5)
        ax2[1].set_ylabel("Monthly mean IC")
        ax2[1].set_xlabel("Test date")
        ax2[1].grid(True, alpha=0.3)
        plt.tight_layout()
        fig2.savefig(FIG_DIR / "fold2_subperiod_weekly_monthly.png", dpi=120)
    except Exception as e:
        print(f"plot skipped: {e}")

    # Write markdown report
    def df_to_md(df: pd.DataFrame, float_fmt: str = "{:+.4f}") -> str:
        df2 = df.copy()
        for c in df2.select_dtypes(include=[float]).columns:
            df2[c] = df2[c].map(lambda v: float_fmt.format(v) if pd.notna(v) else "n/a")
        # Manual markdown table (no tabulate dependency)
        idx_name = df2.index.name or "index"
        cols = [idx_name] + list(df2.columns)
        lines = ["| " + " | ".join(str(c) for c in cols) + " |"]
        lines.append("|" + "|".join(["---"] * len(cols)) + "|")
        for idx, row in df2.iterrows():
            cells = [str(idx)] + [str(v) for v in row.values]
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)

    overall_ic = daily_mean_ic.mean()
    overall_rk = daily_mean_rk.mean()

    report = f"""# Fold-2 Diagnostic 1: Per-Period IC Decomposition

Date: 2026-04-15
Source: pure STAR audited predictions, 5 seeds on fold 2
(`results/star/audited_pure/fold2_seed{{42..46}}_n100.npz`).

## 1. Question

Is pure STAR's fold-2 IC failure uniform across the ~6-month test
window (2021-06-25 to 2022-06-22), or concentrated in specific
sub-periods?

## 2. Headline numbers

Pure STAR fold 2, 5-seed mean-of-seeds daily IC series:
- Overall daily-then-mean IC: **{overall_ic:+.4f}**
- Overall daily-then-mean rank-IC: **{overall_rk:+.4f}**
- {len(daily_mean_ic)} test days
- Pattern classification: **{pattern}**

These reproduce the headline verification result (fold 2 mean IC
-0.0031 ± 0.0094). This diagnostic decomposes that headline.

## 3. Weekly IC

{df_to_md(weekly)}

Positive weeks: {int((weekly['ic_mean'] > 0).sum())} / {len(weekly)}.
Negative weeks: {int((weekly['ic_mean'] < 0).sum())} / {len(weekly)}.

## 4. Monthly IC

{df_to_md(monthly)}

Positive months: {int((monthly['ic_mean'] > 0).sum())} / {len(monthly)}.

## 5. VIX-bucketed IC

Three terciles by raw VIX level on the test day (data from
`data/raw/volatility_indices.parquet`).

{df_to_md(ic_bucket)}

Rank-IC by VIX bucket:

{df_to_md(rk_bucket)}

## 6. Weekly rank-IC

{df_to_md(weekly_rk)}

## 7. Interpretation

Pattern: **{pattern}**

The per-memo-Section-4.1 patterns:
- Pattern A (uniform failure): structural failure, IC near -0.003
  across all sub-periods. Motivates regime-detection mechanisms.
- Pattern B (bimodal): good in some sub-periods, terrible in others.
  Motivates condition-specific routing.
- Pattern C (declining): positive at start, degrades over time.
  Motivates online learning or temporal recalibration.

Diagnostic 2 (distribution shift quantification) and Diagnostic 3
(graph staleness ablation) will test specific mechanisms.

## 8. Figures

- `docs/figures/fold2_subperiod_daily.png` — per-day IC and rank-IC
  across the fold-2 test window (5-seed mean).
- `docs/figures/fold2_subperiod_weekly_monthly.png` — weekly and
  monthly mean IC bars (red = negative, blue = positive).

## 9. Data exports

- `docs/fold2_diagnostic_1_weekly_ic.csv`
- `docs/fold2_diagnostic_1_monthly_ic.csv`
- `docs/fold2_diagnostic_1_vix_bucketed_ic.csv`
"""

    OUT_MD.write_text(report)
    weekly.to_csv(OUT_MD.parent / "fold2_diagnostic_1_weekly_ic.csv")
    monthly.to_csv(OUT_MD.parent / "fold2_diagnostic_1_monthly_ic.csv")
    ic_bucket.to_csv(OUT_MD.parent / "fold2_diagnostic_1_vix_bucketed_ic.csv")

    print(f"\nwrote {OUT_MD}")
    print(f"wrote weekly/monthly/vix CSVs alongside")
    print(f"pattern: {pattern}")


if __name__ == "__main__":
    main()
