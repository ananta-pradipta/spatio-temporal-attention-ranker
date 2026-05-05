"""Diagnose why OW-epiSTAR v1 fold-3 IC is much lower than fold-1/fold-2.

Inspects:
    1. Fold-3 date range (regime classification: 2022 bear?).
    2. Daily IC time series across the test window (drawdown periods).
    3. Per-ticker contribution to fold-3 IC: who drags the mean?
    4. Age distribution of fold-3 active universe (more young IPOs?).
    5. Y vs y_hat spread by month.
    6. Cohort-stratified IC by month within fold-3.

Usage: python scripts/diagnose_ow_v1_fold3.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

RESULT_DIR = Path("results/ow_epistar_v1")
SEEDS = [42, 43, 44, 45, 46]


def load_fold(fold: int) -> dict:
    """Load all seeds for a fold; return averaged y_hat and per-seed details."""
    preds = []
    for s in SEEDS:
        p = np.load(RESULT_DIR / f"fold{fold}_seed{s}_predictions.npz", allow_pickle=True)
        preds.append(p)
    y_hat_avg = np.mean([p["y_hat"] for p in preds], axis=0)
    return {
        "y_hat_avg": y_hat_avg,
        "y": preds[0]["y_true"],
        "mask": preds[0]["loss_mask"],
        "test_idx": preds[0]["test_idx"],
        "tickers": preds[0]["tickers"],
        "dates": pd.to_datetime(preds[0]["dates"]),
        "age_days": preds[0]["age_days"],
        "preds": preds,
    }


def daily_ic(y_hat, y, mask, idx):
    """Per-day IC over test indices."""
    out = {}
    for t in idx:
        m = mask[t]
        if m.sum() < 5:
            continue
        a = y_hat[t, m]; b = y[t, m]
        sa = a.std(); sb = b.std()
        if sa < 1e-9 or sb < 1e-9:
            continue
        out[t] = float(np.corrcoef(a, b)[0, 1])
    return out


def per_ticker_contrib(y_hat, y, mask, idx, tickers):
    """Each ticker's contribution to mean fold IC.

    Approximates by computing IC with one ticker masked out, and
    reporting the change vs the full IC.
    """
    base_ics = list(daily_ic(y_hat, y, mask, idx).values())
    base_mean = float(np.mean(base_ics))
    contribs = []
    for ti, tk in enumerate(tickers):
        m = mask.copy()
        m[:, ti] = False
        ics = list(daily_ic(y_hat, y, m, idx).values())
        if not ics:
            continue
        contribs.append({
            "ticker": tk, "ic_without": float(np.mean(ics)),
            "delta_vs_full": base_mean - float(np.mean(ics)),
        })
    return base_mean, sorted(contribs, key=lambda r: -r["delta_vs_full"])


def main() -> None:
    """Walk through each diagnostic step and print a report."""
    fold1 = load_fold(1)
    fold2 = load_fold(2)
    fold3 = load_fold(3)

    print("=== 1. Fold date ranges ===")
    for f, d in zip([1, 2, 3], [fold1, fold2, fold3]):
        ti = d["test_idx"]
        print(f"fold {f}: test {d['dates'][ti[0]].date()} to {d['dates'][ti[-1]].date()} "
              f"({len(ti)} days)")

    print("\n=== 2. 5-seed mean fold-3 IC vs the per-seed numbers ===")
    daily = daily_ic(fold3["y_hat_avg"], fold3["y"], fold3["mask"], fold3["test_idx"])
    print(f"5-seed-mean prediction: fold-3 IC = {np.mean(list(daily.values())):.4f} "
          f"(daily-then-mean)")
    for s_idx, s in enumerate(SEEDS):
        p = fold3["preds"][s_idx]
        d_s = daily_ic(p["y_hat"], p["y_true"], p["loss_mask"], p["test_idx"])
        print(f"  seed {s}: IC = {np.mean(list(d_s.values())):.4f}, "
              f"n_days = {len(d_s)}")

    print("\n=== 3. Daily IC time series for fold-3 (by month) ===")
    daily_arr = pd.Series(daily, index=[fold3["dates"][t] for t in daily])
    by_month = daily_arr.groupby(pd.Grouper(freq="MS")).agg(["mean", "count"])
    print(by_month.to_string())

    print("\n=== 4. Worst single days for fold-3 ===")
    worst = sorted(daily.items(), key=lambda kv: kv[1])[:10]
    for t, ic in worst:
        d = fold3["dates"][t].date()
        a_day = fold3["mask"][t].sum()
        print(f"  {d}: IC = {ic:+.4f}, n_active = {a_day}")

    print("\n=== 5. Best single days for fold-3 ===")
    best = sorted(daily.items(), key=lambda kv: -kv[1])[:5]
    for t, ic in best:
        d = fold3["dates"][t].date()
        a_day = fold3["mask"][t].sum()
        print(f"  {d}: IC = {ic:+.4f}, n_active = {a_day}")

    print("\n=== 6. Cross-sectional std of y_true (volatility regime) ===")
    for f, d in zip([1, 2, 3], [fold1, fold2, fold3]):
        ti = d["test_idx"]
        sds = []
        for t in ti:
            m = d["mask"][t]
            if m.sum() < 5:
                continue
            sds.append(float(d["y"][t, m].std()))
        print(f"  fold {f}: mean cs-std of y_true = {np.mean(sds):.4f}")

    print("\n=== 7. Mean active universe size ===")
    for f, d in zip([1, 2, 3], [fold1, fold2, fold3]):
        ti = d["test_idx"]
        sizes = [d["mask"][t].sum() for t in ti]
        print(f"  fold {f}: mean active = {np.mean(sizes):.1f}, "
              f"min = {np.min(sizes)}, max = {np.max(sizes)}")

    print("\n=== 8. Age cohort distribution within each fold's test set ===")
    for f, d in zip([1, 2, 3], [fold1, fold2, fold3]):
        ti = d["test_idx"]
        age = d["age_days"]
        m = d["mask"]
        cells = []
        for t in ti:
            for i in range(age.shape[1]):
                if m[t, i]:
                    cells.append(int(age[t, i]))
        cells = np.asarray(cells)
        fresh = (cells <= 60).sum()
        young = ((cells > 60) & (cells <= 252)).sum()
        seasoned = (cells > 252).sum()
        total = len(cells)
        print(f"  fold {f}: total_cells = {total}, "
              f"fresh<=60d {100*fresh/total:.1f}%, "
              f"young 61-252d {100*young/total:.1f}%, "
              f"seasoned>252d {100*seasoned/total:.1f}%")

    print("\n=== 9. Per-ticker contribution to fold-3 IC (top 10 draggers) ===")
    base_mean, contribs = per_ticker_contrib(
        fold3["y_hat_avg"], fold3["y"], fold3["mask"],
        fold3["test_idx"], fold3["tickers"],
    )
    print(f"fold-3 5-seed-mean IC (full universe): {base_mean:+.4f}")
    print("Top 10 draggers (removing them improves IC the most):")
    for r in contribs[:10]:
        ti = list(fold3["tickers"]).index(r["ticker"])
        # Mean age at start of test window for this ticker.
        ti_idx = fold3["test_idx"][0]
        age_at_test_start = fold3["age_days"][ti_idx, ti]
        print(f"  {r['ticker']:>6}: IC without = {r['ic_without']:+.4f} "
              f"(delta = {r['delta_vs_full']:+.4f}), age at test start = {age_at_test_start}d")

    print("\nTop 10 contributors (removing them hurts IC the most):")
    for r in contribs[-10:]:
        ti = list(fold3["tickers"]).index(r["ticker"])
        ti_idx = fold3["test_idx"][0]
        age_at_test_start = fold3["age_days"][ti_idx, ti]
        print(f"  {r['ticker']:>6}: IC without = {r['ic_without']:+.4f} "
              f"(delta = {r['delta_vs_full']:+.4f}), age at test start = {age_at_test_start}d")


if __name__ == "__main__":
    main()
