"""Diagnose CSID v1's fold-3 weakness.

Goal: figure out (a) why CSID's gate closes in fold-3 stress regime
(opposite to design intent), (b) what's special about seed 42's -0.009
fold-3 IC outlier, (c) which months drag the mean.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CSID_DIR = Path("results/csid_v1")
DOW_DIR = Path("results/dow_epistar_v23_no_rate_memory")
SEEDS = [42, 43, 44, 45, 46]


def load_pred(d: Path, fold: int, seed: int):
    return np.load(d / f"fold{fold}_seed{seed}_predictions.npz", allow_pickle=True)


def daily_ic(y_hat, y, mask, idx):
    out = []
    for t in idx:
        m = mask[t]
        if m.sum() < 5:
            continue
        a = y_hat[t, m]; b = y[t, m]
        if a.std() < 1e-9 or b.std() < 1e-9:
            continue
        out.append((t, float(np.corrcoef(a, b)[0, 1])))
    return out


def main() -> None:
    # CSID fold-3, seed 42 (the outlier).
    p = load_pred(CSID_DIR, 3, 42)
    y_hat = p["y_hat"]; y = p["y_true"]; loss_mask = p["loss_mask"]
    test_idx = p["test_idx"]
    dates = pd.to_datetime(p["dates"])
    alpha_arr = p["alpha_csid"]
    cs_struct = p["cs_struct"]

    print("=== CSID fold-3 daily IC by seed ===")
    for seed in SEEDS:
        p_s = load_pred(CSID_DIR, 3, seed)
        d_ic = daily_ic(p_s["y_hat"], p_s["y_true"], p_s["loss_mask"], test_idx)
        ics = np.array([x[1] for x in d_ic])
        print(f"seed {seed}: n={len(ics)} mean={ics.mean():+.4f} median={np.median(ics):+.4f} "
              f"std={ics.std():.4f} pct_neg={100*(ics<0).sum()/len(ics):.1f}%")

    print()
    print("=== CSID fold-3 monthly IC (5-seed average prediction) ===")
    y_hats = [load_pred(CSID_DIR, 3, s)["y_hat"] for s in SEEDS]
    y_hat_avg = np.mean(y_hats, axis=0)
    d_ic = daily_ic(y_hat_avg, y, loss_mask, test_idx)
    df = pd.DataFrame([(dates[t], ic) for t, ic in d_ic], columns=["date", "ic"])
    df["month"] = df["date"].dt.to_period("M")
    monthly = df.groupby("month")["ic"].agg(["mean", "median", "count"])
    print(monthly.to_string())

    print()
    print("=== alpha_csid trajectory across fold-3 (5 seeds avg) ===")
    alpha_avgs = []
    for seed in SEEDS:
        p_s = load_pred(CSID_DIR, 3, seed)
        alpha_avgs.append(p_s["alpha_csid"])
    alpha_avg = np.mean(alpha_avgs, axis=0)
    alpha_test = alpha_avg[test_idx]
    df_a = pd.DataFrame({"date": [dates[t] for t in test_idx], "alpha": alpha_test})
    df_a["month"] = df_a["date"].dt.to_period("M")
    print(df_a.groupby("month")["alpha"].agg(["mean", "min", "max"]).to_string())

    print()
    print("=== alpha vs daily IC correlation (does alpha closing track bad days?) ===")
    daily_ic_arr = np.array([ic for _, ic in d_ic])
    daily_alpha_arr = np.array([alpha_avg[t] for t, _ in d_ic])
    corr = np.corrcoef(daily_alpha_arr, daily_ic_arr)[0, 1]
    print(f"corr(daily_alpha, daily_ic) = {corr:+.4f}")
    print(f"  if positive: alpha is HIGH on good days, LOW on bad days "
          f"(gate is correctly closing on stress)")
    print(f"  if negative: alpha and IC anti-correlated -- gate is closing when it should open")

    print()
    print("=== cs_struct vs daily IC (which factor predicts CSID failure?) ===")
    feat_cols = ["pc1_share_21d", "avg_pairwise_corr_60d", "dispersion_5d", "market_return_5d"]
    for k, col in enumerate(feat_cols):
        col_arr = np.array([cs_struct[t, k] for t, _ in d_ic])
        c = np.corrcoef(col_arr, daily_ic_arr)[0, 1]
        print(f"corr({col}, daily_ic) = {c:+.4f}")

    print()
    print("=== CSID vs DOW-epiSTAR fold-3 daily IC head-to-head (5-seed avg) ===")
    y_hats_dow = [load_pred(DOW_DIR, 3, s)["y_hat"] for s in SEEDS]
    y_hat_dow = np.mean(y_hats_dow, axis=0)
    p_dow = load_pred(DOW_DIR, 3, 42)
    d_ic_dow = daily_ic(y_hat_dow, p_dow["y_true"], p_dow["loss_mask"], p_dow["test_idx"])
    csid_dict = {t: ic for t, ic in d_ic}
    dow_dict = {t: ic for t, ic in d_ic_dow}
    common = sorted(set(csid_dict) & set(dow_dict))
    diffs = []
    for t in common:
        diffs.append((dates[t], csid_dict[t] - dow_dict[t]))
    diffs_arr = np.array([d for _, d in diffs])
    print(f"days where CSID > DOW: {(diffs_arr > 0).sum()} / {len(diffs)} ({100*(diffs_arr>0).sum()/len(diffs):.1f}%)")
    print(f"mean delta (CSID - DOW): {diffs_arr.mean():+.4f}")
    print(f"worst CSID days (CSID - DOW most negative):")
    for d, delta in sorted(diffs, key=lambda x: x[1])[:5]:
        print(f"  {d.date()}: delta={delta:+.4f}")
    print(f"best CSID days:")
    for d, delta in sorted(diffs, key=lambda x: -x[1])[:5]:
        print(f"  {d.date()}: delta={delta:+.4f}")


if __name__ == "__main__":
    main()
