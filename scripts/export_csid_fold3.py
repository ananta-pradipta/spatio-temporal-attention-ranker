"""Export CSID v1 fold-3 predictions and alpha trajectory to CSV.

Outputs under ``results/exports/``:
    csid_fold3_predictions.csv  (date, ticker, seed, y_hat)
    csid_fold3_alpha_per_day.csv (date, seed, alpha_csid + cs_struct features)
    csid_fold3_summary.csv (per-day per-seed daily IC + n_active)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CSID_DIR = Path("results/csid_v1")
EXPORT_DIR = Path("results/exports")
SEEDS = [42, 43, 44, 45, 46]
CS_COLS = ["pc1_share_21d", "avg_pairwise_corr_60d",
           "dispersion_5d", "market_return_5d"]


def main() -> None:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    p0 = np.load(CSID_DIR / f"fold3_seed{SEEDS[0]}_predictions.npz", allow_pickle=True)
    y_true = p0["y_true"]
    loss_mask = p0["loss_mask"]
    tradable = p0["tradable_mask"]
    test_idx = p0["test_idx"]
    tickers = list(p0["tickers"])
    dates_raw = list(p0["dates"])
    panel_dates = pd.to_datetime(dates_raw)

    # 1. Per-(day, ticker, seed) predictions.
    print("[export-csid] building csid_fold3_predictions.csv...")
    rows = []
    for seed in SEEDS:
        p = np.load(CSID_DIR / f"fold3_seed{seed}_predictions.npz", allow_pickle=True)
        y_hat = p["y_hat"]
        for t in test_idx:
            d = panel_dates[t].date()
            for j, ticker in enumerate(tickers):
                if not tradable[t, j]:
                    continue
                rows.append({
                    "date": d, "ticker": ticker, "seed": int(seed),
                    "y_hat": float(y_hat[t, j]),
                    "y_true_5d_fwd_log_return": float(y_true[t, j]),
                    "loss_mask": bool(loss_mask[t, j]),
                })
    pred_df = pd.DataFrame(rows)
    pred_path = EXPORT_DIR / "csid_fold3_predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"[export-csid] wrote {pred_path}: {pred_df.shape}")

    # 2. Per-(day, seed) alpha + cs_struct.
    print("[export-csid] building csid_fold3_alpha_per_day.csv...")
    arows = []
    for seed in SEEDS:
        p = np.load(CSID_DIR / f"fold3_seed{seed}_predictions.npz", allow_pickle=True)
        alpha = p["alpha_csid"]
        cs = p["cs_struct"]
        for t in test_idx:
            row = {
                "date": panel_dates[t].date(), "seed": int(seed),
                "alpha_csid": float(alpha[t]),
            }
            for k, col in enumerate(CS_COLS):
                row[col] = float(cs[t, k])
            arows.append(row)
    alpha_df = pd.DataFrame(arows)
    alpha_path = EXPORT_DIR / "csid_fold3_alpha_per_day.csv"
    alpha_df.to_csv(alpha_path, index=False)
    print(f"[export-csid] wrote {alpha_path}: {alpha_df.shape}")

    # 3. Per-day per-seed daily IC summary.
    print("[export-csid] building csid_fold3_summary.csv...")
    srows = []
    for seed in SEEDS:
        p = np.load(CSID_DIR / f"fold3_seed{seed}_predictions.npz", allow_pickle=True)
        y_hat = p["y_hat"]
        for t in test_idx:
            m = loss_mask[t]
            if m.sum() < 5:
                continue
            a = y_hat[t, m]; b = y_true[t, m]
            if a.std() < 1e-9 or b.std() < 1e-9:
                continue
            ic_p = float(np.corrcoef(a, b)[0, 1])
            a_rank = pd.Series(a).rank().to_numpy()
            b_rank = pd.Series(b).rank().to_numpy()
            ic_s = float(np.corrcoef(a_rank, b_rank)[0, 1])
            srows.append({
                "date": panel_dates[t].date(),
                "seed": int(seed),
                "n_active": int(m.sum()),
                "daily_ic_pearson": ic_p,
                "daily_ic_spearman": ic_s,
                "alpha_csid": float(p["alpha_csid"][t]),
            })
    summary_df = pd.DataFrame(srows)
    summary_path = EXPORT_DIR / "csid_fold3_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"[export-csid] wrote {summary_path}: {summary_df.shape}")

    print()
    print("=== CSID fold-3 EXPORT SUMMARY ===")
    for path in [pred_path, alpha_path, summary_path]:
        print(f"  {path}: {path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
