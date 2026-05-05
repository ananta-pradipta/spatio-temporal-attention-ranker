"""Extract per-day IC time series for RAG-STAR and baselines.

Reads prediction npz files (y_hat, y_true, mask) per (model, fold,
seed), averages predictions across seeds, computes the daily Pearson
IC against the realised target, and saves a small per-fold CSV with
columns: date, model, daily_ic, rolling_ic_20d.

Output: results/exports/per_day_ic_fold{1,2,3}.csv
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path("results/exports")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "RAG-STAR":  "results/dow_epistar_v23_no_rate_memory",
    "MASTER":    "results/baselines_244/master_v2",
    "FactorVAE": "results/baselines_244/factorvae_v2",
    "StockMixer": "results/baselines_244/stockmixer_v2",
}


def per_day_ic(y_hat: np.ndarray, y: np.ndarray, mask: np.ndarray) -> np.ndarray:
    T = y_hat.shape[0]
    out = np.full(T, np.nan, dtype=np.float64)
    for t in range(T):
        m = mask[t]
        if m.sum() < 5:
            continue
        a = y_hat[t, m]
        b = y[t, m]
        if a.std() < 1e-9 or b.std() < 1e-9:
            continue
        out[t] = float(np.corrcoef(a, b)[0, 1])
    return out


def load_avg_predictions(model_dir: str, fold: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Average y_hat across seeds; return (y_hat_avg, y_true, mask, dates)."""
    pattern = f"{model_dir}/fold{fold}_seed*_predictions.npz"
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no predictions at {pattern}")
    per_seed_yhat = []
    y_true = None
    mask = None
    dates = None
    for f in files:
        d = np.load(f, allow_pickle=True)
        per_seed_yhat.append(d["y_hat"])
        if y_true is None:
            y_true = d["y_true"]
            # different models save the mask under different names
            for name in ("loss_mask", "eval_mask", "mask"):
                if name in d:
                    mask = d[name].astype(bool)
                    break
            if mask is None:
                raise RuntimeError(f"no mask in {f}; keys = {list(d.keys())}")
            dates = d["dates"] if "dates" in d else None
    y_hat_avg = np.mean(np.stack(per_seed_yhat), axis=0)
    return y_hat_avg, y_true, mask, dates


def main() -> None:
    for fold in (1, 2, 3):
        rows: list[dict] = []
        date_arr = None
        test_idx = None
        for model_name, model_dir in MODELS.items():
            try:
                y_hat, y_true, mask, dates = load_avg_predictions(model_dir, fold)
            except (FileNotFoundError, RuntimeError) as e:
                print(f"[fold{fold}] skip {model_name}: {e}")
                continue
            ic = per_day_ic(y_hat, y_true, mask)
            # Restrict to test indices: rows where the model's mask
            # has any non-zero entry are the test days.
            active = mask.any(axis=1)
            rolling = pd.Series(ic).rolling(20, min_periods=10).mean()
            for t in np.where(active)[0]:
                date_str = str(dates[t]) if dates is not None else f"day_{t}"
                rows.append({
                    "date": date_str,
                    "model": model_name,
                    "daily_ic": ic[t],
                    "rolling_ic_20d": rolling.iloc[t],
                })
        if not rows:
            print(f"[fold{fold}] no rows")
            continue
        df = pd.DataFrame(rows)
        out = OUT_DIR / f"per_day_ic_fold{fold}.csv"
        df.to_csv(out, index=False)
        print(f"[fold{fold}] wrote {out} with {len(df)} rows, models = {df['model'].unique().tolist()}")


if __name__ == "__main__":
    main()
