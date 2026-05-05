"""Catalyst-window subset IC analysis of saved MTGN run predictions.

Reads run JSON files produced by `src.mtgn.training.train_mtgn` that
contain the `test_predictions` block (dates, tickers, y_hat, y, mask).
Joins against `data/processed/catalyst_days.parquet` to split each
(ticker, date) cell into catalyst-window vs calm-period. Reports per-
mode IC on each subset across seeds.

Usage:
    python3 -m src.mtgn.evaluation.catalyst_slice \\
        results/diag_mtgn/*.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import pandas as pd


def load_run(path: Path) -> dict:
    return json.loads(path.read_text())


def slice_ics(run: dict, catalyst_mask: pd.DataFrame) -> dict[str, float]:
    tp = run.get("test_predictions")
    if not tp:
        return {"test_ic_all": run.get("test_ic", float("nan"))}
    dates = pd.to_datetime(tp["dates"])
    tickers = tp["tickers"]
    y_hat = np.array(tp["y_hat"])
    y = np.array(tp["y"])
    mask = np.array(tp["mask"])

    cat_set = set(
        zip(
            catalyst_mask["ticker"].astype(str).str.upper(),
            pd.to_datetime(catalyst_mask["date"]).dt.normalize(),
        )
    )

    is_cat = np.zeros_like(mask, dtype=bool)
    for j, t in enumerate(tickers):
        for i, d in enumerate(dates):
            if (t.upper(), d.normalize()) in cat_set:
                is_cat[i, j] = True

    def _ic(mask_sub: np.ndarray) -> float:
        ics = []
        for i in range(y_hat.shape[0]):
            m = mask[i] & mask_sub[i]
            if m.sum() < 3:
                continue
            a = y_hat[i, m]
            b = y[i, m]
            if a.std() < 1e-8 or b.std() < 1e-8:
                continue
            ics.append(float(np.corrcoef(a, b)[0, 1]))
        return float(np.mean(ics)) if ics else float("nan")

    all_m = np.ones_like(mask)
    cal_m = ~is_cat
    cat_m = is_cat
    return {
        "ic_all": _ic(all_m),
        "ic_catalyst": _ic(cat_m),
        "ic_calm": _ic(cal_m),
        "n_cat_cells": int(is_cat.sum()),
        "n_cells_total": int(mask.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument(
        "--catalyst",
        type=Path,
        default=Path("data/processed/catalyst_days.parquet"),
    )
    args = parser.parse_args()

    cat = pd.read_parquet(args.catalyst)
    by_mode: dict[str, list[dict]] = defaultdict(list)
    for p in args.runs:
        run = load_run(p)
        mode = run["config"]["retrieval_mode"]
        seed = run["config"]["seed"]
        s = slice_ics(run, cat)
        s["seed"] = seed
        by_mode[mode].append(s)
        print(
            f"{p.name:45s}  mode={mode:<13} seed={seed:>3}  "
            f"all={s['ic_all']:+.4f}  cat={s['ic_catalyst']:+.4f}  "
            f"calm={s['ic_calm']:+.4f}  n_cat={s['n_cat_cells']}"
        )

    print()
    print("Per-mode summary (mean over seeds):")
    print(f"{'mode':<14} {'n':>2}  {'IC_all':>7}  {'IC_cat':>7}  {'IC_calm':>7}  {'cat-calm':>8}")
    for mode, rows in by_mode.items():
        def _m(key):
            vals = [r[key] for r in rows if not np.isnan(r[key])]
            return mean(vals) if vals else float("nan")
        ma = _m("ic_all"); mc = _m("ic_catalyst"); mcm = _m("ic_calm")
        diff = mc - mcm if not (np.isnan(mc) or np.isnan(mcm)) else float("nan")
        print(f"{mode:<14} {len(rows):>2}  {ma:+.4f}  {mc:+.4f}  {mcm:+.4f}  {diff:+.4f}")


if __name__ == "__main__":
    main()
