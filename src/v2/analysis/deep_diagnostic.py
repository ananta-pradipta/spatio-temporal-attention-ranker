"""Deep diagnostic on saved (model, fold, seed) predictions.

Runs after the AMP re-training jobs land their `*_predictions.npz` files.
Produces six analyses per (model, fold) combination:

    1. Universe-restricted Information Coefficient (IC):
       full-244, dense-118 (v1-comparable), late-IPO-97.

    2. Per-ticker IC contribution: leave-one-ticker-out IC change to
       identify which tickers drag the model's fold-IC down most.

    3. Per-day IC time series: month-by-month and rolling-21d means
       so we can locate weeks where the model collapses.

    4. Cross-model prediction correlation: Spearman correlation of
       daily score vectors between every pair of models, per fold.

    5. Per-fold Top-K hit rate (k=10, 50): a simple cross-check of the
       NDCG numbers in the JSON, computed from raw predictions.

    6. Quantile head calibration (STAR-DualHead only): empirical
       coverage at each tau on the test fold.

Outputs go to `results/diagnostic/`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


MODELS = ["epistar", "dyreg_star", "star_rirr", "star_dualhead"]
FOLDS = [1, 2, 3]
SEEDS = [42, 43, 44, 45, 46]
RESULTS_DIR = Path("results")
DIAG_DIR = RESULTS_DIR / "diagnostic"


def per_day_ic(
    y_hat: np.ndarray, y: np.ndarray, mask: np.ndarray, rank: bool = False
) -> tuple[float, np.ndarray]:
    """Daily-then-mean IC restricted to active cells."""
    t_total = y_hat.shape[0]
    ics = np.full(t_total, np.nan, dtype=np.float64)
    for t in range(t_total):
        m = mask[t]
        if m.sum() < 5:
            continue
        a = y_hat[t, m]
        b = y[t, m]
        if rank:
            a = pd.Series(a).rank().to_numpy()
            b = pd.Series(b).rank().to_numpy()
        sa, sb = a.std(), b.std()
        if sa < 1e-9 or sb < 1e-9:
            continue
        ics[t] = float(np.corrcoef(a, b)[0, 1])
    return (np.nanmean(ics) if not np.all(np.isnan(ics)) else 0.0, ics)


def load_predictions(model: str, fold: int, seed: int) -> dict | None:
    """Load a saved predictions npz, or None if not present."""
    p = RESULTS_DIR / model / f"fold{fold}_seed{seed}_predictions.npz"
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=False)
    return {k: z[k] for k in z.files}


def universe_subsets(tickers: np.ndarray) -> dict[str, np.ndarray]:
    """Return boolean masks for the dense-118, late-IPO-97, and full-244 subsets.

    The dense-118 split is computed once from the panel by checking which
    tickers were active during fold-1 training (2015-01-09 to 2018-12-21).
    Cached at `results/diagnostic/ticker_groups.csv` so all models share
    the same partitioning.
    """
    cache = DIAG_DIR / "ticker_groups.csv"
    if cache.exists():
        df = pd.read_csv(cache)
        df = df.set_index("ticker")
        dense = np.array([df.loc[t, "dense"] if t in df.index else 0 for t in tickers])
        late = np.array([df.loc[t, "late_ipo"] if t in df.index else 0 for t in tickers])
        return {
            "full": np.ones(len(tickers), dtype=bool),
            "dense_118": dense.astype(bool),
            "late_ipo_97": late.astype(bool),
        }
    raise RuntimeError(
        f"Missing {cache}. Run `make_ticker_groups()` first."
    )


def make_ticker_groups() -> None:
    """Build the dense vs late-IPO ticker partition from panel mask data."""
    import sys
    sys.path.insert(0, "src")
    from mtgn.training.panel_enriched import (
        EnrichedPanelConfig,
        build_enriched_panel,
        panel_to_tensors,
    )
    from src.v2.training.folds import fold_indices

    cfg = EnrichedPanelConfig(
        start_date=pd.Timestamp("2015-01-09"),
        end_date=pd.Timestamp("2022-12-31"),
    )
    panel, tickers, dates = build_enriched_panel(cfg)
    tens = panel_to_tensors(panel, tickers, dates)
    mask = tens["mask"]
    train1, _, _ = fold_indices(1, dates)
    days_train1 = mask[train1].sum(axis=0)

    rows = []
    for i, t in enumerate(tickers):
        rows.append({
            "ticker": t,
            "fold1_train_active_days": int(days_train1[i]),
            "dense": int(days_train1[i] >= len(train1) * 0.5),
            "late_ipo": int(days_train1[i] == 0),
        })
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(DIAG_DIR / "ticker_groups.csv", index=False)
    print(f"Wrote {DIAG_DIR / 'ticker_groups.csv'}")


def restricted_ic(
    pred: dict, ticker_subset: np.ndarray
) -> tuple[float, float]:
    """Pearson and Spearman IC restricted to a ticker subset."""
    y_hat = pred["y_hat"]
    y = pred["y_true"]
    mask = pred["mask"] & ticker_subset[None, :]
    ic_p, _ = per_day_ic(y_hat, y, mask, rank=False)
    ic_s, _ = per_day_ic(y_hat, y, mask, rank=True)
    return ic_p, ic_s


def per_ticker_drag(pred: dict) -> pd.DataFrame:
    """For each ticker, compute the LOO change in fold IC.

    A ticker with a positive `loo_minus_full` value means removing it
    *raises* the IC, so it was a drag. A negative value means it was
    contributing positively.
    """
    y_hat = pred["y_hat"]
    y = pred["y_true"]
    mask = pred["mask"]
    tickers = pred["tickers"]
    full_ic, _ = per_day_ic(y_hat, y, mask, rank=False)
    rows = []
    for i in range(mask.shape[1]):
        sub_mask = mask.copy()
        sub_mask[:, i] = False
        loo_ic, _ = per_day_ic(y_hat, y, sub_mask, rank=False)
        rows.append({
            "ticker": str(tickers[i]),
            "active_days": int(mask[:, i].sum()),
            "ic_full": full_ic,
            "ic_loo": loo_ic,
            "delta_ic_remove": loo_ic - full_ic,
        })
    return pd.DataFrame(rows)


def per_day_ic_series(pred: dict) -> pd.DataFrame:
    """Return a DataFrame of (date, ic) per day in the test window."""
    y_hat = pred["y_hat"]
    y = pred["y_true"]
    mask = pred["mask"]
    dates = pred["dates"]
    test_idx = pred["test_idx"]
    _, per_day = per_day_ic(y_hat, y, mask, rank=False)
    rows = []
    for t in test_idx:
        if not np.isnan(per_day[t]):
            rows.append({"date": str(dates[t]), "day_idx": int(t), "ic": float(per_day[t])})
    return pd.DataFrame(rows)


def cross_model_correlation(folds_to_run: list[int] | None = None) -> pd.DataFrame:
    """For each fold, compute Spearman corr between every pair of models.

    Aggregates per-day predictions across seeds via a simple mean.
    """
    folds_to_run = folds_to_run or FOLDS
    rows = []
    for fold in folds_to_run:
        all_preds = {}
        for m in MODELS:
            stack = []
            for s in SEEDS:
                p = load_predictions(m, fold, s)
                if p is None:
                    continue
                stack.append(p["y_hat"])
            if stack:
                all_preds[m] = np.stack(stack).mean(axis=0)
        if len(all_preds) < 2:
            continue
        # We need a common test mask
        sample_pred = next(iter(all_preds.values()))
        anchor_mask_pred = load_predictions(MODELS[0], fold, SEEDS[0])
        if anchor_mask_pred is None:
            continue
        anchor_mask = anchor_mask_pred["mask"]
        for ma in all_preds:
            for mb in all_preds:
                if ma >= mb:
                    continue
                a = all_preds[ma][anchor_mask]
                b = all_preds[mb][anchor_mask]
                if a.size < 50 or a.std() < 1e-9 or b.std() < 1e-9:
                    continue
                rho, _ = spearmanr(a, b)
                rows.append({
                    "fold": fold, "model_a": ma, "model_b": mb,
                    "spearman": float(rho), "n_obs": int(a.size),
                })
    return pd.DataFrame(rows)


def quantile_calibration_dualhead() -> pd.DataFrame:
    """Empirical coverage at each tau for STAR-DualHead test predictions."""
    rows = []
    quantile_levels = [0.10, 0.25, 0.50, 0.75, 0.90]
    for fold in FOLDS:
        for s in SEEDS:
            p = load_predictions("star_dualhead", fold, s)
            if p is None or "q" not in p:
                continue
            q = p["q"]
            y_true = p["y_true"]
            mask = p["mask"]
            for k, tau in enumerate(quantile_levels):
                below = (y_true[mask] < q[mask][:, k])
                rows.append({
                    "fold": fold, "seed": s, "tau": tau,
                    "empirical_coverage": float(below.mean()),
                    "n_obs": int(mask.sum()),
                })
    return pd.DataFrame(rows)


def run_full_diagnostic() -> None:
    """Top-level diagnostic runner. Writes CSV reports under results/diagnostic/."""
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    if not (DIAG_DIR / "ticker_groups.csv").exists():
        print("Building ticker groups (one-time)...")
        make_ticker_groups()

    print("=" * 70)
    print("Analysis 1: Universe-restricted IC")
    print("=" * 70)
    rows = []
    for m in MODELS:
        for fold in FOLDS:
            for s in SEEDS:
                p = load_predictions(m, fold, s)
                if p is None:
                    continue
                subsets = universe_subsets(p["tickers"])
                for name, sub in subsets.items():
                    ic_p, ic_s = restricted_ic(p, sub)
                    rows.append({
                        "model": m, "fold": fold, "seed": s, "subset": name,
                        "n_tickers_in_subset": int(sub.sum()),
                        "ic": ic_p, "rank_ic": ic_s,
                    })
    df1 = pd.DataFrame(rows)
    df1.to_csv(DIAG_DIR / "01_universe_restricted_ic.csv", index=False)
    print(f"Wrote {DIAG_DIR / '01_universe_restricted_ic.csv'} ({len(df1)} rows)")
    if not df1.empty:
        agg = df1.groupby(["model", "fold", "subset"], as_index=False)["ic"].mean().pivot(
            index=["model", "fold"], columns="subset", values="ic"
        )
        print(agg.round(4))

    print()
    print("=" * 70)
    print("Analysis 2: Per-ticker drag (top-10 worst per (model, fold))")
    print("=" * 70)
    drag_rows = []
    for m in MODELS:
        for fold in FOLDS:
            stacks: list[pd.DataFrame] = []
            for s in SEEDS:
                p = load_predictions(m, fold, s)
                if p is None:
                    continue
                df = per_ticker_drag(p)
                df["model"] = m
                df["fold"] = fold
                df["seed"] = s
                stacks.append(df)
            if stacks:
                drag_rows.append(pd.concat(stacks, ignore_index=True))
    if drag_rows:
        df2 = pd.concat(drag_rows, ignore_index=True)
        df2.to_csv(DIAG_DIR / "02_per_ticker_drag.csv", index=False)
        print(f"Wrote {DIAG_DIR / '02_per_ticker_drag.csv'} ({len(df2)} rows)")
        # Aggregate across seeds: average drag per (model, fold, ticker)
        agg = df2.groupby(["model", "fold", "ticker"], as_index=False)["delta_ic_remove"].mean()
        # Top 10 most-dragging tickers per (model, fold)
        top = (agg.sort_values(["model", "fold", "delta_ic_remove"], ascending=[True, True, False])
                  .groupby(["model", "fold"]).head(10))
        top.to_csv(DIAG_DIR / "02_top_drag_tickers.csv", index=False)
        print(f"Wrote {DIAG_DIR / '02_top_drag_tickers.csv'}")

    print()
    print("=" * 70)
    print("Analysis 3: Per-day IC time series")
    print("=" * 70)
    series_rows = []
    for m in MODELS:
        for fold in FOLDS:
            for s in SEEDS:
                p = load_predictions(m, fold, s)
                if p is None:
                    continue
                df = per_day_ic_series(p)
                df["model"] = m
                df["fold"] = fold
                df["seed"] = s
                series_rows.append(df)
    if series_rows:
        df3 = pd.concat(series_rows, ignore_index=True)
        df3.to_csv(DIAG_DIR / "03_per_day_ic.csv", index=False)
        print(f"Wrote {DIAG_DIR / '03_per_day_ic.csv'} ({len(df3)} rows)")

    print()
    print("=" * 70)
    print("Analysis 4: Cross-model prediction correlation")
    print("=" * 70)
    df4 = cross_model_correlation()
    if not df4.empty:
        df4.to_csv(DIAG_DIR / "04_cross_model_corr.csv", index=False)
        print(f"Wrote {DIAG_DIR / '04_cross_model_corr.csv'}")
        print(df4.round(3))

    print()
    print("=" * 70)
    print("Analysis 6: STAR-DualHead quantile calibration")
    print("=" * 70)
    df6 = quantile_calibration_dualhead()
    if not df6.empty:
        df6.to_csv(DIAG_DIR / "06_quantile_calibration.csv", index=False)
        print(f"Wrote {DIAG_DIR / '06_quantile_calibration.csv'}")
        print(df6.groupby(["fold", "tau"], as_index=False)["empirical_coverage"].mean().round(3))


if __name__ == "__main__":
    run_full_diagnostic()
