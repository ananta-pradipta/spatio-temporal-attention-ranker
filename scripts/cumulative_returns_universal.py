"""Two-panel cumulative-return figure for the universal panel.

Reference style: FactorVAE Figure 6 (cumulative excess return + cumulative
return of portfolios). Adapted honestly to our 5-fold walk-forward
protocol: folds are INDEPENDENT walk-forward tests with re-training and a
5-day embargo between them, so the curve is concatenated left-to-right
with an equity RESET at each fold boundary (vertical separators, F1-F5
labels). It is not a single continuous tradable backtest.

Panel (a): cumulative excess return of a dollar-neutral long-short
  top-25/bottom-25 book per model (this is itself an excess series).
Panel (b): cumulative return of the long top-quintile book per model,
  against an equal-weight universe-mean benchmark.

Returns are compounded on a NON-OVERLAPPING 5-day rebalanced schedule
(every horizon-th test day) so the curve is economically interpretable,
matching the max-drawdown convention already used in the paper.

Outputs:
  drafts/universal_paper_aaai/figures/cumulative_returns.pdf  (and KDD)
  results/exports/cumulative_returns_universal.csv
"""
from __future__ import annotations

import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT_DIR = Path("results/exports")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "RAG-STAR":     ("results/rag_star_universe_v2_no_ipo", "#1f3b6b"),
    "MASTER":       ("results/baselines_universal_two_regime_val/master", "#9c4221"),
    "FactorVAE":    ("results/baselines_universal_two_regime_val/factorvae", "#2e7d32"),
    "iTransformer": ("results/baselines_universal_two_regime_val/itransformer", "#7c3aed"),
    "StockMixer":   ("results/baselines_universal_two_regime_val/stockmixer", "#666666"),
}
FOLDS = (1, 2, 3, 4, 5)
FOLD_LABEL = {1: "F1 2020", 2: "F2 2021-22", 3: "F3 2022-23",
              4: "F4 2024", 5: "F5 2025"}
K = 25
HORIZON = 5


def load_avg(model_dir: str, fold: int):
    files = sorted(glob.glob(f"{model_dir}/fold{fold}_seed*_predictions.npz"))
    if not files:
        return None
    ys, y_true, mask = [], None, None
    for f in files:
        d = np.load(f, allow_pickle=True)
        ys.append(d["y_hat"])
        if y_true is None:
            y_true = d["y_true"]
            for nm in ("loss_mask", "eval_mask", "mask"):
                if nm in d:
                    mask = d[nm].astype(bool)
                    break
    return np.mean(np.stack(ys), axis=0), y_true, mask


def fold_series(yhat, y_true, mask):
    """Per test-day long-short, long-top-quintile, and benchmark returns."""
    T = yhat.shape[0]
    ls, lq, bm = [], [], []
    for t in range(T):
        m = mask[t]
        if int(m.sum()) < 2 * K:
            continue
        idx = np.flatnonzero(m)
        order = np.argsort(yhat[t, idx])
        top = idx[order[-K:]]
        bot = idx[order[:K]]
        q = max(1, len(idx) // 5)
        topq = idx[order[-q:]]
        ls.append(float(y_true[t, top].mean() - y_true[t, bot].mean()))
        lq.append(float(y_true[t, topq].mean()))
        bm.append(float(y_true[t, idx].mean()))
    return np.array(ls), np.array(lq), np.array(bm)


def cum_nonoverlap(r: np.ndarray) -> np.ndarray:
    """Cumulative simple return on a non-overlapping 5-day schedule, %."""
    steps = r[::HORIZON]
    return np.cumsum(steps) * 100.0


def main():
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    ax_a, ax_b = axes

    # Precompute benchmark (model-independent) per fold from any model's
    # y_true/mask; use RAG-STAR's files as the source of the grid.
    per_model_curves = {nm: {"a": [], "b": []} for nm in MODELS}
    bench_curves = []
    fold_lengths = []

    for fold in FOLDS:
        bench_done = False
        for nm, (mdir, _c) in MODELS.items():
            res = load_avg(mdir, fold)
            if res is None:
                per_model_curves[nm]["a"].append(np.array([]))
                per_model_curves[nm]["b"].append(np.array([]))
                continue
            yhat, y_true, mask = res
            ls, lq, bm = fold_series(yhat, y_true, mask)
            ca = cum_nonoverlap(ls)
            cb = cum_nonoverlap(lq)
            per_model_curves[nm]["a"].append(ca)
            per_model_curves[nm]["b"].append(cb)
            if not bench_done:
                bench_curves.append(cum_nonoverlap(bm))
                fold_lengths.append(len(cum_nonoverlap(bm)))
                bench_done = True
            for i, (va, vb) in enumerate(zip(ca, cb)):
                rows.append({"fold": fold, "model": nm, "step": i,
                             "cum_excess_pct": va, "cum_return_pct": vb})
        if not bench_done:
            bench_curves.append(np.array([]))
            fold_lengths.append(0)

    # Concatenate with reset per fold; track boundaries.
    boundaries = np.cumsum([0] + fold_lengths)

    for nm, (_mdir, color) in MODELS.items():
        xa_all, ya_all, yb_all = [], [], []
        for fi, fold in enumerate(FOLDS):
            ca = per_model_curves[nm]["a"][fi]
            cb = per_model_curves[nm]["b"][fi]
            x0 = boundaries[fi]
            xa_all.append(np.arange(x0, x0 + len(ca)))
            ya_all.append(ca)
            yb_all.append(cb)
        if sum(len(v) for v in ya_all):
            ax_a.plot(np.concatenate(xa_all), np.concatenate(ya_all),
                      color=color, lw=1.2, label=nm)
            ax_b.plot(np.concatenate(xa_all), np.concatenate(yb_all),
                      color=color, lw=1.2, label=nm)

    bx, by = [], []
    for fi in range(len(FOLDS)):
        bc = bench_curves[fi]
        x0 = boundaries[fi]
        bx.append(np.arange(x0, x0 + len(bc)))
        by.append(bc)
    if sum(len(v) for v in by):
        ax_b.plot(np.concatenate(bx), np.concatenate(by), color="black",
                  lw=1.0, ls="--", label="Universe (eq-wt)")

    for ax, title, ylab in [
        (ax_a, "(a) Cumulative excess return (long-short top/bottom 25)",
         "Cumulative excess return (%)"),
        (ax_b, "(b) Cumulative return (long top-quintile)",
         "Cumulative return (%)")]:
        for fi, b in enumerate(boundaries[1:-1], start=1):
            ax.axvline(b, color="grey", lw=0.6, ls=":")
        y0, y1 = ax.get_ylim()
        ax.set_ylim(y0, y1 + 0.12 * (y1 - y0))
        ylab_y = y1 + 0.02 * (y1 - y0)
        for fi in range(len(FOLDS)):
            mid = (boundaries[fi] + boundaries[fi + 1]) / 2
            ax.text(mid, ylab_y, FOLD_LABEL[FOLDS[fi]],
                    ha="center", va="bottom", fontsize=6.5, color="grey")
        ax.axhline(0, color="black", lw=0.5)
        ax.set_title(title, fontsize=9, pad=10)
        ax.set_ylabel(ylab, fontsize=8)
        ax.set_xlabel("non-overlapping 5-day rebalance step "
                      "(folds are independent walk-forward tests)",
                      fontsize=7)
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=7)
    ax_a.legend(loc="upper left", fontsize=7, ncol=2)

    plt.tight_layout()
    for out_dir in [Path("drafts/universal_paper_aaai/figures"),
                    Path("drafts/universal_paper_kdd/figures")]:
        out_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_dir / "cumulative_returns.pdf", bbox_inches="tight")
    plt.close()
    pd.DataFrame(rows).to_csv(
        OUT_DIR / "cumulative_returns_universal.csv", index=False)
    print("Saved cumulative_returns.pdf to 2 dirs;",
          "fold_lengths=", fold_lengths)


if __name__ == "__main__":
    main()
