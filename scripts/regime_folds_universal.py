"""5-fold cumulative-IC chart for the universal-panel paper (Figure 5).

Self-contained: reads universal-panel prediction npz files, averages across
seeds, computes daily Pearson IC, and renders a 5-panel figure (one panel
per fold) of the CUMULATIVE-AVERAGE daily IC per model.

Design goals (2026-05-15 rework): make model differences legible and make
RAG-STAR's competitiveness explicit.
  - No raw per-day noise traces (they buried the signal).
  - Cumulative-average IC (running mean from fold start): no arbitrary
    smoothing window, converges to the fold-mean IC, so "who is ahead and
    stays ahead" is unambiguous.
  - Baselines drawn as a grey min-max envelope plus thin muted lines;
    RAG-STAR drawn bold and high-contrast on top.
  - Light-green shading wherever RAG-STAR's cumulative IC exceeds the best
    baseline (RAG-STAR strictly ahead of every baseline).
"""
from __future__ import annotations

import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

SYS = "RAG-STAR"
SYS_DIR = "results/rag_star_universe_v2_no_ipo"
SYS_COLOR = "#15396b"
BASELINES = {
    "MASTER":     ("results/baselines_universal_two_regime_val/master", "#c0712f"),
    "FactorVAE":  ("results/baselines_universal_two_regime_val/factorvae", "#3f8f43"),
    "iTransformer": ("results/baselines_universal_two_regime_val/itransformer", "#8a6fc0"),
    "StockMixer": ("results/baselines_universal_two_regime_val/stockmixer", "#9a9a9a"),
}
FOLDS = (1, 2, 3, 4, 5)
FOLD_TITLES = {
    1: "F1: COVID crash + recovery (2020)",
    2: "F2: rate-hike rotation (2021-H2 to 2022-H1)",
    3: "F3: post-shock + banking stress (2022-H2 to 2023-H1)",
    4: "F4: AI mega-cap rally (2024)",
    5: "F5: Fed-cut + post-election (2025-H2)",
}


def per_day_ic(y_hat, y, mask):
    T = y_hat.shape[0]
    out = np.full(T, np.nan)
    for t in range(T):
        m = mask[t]
        if m.sum() < 5:
            continue
        a, b = y_hat[t, m], y[t, m]
        if a.std() < 1e-9 or b.std() < 1e-9:
            continue
        out[t] = float(np.corrcoef(a, b)[0, 1])
    return out


def load_avg(model_dir, fold):
    files = sorted(glob.glob(f"{model_dir}/fold{fold}_seed*_predictions.npz"))
    if not files:
        return None
    ys, y_true, mask, dates = [], None, None, None
    for f in files:
        d = np.load(f, allow_pickle=True)
        ys.append(d["y_hat"])
        if y_true is None:
            y_true = d["y_true"]
            for nm in ("loss_mask", "eval_mask", "mask"):
                if nm in d:
                    mask = d[nm].astype(bool); break
            if "dates" in d:
                dates = pd.to_datetime(d["dates"])
    yhat = np.mean(np.stack(ys), axis=0)
    row_active = mask.sum(axis=1) >= 5
    test_rows = np.flatnonzero(row_active)
    test_dates = dates[test_rows] if dates is not None else None
    return yhat, y_true, mask, test_rows, test_dates


def cum_ic(model_dir, fold):
    """Cumulative-average daily IC + fold-mean IC + x dates for a model."""
    res = load_avg(model_dir, fold)
    if res is None:
        return None
    yhat, y_true, mask, test_rows, test_dates = res
    ic = per_day_ic(yhat, y_true, mask)[test_rows]
    s = pd.Series(ic)
    cm = s.expanding(min_periods=5).mean().values
    x = test_dates if test_dates is not None else np.arange(len(ic))
    return x, cm, float(np.nanmean(ic))


FOLD_SHORT = {1: "F1 2020", 2: "F2 2021-22", 3: "F3 2022-23",
              4: "F4 2024", 5: "F5 2025"}


def main():
    # Single wide panel: the five per-fold cumulative-average IC curves
    # concatenated left-to-right. Folds are independent walk-forward
    # tests, so each fold's cumulative mean restarts at its boundary;
    # dotted separators and F1-F5 labels mark the segments.
    fig, ax = plt.subplots(1, 1, figsize=(10, 3.8))

    # Gather per-fold curves keyed by model.
    per_model = {SYS: []}
    for nm in BASELINES:
        per_model[nm] = []
    fold_len, sys_fold_mean = [], []
    for fold in FOLDS:
        rs = cum_ic(SYS_DIR, fold)
        L = len(rs[1]) if rs is not None else 0
        fold_len.append(L)
        sys_fold_mean.append(rs[2] if rs is not None else float("nan"))
        per_model[SYS].append(rs[1] if rs is not None else np.array([]))
        for nm, (mdir, _c) in BASELINES.items():
            r = cum_ic(mdir, fold)
            per_model[nm].append(r[1] if r is not None else np.array([]))
    bnd = np.cumsum([0] + fold_len)

    # Baseline envelope per fold, then concatenated.
    env_lo, env_hi, env_x = [], [], []
    for fi in range(len(FOLDS)):
        cols = [per_model[nm][fi] for nm in BASELINES
                if len(per_model[nm][fi])]
        x0 = bnd[fi]
        if cols:
            B = np.vstack(cols)
            env_lo.append(np.nanmin(B, axis=0))
            env_hi.append(np.nanmax(B, axis=0))
            env_x.append(np.arange(x0, x0 + B.shape[1]))
    if env_x:
        ex = np.concatenate(env_x)
        elo = np.concatenate(env_lo)
        ehi = np.concatenate(env_hi)
        ok = ~np.isnan(elo)
        ax.fill_between(ex, elo, ehi, where=ok, color="#9a9a9a",
                        alpha=0.20, zorder=2, label="baseline range")

    handles, labels = [], []
    for nm, (_m, color) in BASELINES.items():
        xs, ys = [], []
        for fi in range(len(FOLDS)):
            c = per_model[nm][fi]
            xs.append(np.arange(bnd[fi], bnd[fi] + len(c)))
            ys.append(c)
        if sum(len(v) for v in ys):
            (ln,) = ax.plot(np.concatenate(xs), np.concatenate(ys),
                            color=color, lw=0.9, alpha=0.7, zorder=4)
            handles.append(ln); labels.append(nm)

    xs, ys = [], []
    for fi in range(len(FOLDS)):
        c = per_model[SYS][fi]
        xs.append(np.arange(bnd[fi], bnd[fi] + len(c)))
        ys.append(c)
    if sum(len(v) for v in ys):
        X = np.concatenate(xs); Y = np.concatenate(ys)
        (lr,) = ax.plot(X, Y, color=SYS_COLOR, lw=2.4, zorder=10,
                        solid_capstyle="round")
        handles = [lr] + handles
        labels = [f"$\\bf{{{SYS}}}$"] + labels
        if env_x:
            lead = Y > ehi
            ax.fill_between(X, elo, Y,
                            where=lead & ~np.isnan(Y) & ~np.isnan(ehi),
                            color="#2e7d32", alpha=0.16, zorder=3)

    ax.axhline(0, color="black", lw=0.6)
    y0, y1 = ax.get_ylim()
    ax.set_ylim(y0, y1 + 0.16 * (y1 - y0))
    for b in bnd[1:-1]:
        ax.axvline(b, color="grey", lw=0.6, ls=":")
    for fi in range(len(FOLDS)):
        mid = (bnd[fi] + bnd[fi + 1]) / 2
        ax.text(mid, ax.get_ylim()[1], FOLD_SHORT[FOLDS[fi]],
                ha="center", va="top", fontsize=7, color="#333333")
    ax.set_ylabel("Cumulative mean IC", fontsize=8)
    ax.set_xlabel("fold-sequential trading-day index "
                  "(folds are independent walk-forward tests; "
                  "cumulative mean resets per fold)", fontsize=7)
    ax.grid(True, alpha=0.22)
    ax.margins(x=0.01)
    ax.tick_params(labelsize=7)
    ax.set_xticks([])
    ax.legend(handles, labels, loc="lower right", fontsize=7, ncol=3,
              frameon=True, framealpha=0.92, borderpad=0.5)
    plt.tight_layout()
    for out_dir in [Path("drafts/universal_paper_aaai/figures"),
                    Path("drafts/universal_paper_kdd/figures")]:
        out_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_dir / "regime_folds_combined.pdf", bbox_inches="tight")
    plt.close()
    print("Saved regime_folds_combined.pdf (5-fold cumulative IC) to 2 dirs")


if __name__ == "__main__":
    main()
