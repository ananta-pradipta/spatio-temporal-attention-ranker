"""Per-fold robustness charts comparing RAG-STAR to baselines.

For each of the three folds, plots:
    - 20-day rolling per-day IC for RAG-STAR, MASTER, StockMixer
    - Key regime events as vertical markers with short labels
    - Fold mean IC as a horizontal reference line per model

Saves three figures (regime_fold1.pdf, regime_fold2.pdf, regime_fold3.pdf)
into both venue figure directories.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd


DATA_DIR = Path("results/exports")
OUT_DIRS = [
    Path("drafts/paper_aaai/figures"),
    Path("drafts/paper_kdd/figures"),
]

MODEL_STYLE = {
    "RAG-STAR":   {"color": "#1f3b6b", "lw": 1.4, "alpha": 0.95, "zorder": 5},
    "MASTER":     {"color": "#9c4221", "lw": 1.0, "alpha": 0.85, "zorder": 4},
    "StockMixer": {"color": "#666666", "lw": 0.9, "alpha": 0.75, "zorder": 3},
}

FOLD_TITLES = {
    1: "Fold 1: COVID crash and the mRNA rocket-ship era (2020)",
    2: "Fold 2: SPAC peak through rate-hike bear (2021-H2 to 2022-H1)",
    3: "Fold 3: Deep biotech bear and late-2022 Fed pivot (2022-H2)",
}

FOLD_EVENTS = {
    1: [
        ("2020-03-23", "COVID trough"),
        ("2020-08-15", "vaccine trial\ndata begins"),
        ("2020-12-11", "mRNA FDA EUA"),
    ],
    2: [
        ("2021-09-01", "SPAC peak"),
        ("2021-11-30", "Powell hawkish"),
        ("2022-03-16", "1st 25 bp hike"),
        ("2022-05-04", "50 bp hike"),
    ],
    3: [
        ("2022-09-21", "75 bp hike"),
        ("2022-10-13", "CPI shock"),
        ("2022-11-30", "Fed dovish tone"),
        ("2022-12-14", "FOMC pivot"),
    ],
}


def render_fold(fold: int) -> None:
    csv_path = DATA_DIR / f"per_day_ic_fold{fold}.csv"
    if not csv_path.exists():
        print(f"[fold{fold}] missing {csv_path}")
        return
    df = pd.read_csv(csv_path, parse_dates=["date"])

    fig, ax = plt.subplots(figsize=(10, 3.5))

    # Per-model 20-day rolling IC line, with raw daily IC underlaid as
    # a thin transparent trace so the reader can see the true per-day
    # values that the rolling line is smoothing.
    fold_means = {}
    for model_name, style in MODEL_STYLE.items():
        sub = df[df["model"] == model_name].sort_values("date").reset_index(drop=True)
        if sub.empty:
            continue
        # Raw daily (true) IC behind the rolling line
        ax.plot(sub["date"], sub["daily_ic"],
                color=style["color"], lw=0.5, alpha=0.20,
                zorder=style["zorder"] - 2)
        # Rolling already computed; if missing, recompute
        if "rolling_ic_20d" in sub.columns and not sub["rolling_ic_20d"].isna().all():
            rol = sub["rolling_ic_20d"]
        else:
            rol = sub["daily_ic"].rolling(20, min_periods=10).mean()
        ax.plot(sub["date"], rol, label=model_name, **style)
        fold_means[model_name] = float(np.nanmean(sub["daily_ic"]))
        # horizontal reference line for the fold mean
        ax.axhline(fold_means[model_name],
                   color=style["color"], lw=0.6, ls=":", alpha=0.5, zorder=2)

    # Zero reference
    ax.axhline(0.0, color="black", lw=0.5, alpha=0.4, zorder=1)

    # Event annotations
    y_lim = ax.get_ylim()
    y_top = y_lim[1]
    y_offset = (y_top - y_lim[0]) * 0.08
    for date_str, label in FOLD_EVENTS[fold]:
        d = pd.Timestamp(date_str)
        if d < df["date"].min() or d > df["date"].max():
            continue
        ax.axvline(d, color="gray", lw=0.5, alpha=0.5, zorder=1)
        ax.annotate(label, xy=(d, y_top - y_offset),
                    fontsize=7.0, ha="center", va="top",
                    color="dimgray", zorder=4)

    # X axis formatting
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.tick_params(axis="x", which="major", length=4)

    # Cosmetics
    ax.set_ylabel("Daily IC (raw light, 20-day rolling bold)")
    ax.set_title(FOLD_TITLES[fold], fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.4)

    # Legend with fold mean values
    handles = []
    labels = []
    for model_name, style in MODEL_STYLE.items():
        if model_name in fold_means:
            (h,) = ax.plot([], [], label=model_name, **style)
            handles.append(h)
            labels.append(f"{model_name}  (fold mean IC = {fold_means[model_name]:+.4f})")
    ax.legend(handles, labels, loc="lower left", fontsize=7.5,
              framealpha=0.85, handlelength=2.0)

    fig.tight_layout(pad=0.4)
    for od in OUT_DIRS:
        od.mkdir(parents=True, exist_ok=True)
        fig.savefig(od / f"regime_fold{fold}.pdf", dpi=150, bbox_inches="tight")
    print(f"[fold{fold}] saved with mean IC = {fold_means}")
    plt.close(fig)


def main() -> None:
    for fold in (1, 2, 3):
        render_fold(fold)


if __name__ == "__main__":
    main()
