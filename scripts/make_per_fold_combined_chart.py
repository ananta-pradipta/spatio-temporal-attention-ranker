"""Combined per-fold robustness chart for the RAG-STAR paper.

Three stacked panels (one per fold) sharing the same y-axis style and
the same model legend, in a single PDF that spans the full text width
of a 2-column paper. Each panel shows raw daily IC (thin) and 20-day
rolling IC (bold) for RAG-STAR, MASTER, FactorVAE, and StockMixer with
that fold's regime events as vertical markers.

Saves drafts/paper_aaai/figures/regime_folds_combined.pdf and the KDD
copy.
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
    "FactorVAE":  {"color": "#2e7d32", "lw": 1.0, "alpha": 0.85, "zorder": 4},
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


def render_panel(ax, fold: int) -> None:
    csv_path = DATA_DIR / f"per_day_ic_fold{fold}.csv"
    if not csv_path.exists():
        ax.text(0.5, 0.5, f"missing {csv_path}", ha="center", va="center",
                transform=ax.transAxes)
        return
    df = pd.read_csv(csv_path, parse_dates=["date"])

    fold_means = {}
    for model_name, style in MODEL_STYLE.items():
        sub = df[df["model"] == model_name].sort_values("date").reset_index(drop=True)
        if sub.empty:
            continue
        ax.plot(sub["date"], sub["daily_ic"],
                color=style["color"], lw=0.5, alpha=0.20,
                zorder=style["zorder"] - 2)
        if "rolling_ic_20d" in sub.columns and not sub["rolling_ic_20d"].isna().all():
            rol = sub["rolling_ic_20d"]
        else:
            rol = sub["daily_ic"].rolling(20, min_periods=10).mean()
        ax.plot(sub["date"], rol, label=model_name, **style)
        fold_means[model_name] = float(np.nanmean(sub["daily_ic"]))
        ax.axhline(fold_means[model_name],
                   color=style["color"], lw=0.6, ls=":", alpha=0.5, zorder=2)

    ax.axhline(0.0, color="black", lw=0.5, alpha=0.4, zorder=1)

    y_lim = ax.get_ylim()
    y_top = y_lim[1]
    y_offset = (y_top - y_lim[0]) * 0.08
    df_min, df_max = df["date"].min(), df["date"].max()
    for date_str, label in FOLD_EVENTS[fold]:
        d = pd.Timestamp(date_str)
        if d < df_min or d > df_max:
            continue
        ax.axvline(d, color="gray", lw=0.5, alpha=0.5, zorder=1)
        ax.annotate(label, xy=(d, y_top - y_offset),
                    fontsize=6.5, ha="center", va="top",
                    color="dimgray", zorder=4)

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.tick_params(axis="x", which="major", length=4, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)

    ax.set_ylabel("Daily IC", fontsize=8)
    ax.set_title(FOLD_TITLES[fold], fontsize=9, loc="left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.4)

    # Legend per-panel only on first fold to save vertical space
    if fold == 1:
        handles, labels = [], []
        for model_name, style in MODEL_STYLE.items():
            if model_name in fold_means:
                (h,) = ax.plot([], [], **style)
                handles.append(h)
                labels.append(f"{model_name} ({fold_means[model_name]:+.4f})")
        ax.legend(handles, labels, loc="lower left", fontsize=7,
                  framealpha=0.85, handlelength=2.0,
                  title="Model (fold mean IC)", title_fontsize=7)
    else:
        # Annotate fold means on the right side as text since no legend
        info = "  ".join([f"{m}: {fold_means[m]:+.4f}" for m in MODEL_STYLE if m in fold_means])
        ax.text(0.99, 0.05, info, transform=ax.transAxes, fontsize=6.5,
                ha="right", va="bottom", color="dimgray")


def main() -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11, 6.6))
    for fold, ax in zip((1, 2, 3), axes):
        render_panel(ax, fold)
    fig.tight_layout(pad=0.8, h_pad=0.8)

    for od in OUT_DIRS:
        od.mkdir(parents=True, exist_ok=True)
        fig.savefig(od / "regime_folds_combined.pdf", dpi=150, bbox_inches="tight")
    print(f"Saved regime_folds_combined.pdf to {len(OUT_DIRS)} dirs")
    plt.close(fig)


if __name__ == "__main__":
    main()
