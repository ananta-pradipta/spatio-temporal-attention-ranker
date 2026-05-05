"""Introduction-section motivation chart for the RAG-STAR paper.

Three-panel time series 2015-2022 illustrating the regime variety the
paper claims to handle:
    1. XBI biotech sector ETF close (the panel-level price action)
    2. VIX (cross-sectional risk / fear gauge)
    3. 10-year Treasury yield (rate environment / discount rate channel)

Each panel shares the same x-axis. Major regime events are marked with
vertical lines, the three test windows are shaded, and a small legend
in the top panel explains the shading.

Output: drafts/paper_aaai/figures/intro_motivation.pdf
        drafts/paper_kdd/figures/intro_motivation.pdf
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd


XBI_PATH = Path("data/raw/xbi_close.csv")
VIX_PATH = Path("data/raw/volatility_indices.parquet")
FRED_PATH = Path("data/raw/macro_fred_full.csv")
OUT_DIRS = [
    Path("drafts/paper_aaai/figures"),
    Path("drafts/paper_kdd/figures"),
]

START = "2015-01-09"
END = "2022-12-31"

EVENTS = [
    # (date, label, row_index for label staggering 0/1/2)
    ("2020-03-23", "COVID trough", 0),
    ("2020-12-11", "mRNA FDA EUA", 1),
    ("2021-09-01", "SPAC peak", 0),
    ("2021-11-30", "Powell hawkish", 1),
    ("2022-03-16", "1st 25 bp hike", 2),
    ("2022-09-21", "75 bp hike", 0),
    ("2022-12-14", "Fed pivot tone", 1),
]
FOLDS = [
    ("2020-01-02", "2020-12-31", "Fold 1: COVID + mRNA"),
    ("2021-07-01", "2022-06-22", "Fold 2: SPAC -> rate-hike bear"),
    ("2022-07-01", "2022-12-22", "Fold 3: deep bear, late pivot"),
]
FOLD_COLORS = {0: "#fff1e0", 1: "#ffe0d0", 2: "#ffd0c4"}


def main() -> None:
    xbi = (
        pd.read_csv(XBI_PATH, parse_dates=["date"])
        .set_index("date")
        .sort_index()
        .loc[START:END]
    )
    vix_full = pd.read_parquet(VIX_PATH)
    vix = vix_full[["VIX"]].loc[START:END]
    fred = (
        pd.read_csv(FRED_PATH, parse_dates=["date"])
        .set_index("date")
        .sort_index()
        .loc[START:END]
    )
    dgs10 = fred[["DGS10"]].dropna()

    fig, axes = plt.subplots(
        3, 1, figsize=(8.5, 5.6), sharex=True,
        gridspec_kw={"height_ratios": [1.2, 1.0, 1.0], "hspace": 0.18},
    )

    ax_xbi, ax_vix, ax_rate = axes

    for ax in axes:
        for i, (ts, te, _) in enumerate(FOLDS):
            ax.axvspan(pd.Timestamp(ts), pd.Timestamp(te),
                       color=FOLD_COLORS[i], alpha=0.55, zorder=1)
        for date_str, _, _ in EVENTS:
            ax.axvline(pd.Timestamp(date_str), color="gray",
                       lw=0.45, alpha=0.55, zorder=1.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.25, linewidth=0.4)

    # Panel 1: XBI close
    ax_xbi.plot(xbi.index, xbi["close"], color="#1f3b6b",
                linewidth=1.0, label="XBI close", zorder=4)
    ax_xbi.set_ylabel("XBI close (USD)", fontsize=9)

    # Event labels (top panel only) on 3 staggered rows so 2022 events
    # do not collide.
    y_lo, y_hi = xbi["close"].min(), xbi["close"].max()
    ax_xbi.set_ylim(y_lo * 0.85, y_hi * 1.42)
    label_band_top = y_hi * 1.40
    label_band_bot = y_hi * 1.05
    band_h = label_band_top - label_band_bot
    row_y = {
        0: label_band_top,
        1: label_band_top - band_h * 0.42,
        2: label_band_top - band_h * 0.80,
    }
    for date_str, label, row in EVENTS:
        d = pd.Timestamp(date_str)
        v = float(xbi["close"].asof(d)) if d in xbi.index else float(xbi["close"].iloc[
            xbi.index.get_indexer([d], method="nearest")[0]
        ])
        ax_xbi.scatter([d], [v], s=12, color="black", zorder=6)
        ax_xbi.annotate(label, xy=(d, v), xytext=(d, row_y[row]),
                        fontsize=6.8, ha="center", va="top",
                        arrowprops=dict(arrowstyle="-", linewidth=0.35,
                                        color="gray", shrinkB=2),
                        zorder=5)

    # Panel 2: VIX
    ax_vix.plot(vix.index, vix["VIX"], color="#9c4221",
                linewidth=0.85, label="VIX", zorder=4)
    ax_vix.axhline(20, color="gray", lw=0.4, ls="--", alpha=0.5)
    ax_vix.set_ylabel("VIX (annualised %)", fontsize=9)
    ax_vix.set_ylim(8, 88)

    # Panel 3: 10Y Treasury yield
    ax_rate.plot(dgs10.index, dgs10["DGS10"], color="#2c6e49",
                 linewidth=0.85, label="10Y Treasury yield", zorder=4)
    ax_rate.set_ylabel("10Y yield (%)", fontsize=9)
    ax_rate.set_ylim(0.3, 4.7)

    # X axis formatting on bottom panel
    ax_rate.xaxis.set_major_locator(mdates.YearLocator())
    ax_rate.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_rate.tick_params(axis="x", which="major", length=4, labelsize=8)

    # Title and shared legend (only in the XBI panel)
    ax_xbi.set_title(
        "Regime variety in the 244-ticker biotech panel, 2015-2022 "
        "(test windows shaded)",
        fontsize=10,
    )

    # Custom legend for fold shading
    from matplotlib.patches import Patch
    fold_patches = [
        Patch(facecolor=FOLD_COLORS[i], alpha=0.55, label=lbl)
        for i, (_, _, lbl) in enumerate(FOLDS)
    ]
    ax_xbi.legend(handles=fold_patches, loc="lower left", fontsize=7,
                  framealpha=0.85, handlelength=1.5)

    fig.tight_layout(pad=0.4)

    for od in OUT_DIRS:
        od.mkdir(parents=True, exist_ok=True)
        fig.savefig(od / "intro_motivation.pdf", dpi=150, bbox_inches="tight")
    print(f"Saved intro_motivation.pdf to {len(OUT_DIRS)} dirs")


if __name__ == "__main__":
    main()
