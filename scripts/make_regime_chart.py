"""Produce the regime-narrative chart for the paper.

Plots XBI (biotech sector ETF) close over 2015-2022 with the three
test windows shaded and major events annotated. Saves to
drafts/paper_aaai/figures/regime_xbi.pdf and the KDD copy.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd


XBI_PATH = Path("data/raw/xbi_close.csv")
OUT_DIRS = [
    Path("drafts/paper_aaai/figures"),
    Path("drafts/paper_kdd/figures"),
]
EVENTS = [
    # (date, short_label, row 0/1/2)
    ("2020-03-23", "COVID trough", 0),
    ("2020-12-11", "mRNA FDA EUA", 1),
    ("2021-09-01", "SPAC peak", 0),
    ("2022-01-26", "Powell hawkish", 1),
    ("2022-03-16", "1st 25 bp hike", 2),
    ("2022-09-21", "75 bp hike", 0),
    ("2022-10-13", "CPI shock", 1),
    ("2022-12-14", "Fed pivot tone", 2),
]
FOLDS = [
    ("2020-01-02", "2020-12-31", "Fold 1: COVID crash + biotech rocket"),
    ("2021-07-01", "2022-06-22", "Fold 2: SPAC bull -> rate-hike bear"),
    ("2022-07-01", "2022-12-22", "Fold 3: deep bear, late Fed pivot"),
]


def main() -> None:
    df = pd.read_csv(XBI_PATH, parse_dates=["date"]).set_index("date").sort_index()
    df = df.loc["2015-01-09":"2022-12-31"]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(df.index, df["close"], color="#1f3b6b", linewidth=1.0,
            label="XBI close", zorder=3)

    cmap = {0: "#fff1e0", 1: "#ffe0d0", 2: "#ffd0c4"}
    for i, (ts, te, lbl) in enumerate(FOLDS):
        ax.axvspan(pd.Timestamp(ts), pd.Timestamp(te),
                   color=cmap[i], alpha=0.6, zorder=1, label=lbl)

    # Events are anchored at the top of the panel and connected by
    # vertical leader lines to the price marker. Three label rows
    # ensure that close-together events in 2022 do not overlap.
    y_lo = df["close"].min() * 0.85
    y_hi = df["close"].max() * 1.45  # extra headroom for 3-row stack
    ax.set_ylim(y_lo, y_hi)
    label_band_top = y_hi * 0.99
    label_band_bot = df["close"].max() * 1.08
    band_h = label_band_top - label_band_bot
    row_y = {
        0: label_band_top,
        1: label_band_top - band_h * 0.40,
        2: label_band_top - band_h * 0.78,
    }
    for date_str, label, row in EVENTS:
        d = pd.Timestamp(date_str)
        if d not in df.index:
            d = df.index.asof(d)
        v = float(df["close"].loc[d])
        ax.scatter([d], [v], s=14, color="black", zorder=5)
        ax.annotate(label, xy=(d, v), xytext=(d, row_y[row]),
                    fontsize=7.5, ha="center", va="top",
                    arrowprops=dict(arrowstyle="-", linewidth=0.4,
                                    color="gray", shrinkB=2),
                    zorder=4)

    ax.set_xlim(df.index[0], df.index[-1])
    ax.set_ylabel("XBI close (USD)")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.tick_params(axis="x", which="major", length=4)
    ax.tick_params(axis="x", which="minor", length=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.4)

    leg = ax.legend(loc="lower left", fontsize=7.5, framealpha=0.85,
                    handlelength=2.0, ncol=1)
    fig.tight_layout(pad=0.4)

    for od in OUT_DIRS:
        od.mkdir(parents=True, exist_ok=True)
        fig.savefig(od / "regime_xbi.pdf", dpi=150, bbox_inches="tight")
    print(f"Saved regime_xbi.pdf to {len(OUT_DIRS)} dirs")


if __name__ == "__main__":
    main()
