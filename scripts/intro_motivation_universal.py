"""Figure 1 (intro motivation) for the universal-panel paper.

Replaces the legacy biotech rendering. Plots, over 2015-2025, the macro
regime context the 600-ticker S&P 500 Universal Ticker panel spans:
the CBOE VIX and the 10-year US Treasury yield, with the five
walk-forward test windows shaded. (A broad-index level panel was
dropped because the available SPY series ends in early 2023 and would
not cover folds F4-F5; VIX and the 10Y yield both span 2015-2025.)

Sources (already on Wulver):
  data/processed/risk_features_sp500_ext.parquet      -> vix
  data/processed/macro_duration_features_sp500_ext.parquet -> dgs10
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

START, END = "2015-01-09", "2025-12-31"

# Five walk-forward test windows (Section 3).
FOLDS = [
    ("F1", "2020-01-01", "2020-12-31", "COVID crash + recovery"),
    ("F2", "2021-07-01", "2022-06-30", "rate-hike rotation"),
    ("F3", "2022-07-01", "2023-06-30", "post-shock + banking"),
    ("F4", "2024-01-01", "2024-12-31", "AI mega-cap rally"),
    ("F5", "2025-07-01", "2025-12-31", "Fed-cut + post-election"),
]


def _dt_index(df):
    idx = pd.to_datetime(df.index)
    df = df.copy()
    df.index = idx
    return df.sort_index()


def main():
    risk = _dt_index(pd.read_parquet(
        "data/processed/risk_features_sp500_ext.parquet"))["vix"]
    macro = pd.read_parquet(
        "data/processed/macro_duration_features_sp500_ext.parquet")
    macro = macro.set_index(pd.to_datetime(macro["date"])).sort_index() \
        if "date" in macro.columns else _dt_index(macro)
    dgs10 = macro["dgs10"]

    sl = slice(pd.Timestamp(START), pd.Timestamp(END))
    vix, y10 = risk.loc[sl], dgs10.loc[sl]

    fig, axes = plt.subplots(2, 1, figsize=(9, 4.6), sharex=True)
    series = [
        (axes[0], vix, "VIX (annualised %)", "#9c4221"),
        (axes[1], y10, "10Y Treasury yield (%)", "#2e7d32"),
    ]
    for ax, s, lab, c in series:
        ax.plot(s.index, s.values, color=c, lw=1.1)
        ax.set_ylabel(lab, fontsize=8)
        ax.grid(True, alpha=0.22)
        ax.margins(x=0.01)
        ax.tick_params(labelsize=7)
        for _, a, b, _d in FOLDS:
            ax.axvspan(pd.Timestamp(a), pd.Timestamp(b),
                       color="#888888", alpha=0.16, zorder=0)
    # Fold labels along the top panel.
    y_top = axes[0].get_ylim()[1]
    for name, a, b, _d in FOLDS:
        mid = pd.Timestamp(a) + (pd.Timestamp(b) - pd.Timestamp(a)) / 2
        axes[0].text(mid, y_top, name, ha="center", va="bottom",
                     fontsize=7.5, fontweight="bold", color="#333333")
    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for lbl in axes[-1].get_xticklabels():
        lbl.set_fontsize(7)
    axes[0].set_ylim(top=y_top * 1.10)
    plt.tight_layout()
    for out_dir in [Path("drafts/universal_paper_aaai/figures"),
                    Path("drafts/universal_paper_kdd/figures")]:
        out_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_dir / "intro_motivation.pdf", bbox_inches="tight")
    plt.close()
    print("Saved intro_motivation.pdf (universal 2015-2025) to 2 dirs;",
          f"VIX {vix.index.min().date()}..{vix.index.max().date()} n={vix.notna().sum()};",
          f"y10 {y10.index.min().date()}..{y10.index.max().date()} n={y10.notna().sum()}")


if __name__ == "__main__":
    main()
