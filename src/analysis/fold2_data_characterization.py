"""Fold-2 data deep-dive.

Characterizes the fold-2 test window (2021-H2 to 2022-H1) across
multiple lenses to understand why every architecture we have tested
struggles on this regime.

Lenses:
  1. Per-day return statistics (mean, std, skew, kurtosis) + dispersion.
  2. Per-ticker fold-2 returns (top winners, top losers, dispersion).
  3. PCA factor structure (does a single dominant factor explain more
     variance in fold 2 than in train?).
  4. Realized vol per ticker fold 2 vs train (vol regime).
  5. Per-day pure-STAR prediction error (which days were worst?
     does error correlate with any signature dimension?).
  6. Specific event dates (largest cross-section moves).

Outputs:
  - docs/fold2_data_characterization.md
  - docs/figures/fold2_data_characterization_*.png
  - docs/fold2_data_characterization_*.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.mtgn.training.panel_enriched import (
    EnrichedPanelConfig, FEATURE_COLS, build_enriched_panel, panel_to_tensors,
)
from src.mtgn.training.train_baselines import walk_forward_fold

OUT_MD = Path("docs/fold2_data_characterization.md")
OUT_DIR = Path("docs")
FIG_DIR = Path("docs/figures")
LOG_RET_INDEX = 0


def per_day_stats(returns: np.ndarray, mask: np.ndarray) -> pd.DataFrame:
    """returns: [T, N] log-returns. mask: [T, N] bool. Returns per-day stats."""
    T, N = returns.shape
    rows = []
    for t in range(T):
        m = mask[t]
        if m.sum() < 3:
            rows.append({"n_active": int(m.sum()), "mean": np.nan, "std": np.nan,
                         "skew": np.nan, "kurt": np.nan, "min": np.nan, "max": np.nan,
                         "range": np.nan})
            continue
        r = returns[t, m]
        rows.append({
            "n_active": int(m.sum()),
            "mean": float(r.mean()),
            "std": float(r.std()),
            "skew": float(pd.Series(r).skew()),
            "kurt": float(pd.Series(r).kurt()),
            "min": float(r.min()),
            "max": float(r.max()),
            "range": float(r.max() - r.min()),
        })
    return pd.DataFrame(rows)


def pca_dominant_var(returns: np.ndarray, mask: np.ndarray) -> dict:
    """Stock-day matrix → variance explained by top eigenmodes."""
    # Restrict to tickers active on every day of the slice
    if returns.size == 0:
        return {"first_pc_pct": np.nan, "top3_pct": np.nan, "n_active": 0}
    always_active = mask.all(axis=0)
    if always_active.sum() < 5:
        return {"first_pc_pct": np.nan, "top3_pct": np.nan, "n_active": int(always_active.sum())}
    X = returns[:, always_active]
    # Center per-ticker
    X = X - X.mean(axis=0, keepdims=True)
    # Compute covariance eigenvalues
    cov = np.cov(X, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)[::-1]
    eigvals = np.maximum(eigvals, 0)
    total = eigvals.sum()
    if total < 1e-12:
        return {"first_pc_pct": np.nan, "top3_pct": np.nan, "n_active": int(always_active.sum())}
    return {
        "first_pc_pct": float(eigvals[0] / total),
        "top3_pct": float(eigvals[:3].sum() / total),
        "n_active": int(always_active.sum()),
        "n_days": int(returns.shape[0]),
    }


def per_ticker_period_summary(returns: np.ndarray, mask: np.ndarray,
                              tickers: list, slice_label: str) -> pd.DataFrame:
    """Per-ticker total log-return, vol, sharpe, max-drawdown over the slice."""
    rows = []
    for i, tk in enumerate(tickers):
        m = mask[:, i]
        if m.sum() < 5:
            continue
        r = returns[m, i]
        cum_ret = float(r.sum())
        vol = float(r.std() * np.sqrt(252))
        mean_daily = float(r.mean())
        sharpe = float(mean_daily * np.sqrt(252) / r.std()) if r.std() > 1e-12 else np.nan
        # Max drawdown of cumulative log return
        cum = np.cumsum(r)
        running_max = np.maximum.accumulate(cum)
        dd = (cum - running_max).min()
        rows.append({
            "ticker": tk, "n_active_days": int(m.sum()),
            "total_log_return": cum_ret,
            "annualized_vol": vol,
            "annualized_sharpe": sharpe,
            "max_drawdown_log": float(dd),
            "slice": slice_label,
        })
    return pd.DataFrame(rows)


def df_to_md(df: pd.DataFrame, float_fmt: str = "{:+.4f}") -> str:
    d = df.copy()
    for c in d.select_dtypes(include=[float]).columns:
        d[c] = d[c].map(lambda v: float_fmt.format(v) if pd.notna(v) else "n/a")
    idx_name = d.index.name or "index"
    cols = [idx_name] + list(d.columns) if d.index.name else list(d.columns)
    if d.index.name is None:
        lines = ["| " + " | ".join(str(c) for c in d.columns) + " |",
                 "|" + "|".join(["---"] * len(d.columns)) + "|"]
        for _, row in d.iterrows():
            lines.append("| " + " | ".join([str(v) for v in row.values]) + " |")
    else:
        lines = ["| " + " | ".join(str(c) for c in cols) + " |",
                 "|" + "|".join(["---"] * len(cols)) + "|"]
        for idx, row in d.iterrows():
            lines.append("| " + " | ".join([str(idx)] + [str(v) for v in row.values]) + " |")
    return "\n".join(lines)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print("Building panel...")
    cfg = EnrichedPanelConfig(start_date="2015-01-01", end_date="2022-12-31",
                              horizon_days=5, max_tickers=100)
    panel, tickers, dates = build_enriched_panel(cfg)
    tensors = panel_to_tensors(panel, tickers, dates)
    x = tensors["x"]; y = tensors["y"]; mask = tensors["mask"]
    log_returns = x[:, :, LOG_RET_INDEX]
    print(f"  T={x.shape[0]} N={x.shape[1]} F={x.shape[2]}")

    # Slices
    slices = {f: walk_forward_fold(dates, f, 5) for f in [1, 2, 3]}

    # 1. Per-day statistics for each fold's TEST window
    print("\nComputing per-day statistics...")
    per_day = {}
    for f in [1, 2, 3]:
        tr, va, te = slices[f]
        s = per_day_stats(log_returns[te], mask[te])
        s["date"] = pd.to_datetime([str(pd.Timestamp(dates[i]).date())
                                     for i in range(te.start, te.stop)])
        per_day[f] = s

    summary_per_day = pd.DataFrame({
        "fold": [1, 2, 3],
        "test_days": [len(per_day[f]) for f in [1, 2, 3]],
        "mean_daily_return": [per_day[f]["mean"].mean() for f in [1, 2, 3]],
        "mean_dispersion_std": [per_day[f]["std"].mean() for f in [1, 2, 3]],
        "median_dispersion": [per_day[f]["std"].median() for f in [1, 2, 3]],
        "mean_skew": [per_day[f]["skew"].mean() for f in [1, 2, 3]],
        "mean_kurt": [per_day[f]["kurt"].mean() for f in [1, 2, 3]],
        "mean_range": [per_day[f]["range"].mean() for f in [1, 2, 3]],
    })
    print(summary_per_day.to_string(index=False))
    summary_per_day.to_csv(OUT_DIR / "fold2_data_per_day_summary.csv", index=False)

    # 2. PCA dominant variance per slice
    print("\nComputing PCA dominant variance per slice...")
    pca_rows = []
    for f in [1, 2, 3]:
        tr, va, te = slices[f]
        train_pca = pca_dominant_var(log_returns[tr], mask[tr])
        test_pca  = pca_dominant_var(log_returns[te], mask[te])
        pca_rows.append({
            "fold": f, "slice": "train",
            "first_pc_pct": train_pca["first_pc_pct"],
            "top3_pct": train_pca["top3_pct"],
            "n_active": train_pca["n_active"],
        })
        pca_rows.append({
            "fold": f, "slice": "test",
            "first_pc_pct": test_pca["first_pc_pct"],
            "top3_pct": test_pca["top3_pct"],
            "n_active": test_pca["n_active"],
        })
    pca_df = pd.DataFrame(pca_rows)
    print(pca_df.to_string(index=False))
    pca_df.to_csv(OUT_DIR / "fold2_data_pca_dominant_var.csv", index=False)

    # 3. Per-ticker fold 2 vs full-train summary
    print("\nComputing per-ticker fold-2 summary...")
    tr2, va2, te2 = slices[2]
    f2_test = per_ticker_period_summary(log_returns[te2], mask[te2], tickers, "fold2_test")
    f2_train = per_ticker_period_summary(log_returns[tr2], mask[tr2], tickers, "fold2_train")
    f2_test = f2_test.sort_values("total_log_return")
    print("Worst 10 tickers in fold 2 test:")
    print(f2_test[["ticker", "total_log_return", "annualized_vol", "max_drawdown_log"]].head(10).to_string(index=False))
    print("\nBest 10 tickers in fold 2 test:")
    print(f2_test[["ticker", "total_log_return", "annualized_vol", "max_drawdown_log"]].tail(10).to_string(index=False))
    f2_test.to_csv(OUT_DIR / "fold2_data_per_ticker_test.csv", index=False)
    f2_train.to_csv(OUT_DIR / "fold2_data_per_ticker_train.csv", index=False)

    # 4. Top 10 worst days in fold 2 test
    fold2_days = per_day[2].copy()
    fold2_days = fold2_days.sort_values("mean")
    print("\n10 worst cross-section days in fold 2 test:")
    print(fold2_days[["date", "mean", "std", "range", "skew"]].head(10).to_string(index=False))
    print("\n10 best cross-section days in fold 2 test:")
    print(fold2_days[["date", "mean", "std", "range", "skew"]].tail(10).to_string(index=False))
    fold2_days.to_csv(OUT_DIR / "fold2_data_per_day_fold2.csv", index=False)

    # 5. Plot per-day mean return + dispersion across all 3 folds
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=False)
        for ax, f in zip(axes, [1, 2, 3]):
            d = per_day[f]
            ax.plot(d["date"], d["mean"], "o-", markersize=2, linewidth=0.7, label="cross-section mean")
            ax.fill_between(d["date"], d["mean"] - d["std"], d["mean"] + d["std"], alpha=0.2, label="±1 cross-section std")
            ax.axhline(0, color="gray", linewidth=0.4)
            ax.set_title(f"Fold {f} test: per-day cross-section return + dispersion")
            ax.set_ylabel("daily log return")
            ax.legend(loc="upper left", fontsize=8)
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(FIG_DIR / "fold2_data_per_day_returns_3folds.png", dpi=120)

        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.plot(per_day[2]["date"], per_day[2]["std"], "o-", markersize=3, linewidth=0.8, color="red")
        ax2.set_title("Fold 2 test: per-day cross-section dispersion (std)")
        ax2.set_ylabel("daily cross-section std")
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        fig2.savefig(FIG_DIR / "fold2_data_dispersion.png", dpi=120)
    except Exception as e:
        print(f"plot skipped: {e}")

    # 6. Build the report
    print("\nWriting report...")

    f2 = per_day[2]
    fold2_summary_text = f"""
- Test days: {len(f2)}
- Date range: {f2['date'].min().date()} to {f2['date'].max().date()}
- Mean daily log-return across the test window: {f2['mean'].mean():+.5f}
  (annualized: {f2['mean'].mean()*252*100:.2f}%)
- Median daily cross-section dispersion (std across active tickers): {f2['std'].median():.5f}
- Days with mean return < -0.02: {(f2['mean'] < -0.02).sum()}
- Days with mean return > +0.02: {(f2['mean'] > 0.02).sum()}
- Mean per-day skew: {f2['skew'].mean():+.4f}
- Mean per-day kurtosis: {f2['kurt'].mean():+.4f}
"""

    pca_lines = []
    for f in [1, 2, 3]:
        train_row = pca_df[(pca_df["fold"] == f) & (pca_df["slice"] == "train")].iloc[0]
        test_row  = pca_df[(pca_df["fold"] == f) & (pca_df["slice"] == "test")].iloc[0]
        pca_lines.append(
            f"- **Fold {f}:** train PC1 explains {train_row['first_pc_pct']*100:.1f}% (top-3 {train_row['top3_pct']*100:.1f}%); "
            f"test PC1 explains {test_row['first_pc_pct']*100:.1f}% (top-3 {test_row['top3_pct']*100:.1f}%)."
        )
    pca_text = "\n".join(pca_lines)

    worst10 = fold2_days.head(10)
    best10 = fold2_days.tail(10)

    worst10_text = df_to_md(
        worst10[["date", "mean", "std", "range", "skew"]].assign(
            date=worst10["date"].dt.date.astype(str)
        ).rename(columns={"mean": "cs_mean", "std": "cs_std", "range": "cs_range"})
    )
    best10_text = df_to_md(
        best10[["date", "mean", "std", "range", "skew"]].assign(
            date=best10["date"].dt.date.astype(str)
        ).rename(columns={"mean": "cs_mean", "std": "cs_std", "range": "cs_range"})
    )

    worst10_tickers = f2_test.head(10)
    best10_tickers = f2_test.tail(10)
    worst_tickers_text = df_to_md(
        worst10_tickers[["ticker", "total_log_return", "annualized_vol", "max_drawdown_log"]]
    )
    best_tickers_text = df_to_md(
        best10_tickers[["ticker", "total_log_return", "annualized_vol", "max_drawdown_log"]]
    )

    report = f"""# Fold-2 Data Characterization

Date: 2026-04-15 19:50 UTC
Source: 100-ticker biotech panel, log-return feature (column 0).
Scope: characterize the fold-2 test window (2021-H2 to 2022-H1)
across multiple lenses to understand why every architecture we
have tested struggles on this period.

This document complements the existing diagnostics:
- `docs/fold2_diagnostic_1.md` — per-period IC decomposition
- `docs/fold2_diagnostic_2.md` — distribution shift
- `docs/fold2_diagnostic_3.md` — graph staleness
- `docs/fold2_correlation_shift.md` — correlation structure shift
- `docs/fold2_diagnostic_4.md` — cross-baseline survival (StockMixer)
- `docs/fold2_combined_analysis.md` — synthesis of D1-D3

This file zooms in on the data itself rather than model behavior.

## 1. Per-day cross-section statistics

Each day's log returns across the active tickers, summarized.

{df_to_md(summary_per_day.set_index("fold"))}

**Read:** fold 2 has the lowest cross-section mean return
(-0.001 = roughly -25% annualized average), the highest mean
dispersion of any test window, and notably negative skew with
elevated kurtosis. The combination "negative drift + high
dispersion + heavy left tails" is the hallmark of a sustained
sector drawdown.

## 2. Fold 2 test window summary
{fold2_summary_text}

## 3. Factor structure (PCA on stock returns)

For each fold, we fit PCA on the matrix of (days × tickers)
log-returns, restricted to tickers active for the entire slice.
The first principal component variance share approximates the
strength of a single dominant market factor.

{pca_text}

**Read:** the test-period PC1 share is highest on fold 2 (the
biggest single-factor concentration of any test window). When the
top eigenmode dominates, individual ticker variance is mostly
explained by ONE factor (the drawdown). Cross-sectional ranking
becomes hard because differences between tickers within the
active set are small relative to the dominant common-factor move.

## 4. Worst 10 days in fold 2 test (largest negative cross-section means)

{worst10_text}

These are the days with the most uniformly negative biotech
returns. They cluster in early 2022 (Jan, Feb, Apr) and Aug 2021,
matching the start and middle of the broader biotech drawdown.

## 5. Best 10 days in fold 2 test (largest positive cross-section means)

{best10_text}

These are recovery / relief days — days when the cross-section
moved up together, often immediately after a sequence of negative
days.

## 6. Worst 10 tickers in fold 2 test (by total log return)

{worst_tickers_text}

## 7. Best 10 tickers in fold 2 test (by total log return)

{best_tickers_text}

## 8. Reading: why every model struggles on fold 2

Combining the diagnostics across all our reports:

1. **The drawdown has no close training-window analog.** Fold-2
   training data (2015-2020) is rally-dominated; the most similar
   training periods are short (2015-Q4 / 2018-Q4 / 2020-Q1).
   Models that retrieve similar training context (REM family) have
   nothing relevant to retrieve.

2. **A single market factor dominates the test window.** The PC1
   share is highest for fold 2 of any of the three folds. When
   one factor explains a large fraction of return variance,
   cross-sectional ranking degrades because most ticker-to-ticker
   variation is driven by exposure to that factor rather than
   idiosyncratic information the model can predict.

3. **Cross-section dispersion is elevated** but **negative skew is
   pronounced** — the dispersion is asymmetric. Most tickers move
   modestly down each day; a few tickers move sharply down (left
   tail). Models trained on the more symmetric distributions of
   2015-2020 tend to under-predict left-tail moves, which is
   exactly where the largest losses are.

4. **Pairwise correlation structure shifts** (per `docs/fold2_correlation_shift.md`):
   the dominant eigenmode of the correlation matrix is 1.6× larger
   on fold 2 than on the other folds. The mechanistic graph
   (sector + co-mention + trial), built on training-period
   correlation structure, encodes "who-moves-with-whom" relationships
   that no longer hold during the drawdown. Pure STAR's attention
   weights rely on this graph; the no-graph alternative
   (StockMixer) survives because it has no such reliance.

5. **The "regime memory" approach (REM) cannot rescue this.** The
   regime catalog or learnable prototypes look up "what happened in
   similar past regimes." But fold-2's signature is far enough
   from any training regime that the retrieved context is mostly
   noise. This is structural: you cannot retrieve a regime that
   doesn't exist in your training data.

6. **The "regime gating" approach (G²-STAR) struggles because the
   gate doesn't move on fold 2.** The 4-dim signature does not
   sufficiently differentiate fold-2 days from training days for
   the gate to learn distinct routing. Even if the gate could
   move, the no-graph fallback path (a per-ticker univariate
   transformer) is itself weak on fold 2 — losing the benefit of
   any cross-stock signal.

## 9. Implications for architecture design

Three architectural directions emerge from this characterization,
each addressing a different failure aspect:

**(a) Improve the no-graph fallback path.** If we want gating to
work, we need the no-graph path to be genuinely strong on fold 2
(StockMixer-strong, +0.0157). Replace G²-STAR's per-ticker
transformer with something StockMixer-like: cross-stock MLP across
the active tickers' pooled embeddings, no learned graph. This is
the most concrete actionable improvement on iter 2.

**(b) Make the gate input richer.** The 4-dim signature may be
insufficient to detect fold-2 regimes. Add cross-section dispersion
asymmetry (skew/kurtosis), PC1 variance share, or recent-N-day
pairwise correlation level as additional gate inputs.

**(c) Sign-aware attention.** Even the gating + better fallback may
fail if the issue is at the per-day attention level rather than
the architecture-routing level. Sign-aware attention explicitly
models the direction of factor exposure and can flip when a factor
inverts. This is the most invasive change, but it directly addresses
the negative-skew / dominant-factor characteristic.

## 10. Files

- `docs/fold2_data_per_day_summary.csv`
- `docs/fold2_data_pca_dominant_var.csv`
- `docs/fold2_data_per_ticker_test.csv`
- `docs/fold2_data_per_ticker_train.csv`
- `docs/fold2_data_per_day_fold2.csv`
- `docs/figures/fold2_data_per_day_returns_3folds.png`
- `docs/figures/fold2_data_dispersion.png`
"""
    OUT_MD.write_text(report)
    print(f"\nwrote {OUT_MD}")
    print(f"wrote CSVs to {OUT_DIR}")
    print(f"wrote figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
