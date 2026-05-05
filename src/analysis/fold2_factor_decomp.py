"""Diagnostic 2: Factor decomposition on fold 2.

Regress each day's cross-sectional returns on ticker factor exposures.
Quantify how much variance each factor explains.

Factors (from the 22-dim panel):
  - SIZE    = log_market_cap
  - QUALITY = cash_runway_q (cash-flow proxy)
  - GROWTH  = rd_intensity
  - REV_GRW = revenue_growth_yoy
  - MOM20   = log_return_20d (20-day momentum)
  - VOL60   = realized_vol_60d
  - VOL20   = realized_vol_20d

Method: at day t, use each ticker's latest available factor value
(from fold-2 test day t itself, so causal for factor exposure but not
causal for predicting day-t return — just characterizing factor
behavior). Regress next-5d forward returns on factor exposures.

Output: docs/fold2_factor_decomp.md + CSV of per-fold R² and factor
coefficients.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.mtgn.training.panel_enriched import (
    EnrichedPanelConfig, FEATURE_COLS, build_enriched_panel, panel_to_tensors,
)
from src.mtgn.training.train_baselines import walk_forward_fold


OUT_MD = Path("docs/fold2_factor_decomp.md")

# Factor indices in FEATURE_COLS
FACTOR_SPEC = {
    "SIZE":    "log_market_cap",
    "QUALITY": "cash_runway_q",
    "GROWTH":  "rd_intensity",
    "REV_GRW": "revenue_growth_yoy",
    "MOM20":   "log_return_20d",
    "VOL60":   "realized_vol_60d",
    "VOL20":   "realized_vol_20d",
    "CASH":    "cash_to_mc",
}


def cross_sectional_factor_regression(x: np.ndarray, y: np.ndarray, mask: np.ndarray,
                                      factor_indices: dict, date_slice: slice) -> pd.DataFrame:
    """For each day in date_slice, regress y[t, mask] on factor features.
    Returns per-day results: R², univariate coefficients per factor, etc."""
    rows = []
    for t in range(date_slice.start, date_slice.stop):
        m = mask[t]
        if m.sum() < 10:
            continue
        y_t = y[t, m]
        row = {"day": t, "n_active": int(m.sum()), "y_std": float(y_t.std())}
        # Multivariate regression y ~ factors + intercept
        facs = np.column_stack([x[t, m, idx] for idx in factor_indices.values()])  # [N_active, K]
        # Handle NaN / inf
        facs = np.where(np.isfinite(facs), facs, 0.0)
        # Add intercept
        A = np.column_stack([np.ones(facs.shape[0]), facs])
        try:
            coef, residuals, rank, sv = np.linalg.lstsq(A, y_t, rcond=None)
            y_hat = A @ coef
            ss_res = float(((y_t - y_hat) ** 2).sum())
            ss_tot = float(((y_t - y_t.mean()) ** 2).sum())
            r2 = 1.0 - ss_res / (ss_tot + 1e-12)
            row["r2_multi"] = r2
            for k, name in enumerate(factor_indices.keys()):
                row[f"coef_{name}"] = float(coef[k + 1])
        except Exception:
            row["r2_multi"] = np.nan

        # Univariate R² per factor (to compare which factor alone is most predictive)
        for name, idx in factor_indices.items():
            xi = facs[:, list(factor_indices.keys()).index(name)]
            if xi.std() < 1e-12 or y_t.std() < 1e-12:
                row[f"r2_uni_{name}"] = np.nan
                continue
            corr = float(np.corrcoef(xi, y_t)[0, 1])
            row[f"r2_uni_{name}"] = corr * corr  # R² = correlation²
            row[f"corr_{name}"] = corr

        rows.append(row)
    return pd.DataFrame(rows)


def main():
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)

    print("Building panel...")
    cfg = EnrichedPanelConfig(start_date="2015-01-01", end_date="2022-12-31",
                              horizon_days=5, max_tickers=100)
    panel, tickers, dates = build_enriched_panel(cfg)
    tensors = panel_to_tensors(panel, tickers, dates)
    x = tensors["x"]; y = tensors["y"]; mask = tensors["mask"]

    factor_indices = {
        name: FEATURE_COLS.index(col) for name, col in FACTOR_SPEC.items()
    }

    results = {}
    for fold in [1, 2, 3]:
        tr, va, te = walk_forward_fold(dates, fold, 5)
        print(f"\n=== Fold {fold} test window, daily factor regression ===")
        df = cross_sectional_factor_regression(x, y, mask, factor_indices, te)
        results[fold] = df
        print(f"  Mean R² (multivariate, 8 factors): {df['r2_multi'].mean():.4f}")
        print(f"  Median R²: {df['r2_multi'].median():.4f}")
        print(f"  Max R²:    {df['r2_multi'].max():.4f}")
        print("  Mean univariate R² per factor:")
        for name in FACTOR_SPEC.keys():
            col = f"r2_uni_{name}"
            if col in df.columns:
                print(f"    {name:8s}  R² {df[col].mean():.4f}  corr {df.get(f'corr_{name}', pd.Series([np.nan])).mean():+.4f}")

    # Aggregate + export
    summary = []
    for fold in [1, 2, 3]:
        df = results[fold]
        row = {"fold": fold, "n_days": len(df), "mean_R2_multi": df["r2_multi"].mean()}
        for name in FACTOR_SPEC.keys():
            row[f"R2_uni_{name}"] = df.get(f"r2_uni_{name}", pd.Series([np.nan])).mean()
            row[f"corr_{name}"] = df.get(f"corr_{name}", pd.Series([np.nan])).mean()
        summary.append(row)
    summary_df = pd.DataFrame(summary).set_index("fold")
    summary_df.to_csv(OUT_MD.parent / "fold2_factor_decomp_summary.csv")

    def md_table(df: pd.DataFrame) -> str:
        d = df.copy()
        for c in d.select_dtypes(include=[float]).columns:
            d[c] = d[c].map(lambda v: f"{v:+.4f}" if pd.notna(v) else "n/a")
        idx_name = d.index.name or "index"
        lines = ["| " + " | ".join([idx_name] + list(d.columns)) + " |",
                 "|" + "|".join(["---"] * (len(d.columns) + 1)) + "|"]
        for idx, row in d.iterrows():
            lines.append("| " + " | ".join([str(idx)] + [str(v) for v in row.values]) + " |")
        return "\n".join(lines)

    r2_cols = [c for c in summary_df.columns if c.startswith("R2_uni_") or c == "mean_R2_multi"]
    corr_cols = [c for c in summary_df.columns if c.startswith("corr_")]
    r2_df = summary_df[r2_cols]
    corr_df = summary_df[corr_cols]

    report = f"""# Fold-2 Factor Decomposition (Diagnostic 2, Phase 1)

Date: 2026-04-16
Method: per-day cross-sectional regression of 5d forward returns on 8
factor features. Computes R² per factor per day, then averages over
each fold's test window.

Factors:
- SIZE    = log market cap
- QUALITY = cash runway (quarters of runway based on cash + cash flow)
- GROWTH  = R&D intensity
- REV_GRW = revenue growth year-over-year
- MOM20   = 20-day log return (momentum)
- VOL20   = 20-day realized volatility
- VOL60   = 60-day realized volatility
- CASH    = cash / market cap

## 1. Mean R² per factor per fold

### Variance explained (R² values, averaged daily)

{md_table(r2_df)}

### Factor correlations with 5d forward return (average per fold)

{md_table(corr_df)}

## 2. Reading the table

A **flipping correlation sign** across folds reveals factor inversion.
Specifically, compare:
- Fold 1 (2020 COVID rally): strong growth/momentum, weak quality
- Fold 2 (2021-22 drawdown): expected quality-flip (positive quality correlation, negative momentum)
- Fold 3 (2022-H2 recovery): expected reversion closer to fold 1 pattern

If QUALITY correlation flips from negative (fold 1) to positive (fold 2) and back to negative (fold 3), this quantifies the quality-factor-flip hypothesis.

## 3. Multivariate R²

| Fold | 8-factor R² |
|---|---|
| 1 | {summary_df.loc[1, 'mean_R2_multi']:+.4f} |
| 2 | {summary_df.loc[2, 'mean_R2_multi']:+.4f} |
| 3 | {summary_df.loc[3, 'mean_R2_multi']:+.4f} |

Higher R² means a simple linear factor model EXPLAINS MORE variance
on that fold — so the universe behavior is more factor-driven and
less idiosyncratic. Fold 2 with high R² means a linear factor model
would predict well and a complex deep model should be BETTER than
this linear benchmark, not worse.

## 4. Files

- `docs/fold2_factor_decomp_summary.csv`
"""
    OUT_MD.write_text(report)
    print(f"\nwrote {OUT_MD}")


if __name__ == "__main__":
    main()
