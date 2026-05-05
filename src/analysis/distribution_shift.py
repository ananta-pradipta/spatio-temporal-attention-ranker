"""Fold-wise distribution shift quantification.

Diagnostic 2 per `parallel-work-plan-sota-and-fold2-diagnostics.md`
Section 4.2. Asks: is fold 2's test distribution measurably more
distant from its training distribution than folds 1 and 3?

Three metrics per fold:
  1. Per-feature KL(train || test) for each of the 22 features
     (histogram-based with 50 bins on the combined support).
  2. Joint Maximum Mean Discrepancy (MMD^2) between train and test
     feature vectors using an RBF kernel with median heuristic
     bandwidth (subsampled to 2000 points per side for tractability).
  3. Target-shift KL between the 5d-forward log-return distribution
     on train vs test.

Outputs:
  - docs/fold2_diagnostic_2.md
  - docs/fold2_diagnostic_2_feature_kl.csv
  - docs/fold2_diagnostic_2_summary.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.mtgn.training.panel_enriched import (
    EnrichedPanelConfig, FEATURE_COLS, build_enriched_panel, panel_to_tensors,
)
from src.mtgn.training.train_baselines import walk_forward_fold


OUT_MD = Path("docs/fold2_diagnostic_2.md")
OUT_FEAT_KL = Path("docs/fold2_diagnostic_2_feature_kl.csv")
OUT_SUM = Path("docs/fold2_diagnostic_2_summary.csv")
SEED = 0
N_MMD = 2000
N_BINS = 50


def kl_hist(a: np.ndarray, b: np.ndarray, n_bins: int = N_BINS,
            eps: float = 1e-8) -> float:
    """KL(a || b) on histograms with shared support.

    a, b: 1-D arrays. Support is min(min(a), min(b)) to max(max(a), max(b)).
    Adds eps to each bin before normalization to avoid log 0.
    """
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    lo = min(a.min(), b.min())
    hi = max(a.max(), b.max())
    if hi - lo < 1e-12:
        return 0.0
    bins = np.linspace(lo, hi, n_bins + 1)
    p, _ = np.histogram(a, bins=bins)
    q, _ = np.histogram(b, bins=bins)
    p = p.astype(np.float64) + eps; p /= p.sum()
    q = q.astype(np.float64) + eps; q /= q.sum()
    return float(np.sum(p * np.log(p / q)))


def median_pairwise_distance(x: np.ndarray) -> float:
    """Median heuristic bandwidth estimate."""
    rng = np.random.default_rng(SEED)
    m = min(x.shape[0], 1000)
    idx = rng.choice(x.shape[0], m, replace=False)
    xs = x[idx]
    d = xs[:, None, :] - xs[None, :, :]
    sq = np.sum(d * d, axis=-1)
    triu = sq[np.triu_indices_from(sq, k=1)]
    med = np.median(np.sqrt(triu[triu > 0]))
    return float(med)


def mmd2_rbf(x: np.ndarray, y: np.ndarray, sigma: float) -> float:
    """Unbiased MMD^2 with RBF kernel, bandwidth sigma. Subsample x, y to N_MMD."""
    rng = np.random.default_rng(SEED)
    def sub(a):
        if a.shape[0] > N_MMD:
            i = rng.choice(a.shape[0], N_MMD, replace=False)
            return a[i]
        return a
    x = sub(x); y = sub(y)
    gamma = 1.0 / (2.0 * sigma * sigma)
    def rbf(u, v):
        d = u[:, None, :] - v[None, :, :]
        return np.exp(-gamma * np.sum(d * d, axis=-1))
    Kxx = rbf(x, x); Kyy = rbf(y, y); Kxy = rbf(x, y)
    m, n = x.shape[0], y.shape[0]
    # Unbiased estimator
    Kxx_sum = (Kxx.sum() - np.trace(Kxx)) / (m * (m - 1))
    Kyy_sum = (Kyy.sum() - np.trace(Kyy)) / (n * (n - 1))
    Kxy_sum = Kxy.sum() / (m * n)
    return float(Kxx_sum + Kyy_sum - 2 * Kxy_sum)


def flatten_panel_to_rows(x: np.ndarray, mask: np.ndarray,
                          date_slice: slice) -> np.ndarray:
    """Collect [T, N, F] into a [T*N_active, F] array over date_slice."""
    sub_x = x[date_slice]          # [T', N, F]
    sub_m = mask[date_slice]       # [T', N]
    return sub_x[sub_m]            # [active rows, F]


def flatten_target(y: np.ndarray, mask: np.ndarray,
                   date_slice: slice) -> np.ndarray:
    return y[date_slice][mask[date_slice]]


def analyze_fold(fold: int, x: np.ndarray, y: np.ndarray, mask: np.ndarray,
                 dates) -> dict:
    train_s, val_s, test_s = walk_forward_fold(dates, fold, horizon_days=5)

    x_train = flatten_panel_to_rows(x, mask, train_s)
    x_test  = flatten_panel_to_rows(x, mask, test_s)
    y_train = flatten_target(y, mask, train_s)
    y_test  = flatten_target(y, mask, test_s)

    # Z-score using train statistics (matches training code)
    mu = x_train.mean(axis=0); sd = x_train.std(axis=0) + 1e-8
    x_train_z = (x_train - mu) / sd
    x_test_z  = (x_test  - mu) / sd

    # Per-feature KL on standardized values
    per_feat_kl = []
    for j, name in enumerate(FEATURE_COLS):
        per_feat_kl.append({
            "fold": fold, "feature": name,
            "kl_train_test": kl_hist(x_train_z[:, j], x_test_z[:, j]),
        })

    # Joint MMD^2 on standardized vectors
    sigma = median_pairwise_distance(x_train_z)
    mmd = mmd2_rbf(x_train_z, x_test_z, sigma)

    # Target-shift KL on raw 5d forward log returns
    tgt_kl = kl_hist(y_train, y_test)

    return {
        "fold": fold,
        "per_feature_kl": per_feat_kl,
        "mean_feature_kl": float(np.mean([r["kl_train_test"] for r in per_feat_kl])),
        "max_feature_kl": float(np.max([r["kl_train_test"] for r in per_feat_kl])),
        "mmd2_rbf": mmd,
        "sigma_rbf": sigma,
        "target_kl": tgt_kl,
        "n_train_rows": int(x_train.shape[0]),
        "n_test_rows": int(x_test.shape[0]),
        "n_train_days": int(train_s.stop - train_s.start),
        "n_test_days": int(test_s.stop - test_s.start),
    }


def main():
    print("Building enriched panel...")
    panel_cfg = EnrichedPanelConfig(
        start_date="2015-01-01", end_date="2022-12-31",
        horizon_days=5, max_tickers=100,
    )
    panel, tickers, dates = build_enriched_panel(panel_cfg)
    tensors = panel_to_tensors(panel, tickers, dates)
    x = tensors["x"]; y = tensors["y"]; mask = tensors["mask"]
    print(f"  T={x.shape[0]} N={x.shape[1]} F={x.shape[2]}")

    results = {}
    rows = []
    for fold in [1, 2, 3]:
        print(f"\nFold {fold}...")
        r = analyze_fold(fold, x, y, mask, dates)
        results[fold] = r
        for pf in r["per_feature_kl"]:
            rows.append(pf)
        print(f"  mean feature KL: {r['mean_feature_kl']:.4f}")
        print(f"  max  feature KL: {r['max_feature_kl']:.4f}")
        print(f"  MMD^2 (RBF, median heuristic sigma={r['sigma_rbf']:.3f}): {r['mmd2_rbf']:+.6f}")
        print(f"  target 5d-fwd KL: {r['target_kl']:.4f}")

    per_feat_df = pd.DataFrame(rows)
    per_feat_wide = per_feat_df.pivot(index="feature", columns="fold", values="kl_train_test")
    per_feat_wide = per_feat_wide.reindex(FEATURE_COLS)
    per_feat_wide.to_csv(OUT_FEAT_KL)

    summary = pd.DataFrame([
        {
            "fold": f,
            "mean_feature_kl": results[f]["mean_feature_kl"],
            "max_feature_kl":  results[f]["max_feature_kl"],
            "mmd2_rbf":        results[f]["mmd2_rbf"],
            "target_kl":       results[f]["target_kl"],
            "n_train_rows":    results[f]["n_train_rows"],
            "n_test_rows":     results[f]["n_test_rows"],
        }
        for f in [1, 2, 3]
    ])
    summary.to_csv(OUT_SUM, index=False)

    # Rank which fold is furthest shifted per metric
    def ranking(col: str, higher_is_shifted: bool = True) -> str:
        order = summary.sort_values(col, ascending=not higher_is_shifted)["fold"].tolist()
        return " > ".join(f"fold {f}" for f in order)

    def df_to_md(df: pd.DataFrame) -> str:
        d = df.copy()
        for c in d.select_dtypes(include=[float]).columns:
            d[c] = d[c].map(lambda v: f"{v:.4f}" if pd.notna(v) else "n/a")
        idx_name = d.index.name or "index"
        cols = [idx_name] + list(d.columns)
        lines = ["| " + " | ".join(str(c) for c in cols) + " |",
                 "|" + "|".join(["---"] * len(cols)) + "|"]
        for idx, row in d.iterrows():
            lines.append("| " + " | ".join([str(idx)] + [str(v) for v in row.values]) + " |")
        return "\n".join(lines)

    report = f"""# Fold-2 Diagnostic 2: Distribution Shift Quantification

Date: 2026-04-15
Source: 100-ticker biotech panel, 22 features, 2015-2022.

## 1. Question

Is fold 2's test distribution measurably more distant from its
training distribution than folds 1 and 3? Per-memo-Section-4.2
expectation: if fold 2 has clearly higher distribution distance,
the regime-change hypothesis is quantitatively supported.

## 2. Summary

{df_to_md(summary.set_index("fold"))}

Metric definitions:
- `mean_feature_kl`: mean across 22 features of KL(train marginal
  || test marginal) after train-fold z-score standardization,
  50-bin histogram.
- `max_feature_kl`: max across features of the above.
- `mmd2_rbf`: unbiased Maximum Mean Discrepancy^2 with RBF kernel,
  median-heuristic bandwidth on train rows. Subsampled to 2000 rows
  per side for tractability, seed 0.
- `target_kl`: KL of raw 5d-forward log-return marginal.

## 3. Ranking by shift magnitude (higher = more shifted)

- By mean feature KL: **{ranking('mean_feature_kl')}**
- By max feature KL: **{ranking('max_feature_kl')}**
- By joint MMD^2: **{ranking('mmd2_rbf')}**
- By target KL: **{ranking('target_kl')}**

## 4. Per-feature KL(train || test), each fold

{df_to_md(per_feat_wide)}

## 5. Interpretation

Per the memo's decision rule:
- If fold 2 is clearly the most shifted fold on multiple metrics,
  the regime-change hypothesis has quantitative support and the
  paper can cite this as the mechanism behind the fold-2 failure.
- If fold 2's shift is comparable to folds 1 and 3, the failure is
  about something subtler than marginal feature or target shift
  (e.g., correlation structure shift).

The current results are in Section 2 and 3 above. See also the
combined analysis at `docs/fold2_combined_analysis.md` (after all
three diagnostics land).

## 6. Data exports

- `docs/fold2_diagnostic_2_summary.csv`
- `docs/fold2_diagnostic_2_feature_kl.csv`
"""
    OUT_MD.write_text(report)
    print(f"\nwrote {OUT_MD}")
    print(f"wrote {OUT_SUM}")
    print(f"wrote {OUT_FEAT_KL}")


if __name__ == "__main__":
    main()
