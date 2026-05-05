"""Per-fold ticker-return correlation structure shift.

Tests the leading remaining fold-2 failure hypothesis from
`docs/fold2_combined_analysis.md` Section 2.1: correlation structure
between tickers changes between train and test even when marginal
distributions do not.

For each fold:
  - Build an N x N pairwise log-return correlation matrix C_train
    from the train slice (using only active-ticker days).
  - Build C_test from the test slice.
  - Report Frobenius norm of (C_train - C_test), spectral norm, and
    element-wise mean absolute difference (with diagonal excluded).
  - Report fraction of ticker pairs whose correlation signs flip.

If fold 2 shows substantially larger correlation shift than folds 1
and 3, the correlation-structure-shift hypothesis is supported.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.mtgn.training.panel_enriched import (
    EnrichedPanelConfig, build_enriched_panel, panel_to_tensors,
)
from src.mtgn.training.train_baselines import walk_forward_fold


OUT_MD = Path("docs/fold2_correlation_shift.md")
OUT_SUM = Path("docs/fold2_correlation_shift_summary.csv")
OUT_FULL = Path("docs/fold2_correlation_shift_per_fold.csv")
LOG_RET_FEATURE_INDEX = 0  # `log_return` is the first feature in FEATURE_COLS


def ticker_correlation_matrix(x_log_ret: np.ndarray, mask: np.ndarray,
                              date_slice: slice, min_overlap: int = 30) -> np.ndarray:
    """Compute pairwise Pearson correlation between tickers using their
    aligned log-return time series over the given date slice.

    x_log_ret: [T, N] log-return feature values.
    mask:      [T, N] bool active mask.
    Returns:   [N, N] symmetric correlation matrix, NaN where overlap is
               below `min_overlap` days.
    """
    sub = x_log_ret[date_slice]    # [T', N]
    sub_m = mask[date_slice]       # [T', N]
    T_sub, N = sub.shape
    C = np.full((N, N), np.nan, dtype=np.float64)
    for i in range(N):
        xi = sub[:, i]; mi = sub_m[:, i]
        for j in range(i, N):
            xj = sub[:, j]; mj = sub_m[:, j]
            joint = mi & mj
            if joint.sum() < min_overlap:
                continue
            a = xi[joint]; b = xj[joint]
            if a.std() < 1e-12 or b.std() < 1e-12:
                continue
            c = float(np.corrcoef(a, b)[0, 1])
            C[i, j] = c; C[j, i] = c
    np.fill_diagonal(C, 1.0)
    return C


def shift_metrics(C_train: np.ndarray, C_test: np.ndarray) -> dict:
    """Quantify the change between two correlation matrices."""
    N = C_train.shape[0]
    valid = np.isfinite(C_train) & np.isfinite(C_test)
    # Exclude diagonal
    diag_mask = np.eye(N, dtype=bool)
    off_diag = valid & ~diag_mask
    if off_diag.sum() == 0:
        return {"n_valid_pairs": 0}
    diff = C_train - C_test
    diff_off = diff[off_diag]
    frob = float(np.sqrt((diff_off ** 2).sum()))
    mean_abs = float(np.abs(diff_off).mean())
    # Element-wise RMSE normalized by pair count for comparability
    rmse = float(np.sqrt((diff_off ** 2).mean()))

    # Spectral norm requires full matrix; impute NaN with 0 to compute
    C_tr = np.where(np.isfinite(C_train), C_train, 0.0)
    C_te = np.where(np.isfinite(C_test), C_test, 0.0)
    spec = float(np.linalg.norm(C_tr - C_te, ord=2))

    # Sign flips on off-diagonal valid entries
    signs_equal = np.sign(C_train[off_diag]) == np.sign(C_test[off_diag])
    flip_frac = float(1.0 - signs_equal.mean())

    # Mean correlation levels (not shifts, for context)
    mean_train = float(np.nanmean(C_train[off_diag]))
    mean_test  = float(np.nanmean(C_test[off_diag]))

    return {
        "frobenius_norm_diff": frob,
        "mean_abs_diff": mean_abs,
        "rmse_diff": rmse,
        "spectral_norm_diff": spec,
        "sign_flip_fraction": flip_frac,
        "mean_train_corr": mean_train,
        "mean_test_corr": mean_test,
        "n_valid_pairs": int(off_diag.sum()),
    }


def main():
    print("Building enriched panel...")
    cfg = EnrichedPanelConfig(
        start_date="2015-01-01", end_date="2022-12-31",
        horizon_days=5, max_tickers=100,
    )
    panel, tickers, dates = build_enriched_panel(cfg)
    tensors = panel_to_tensors(panel, tickers, dates)
    x = tensors["x"]; mask = tensors["mask"]
    print(f"  T={x.shape[0]} N={x.shape[1]} F={x.shape[2]}")

    # Extract log_return time series
    x_log_ret = x[:, :, LOG_RET_FEATURE_INDEX]   # [T, N]

    rows = []
    for fold in [1, 2, 3]:
        tr, va, te = walk_forward_fold(dates, fold, 5)
        print(f"\nFold {fold}: train [{tr.start}:{tr.stop}] test [{te.start}:{te.stop}]")
        C_train = ticker_correlation_matrix(x_log_ret, mask, tr)
        C_test  = ticker_correlation_matrix(x_log_ret, mask, te)
        m = shift_metrics(C_train, C_test)
        m["fold"] = fold
        m["n_train_days"] = int(tr.stop - tr.start)
        m["n_test_days"]  = int(te.stop - te.start)
        rows.append(m)
        print(f"  Frobenius of C_train-C_test: {m['frobenius_norm_diff']:.4f}")
        print(f"  mean |Δ|   : {m['mean_abs_diff']:.4f}")
        print(f"  RMSE(Δ)    : {m['rmse_diff']:.4f}")
        print(f"  spectral   : {m['spectral_norm_diff']:.4f}")
        print(f"  sign flips : {m['sign_flip_fraction']:.4f}")
        print(f"  mean train corr: {m['mean_train_corr']:+.4f}  mean test corr: {m['mean_test_corr']:+.4f}")

    df = pd.DataFrame(rows).set_index("fold")
    df.to_csv(OUT_FULL)

    summary = df[["frobenius_norm_diff", "mean_abs_diff", "rmse_diff",
                  "spectral_norm_diff", "sign_flip_fraction",
                  "mean_train_corr", "mean_test_corr"]]
    summary.to_csv(OUT_SUM)

    def df_to_md(df_: pd.DataFrame, float_fmt: str = "{:+.4f}") -> str:
        d = df_.copy()
        for c in d.select_dtypes(include=[float]).columns:
            d[c] = d[c].map(lambda v: float_fmt.format(v) if pd.notna(v) else "n/a")
        idx_name = d.index.name or "index"
        cols = [idx_name] + list(d.columns)
        lines = ["| " + " | ".join(str(c) for c in cols) + " |",
                 "|" + "|".join(["---"] * len(cols)) + "|"]
        for idx, row in d.iterrows():
            lines.append("| " + " | ".join([str(idx)] + [str(v) for v in row.values]) + " |")
        return "\n".join(lines)

    def rank_by(col: str) -> str:
        order = df.sort_values(col, ascending=False).index.tolist()
        return " > ".join(f"fold {f}" for f in order)

    frob_rank = rank_by("frobenius_norm_diff")
    rmse_rank = rank_by("rmse_diff")
    spec_rank = rank_by("spectral_norm_diff")
    flip_rank = rank_by("sign_flip_fraction")
    shift_lvl_rank = rank_by("mean_abs_diff")

    verdict = (
        "The fold-2-correlation-shift hypothesis is supported."
        if (df.loc[2, "frobenius_norm_diff"] > df.loc[1, "frobenius_norm_diff"]
            and df.loc[2, "frobenius_norm_diff"] > df.loc[3, "frobenius_norm_diff"])
        else
        "The fold-2-correlation-shift hypothesis is NOT supported by Frobenius norm alone."
    )

    report = f"""# Fold-2 Correlation Structure Shift

Date: 2026-04-15
Source: 100-ticker biotech panel log-return series, walk-forward
folds, pairwise Pearson correlation across tickers computed per
slice with min-overlap of 30 joint active days.

## 1. Question

Does fold 2's test-window pairwise ticker correlation structure
differ more from its train-window correlation structure than for
folds 1 or 3? If yes, this supports the leading remaining
hypothesis from `docs/fold2_combined_analysis.md` Section 2.1:
correlation structure shift is the mechanism behind pure STAR's
fold-2 failure.

## 2. Summary

{df_to_md(summary)}

Column definitions (comparing train-fold vs test-fold correlation
matrices C_train, C_test, both N x N symmetric with 1 on diagonal):
- `frobenius_norm_diff`: sqrt of sum of squared off-diagonal
  differences in C_train - C_test. Higher = more structural shift.
- `mean_abs_diff`: mean absolute off-diagonal difference. Scale-
  invariant per-pair mean.
- `rmse_diff`: root mean squared off-diagonal difference.
- `spectral_norm_diff`: largest singular value of C_train - C_test
  (NaNs imputed with 0).
- `sign_flip_fraction`: fraction of off-diagonal pairs whose
  correlation sign flipped between train and test.
- `mean_train_corr`, `mean_test_corr`: average off-diagonal
  correlation level in each slice (context, not shift).

## 3. Ranking (higher = more shifted)

- By Frobenius norm: **{frob_rank}**
- By RMSE: **{rmse_rank}**
- By spectral norm: **{spec_rank}**
- By sign-flip fraction: **{flip_rank}**
- By mean |Δ|: **{shift_lvl_rank}**

## 4. Verdict

{verdict}

## 5. Interpretation

Context: mean test-window correlation level shows how strongly
tickers co-move on average during each test period. If fold 2's
mean correlation is substantially higher or lower than fold 2's
mean training correlation, that is a level-shift story (returns
get more correlated during drawdowns, e.g.). If the sign-flip
fraction is elevated on fold 2, that is a structural story (pairs
that were positively correlated in training become negative
predictors during the drawdown).

The paper's fold-2 discussion can cite whichever metric is most
supportive, accompanied by the table above for full transparency.

## 6. Follow-up

If the Frobenius or spectral shift is substantially larger on
fold 2:
- Re-state fold-2 failure mechanism as correlation-structure shift
  with this table as evidence.
- Motivate time-varying / adaptive correlation-graph methods
  directly from the empirical gap.

If the shift is not substantially larger on fold 2 than on fold 1
or 3:
- Correlation-structure shift is not the dominant mechanism.
- Revisit attention-pattern brittleness as the next hypothesis.
- Consider a regime-gated attention experiment as the direct test.

## 7. Data exports

- `docs/fold2_correlation_shift_summary.csv`
- `docs/fold2_correlation_shift_per_fold.csv`
"""
    OUT_MD.write_text(report)
    print(f"\nwrote {OUT_MD}")
    print(f"wrote {OUT_SUM}")
    print(f"wrote {OUT_FULL}")


if __name__ == "__main__":
    main()
