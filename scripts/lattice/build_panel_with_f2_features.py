"""Build the F2-targeted feature panel at data/lattice_f2feats/processed/.

Adds three new columns engineered from the existing panel + macro state:

  rate_beta_60d         Rolling 60-day regression beta of per-ticker log_return
                        against DGS10 daily change. Captures "this ticker moves
                        opposite to rates" — the direct F2 mechanism.
  delta_rv_20d          realized_vol_20d - realized_vol_60d. Vol-acceleration
                        signal; positive (vol expanding) is F2-bad.
  low_vol_decile_flag   Cross-sectional rank of realized_vol_60d per day,
                        binned to deciles {0..9}. Discrete, regime-stable
                        encoding of the dominant F2 signal identified by
                        the per-feature IC sweep.

The new panel has 26 + 3 = 29 feature columns. The canonical panel and the
currently running v2 sweeps are not touched. Per-fold scalers are fit
at experiments/lattice_f2feats/foldX/scalers.pkl.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from src.lattice.data.folds import fold_indices
from src.lattice.training.standardise import fit_fold_scaler, save_fold_scaler


SRC_DIR = Path("data/lattice/processed")
DST_DIR = Path("data/lattice_f2feats/processed")
SCALER_ROOT = Path("experiments/lattice_f2feats")

ROLLING_RATE_BETA_WINDOW = 60


def _rolling_beta(ret: pd.Series, rate_chg: pd.Series, window: int) -> pd.Series:
    """Rolling slope of OLS regression ret ~ rate_chg over `window` days.

    Uses the closed-form beta = cov(x,y) / var(x). NaN if window short.
    """
    cov = ret.rolling(window, min_periods=20).cov(rate_chg)
    var = rate_chg.rolling(window, min_periods=20).var()
    beta = cov / var.replace(0, np.nan)
    return beta.clip(lower=-5.0, upper=5.0)


def _build_rate_beta(panel: pd.DataFrame, macro: pd.DataFrame) -> pd.Series:
    """Per-(ticker, date) rolling rate beta vs DGS10 daily change."""
    rate = macro.set_index("date")["dgs10"].astype(float)
    rate_chg = rate.diff().rename("rate_chg")
    print(f"[f2feats] rate_chg coverage: {rate_chg.notna().sum()} / "
          f"{len(rate_chg)} dates", flush=True)

    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    panel = panel.merge(
        rate_chg.reset_index(), how="left", on="date",
    )

    out = np.full(len(panel), np.nan, dtype=np.float64)
    for ticker, grp in panel.groupby("ticker", sort=False):
        if len(grp) < ROLLING_RATE_BETA_WINDOW + 5:
            continue
        ret = grp["log_return"].astype(float)
        rc = grp["rate_chg"].astype(float)
        b = _rolling_beta(ret, rc, ROLLING_RATE_BETA_WINDOW)
        out[grp.index] = b.values
    s = pd.Series(out, index=panel.index, name="rate_beta_60d")
    return s.fillna(0.0).astype(np.float32)


def _build_delta_rv(panel: pd.DataFrame) -> pd.Series:
    """realized_vol_20d - realized_vol_60d. Vol acceleration."""
    s = (panel["realized_vol_20d"].astype(float)
         - panel["realized_vol_60d"].astype(float))
    return s.fillna(0.0).astype(np.float32)


def _build_low_vol_decile(panel: pd.DataFrame) -> pd.Series:
    """Cross-sectional decile rank of realized_vol_60d, per day."""
    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"])
    out = panel.groupby("date")["realized_vol_60d"].transform(
        lambda x: pd.qcut(x, 10, labels=False, duplicates="drop")
        if x.notna().sum() >= 20 else pd.Series([np.nan] * len(x), index=x.index),
    )
    return out.fillna(-1).astype(np.int8)


def main() -> None:
    DST_DIR.mkdir(parents=True, exist_ok=True)
    SCALER_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"[f2feats] copying canonical artifacts to {DST_DIR}", flush=True)
    for name in ("panel_features.parquet", "cohorts.parquet",
                 "active_mask.parquet", "macro_state.parquet"):
        src = SRC_DIR / name
        if not src.exists():
            raise FileNotFoundError(f"missing canonical artifact: {src}")
        shutil.copyfile(src, DST_DIR / name)

    panel = pd.read_parquet(DST_DIR / "panel_features.parquet")
    macro = pd.read_parquet(DST_DIR / "macro_state.parquet")
    panel["date"] = pd.to_datetime(panel["date"])
    macro["date"] = pd.to_datetime(macro["date"])
    print(f"[f2feats] panel rows={len(panel):,} tickers={panel.ticker.nunique()} "
          f"dates={panel.date.nunique()}", flush=True)

    print("[f2feats] computing rate_beta_60d ...", flush=True)
    rate_beta = _build_rate_beta(panel, macro)
    print("[f2feats] computing delta_rv_20d ...", flush=True)
    delta_rv = _build_delta_rv(panel)
    print("[f2feats] computing low_vol_decile_flag ...", flush=True)
    decile = _build_low_vol_decile(panel)

    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    panel["rate_beta_60d"] = rate_beta.values
    panel["delta_rv_20d"] = delta_rv.values
    panel["low_vol_decile_flag"] = decile.values

    print(f"[f2feats] rate_beta_60d: nonzero={int((panel['rate_beta_60d']!=0).sum()):,} "
          f"mean={panel['rate_beta_60d'].mean():+.4f} "
          f"std={panel['rate_beta_60d'].std():.4f} "
          f"range=[{panel['rate_beta_60d'].min():+.4f}, {panel['rate_beta_60d'].max():+.4f}]",
          flush=True)
    print(f"[f2feats] delta_rv_20d: nonzero={int((panel['delta_rv_20d']!=0).sum()):,} "
          f"mean={panel['delta_rv_20d'].mean():+.5f} "
          f"std={panel['delta_rv_20d'].std():.5f}", flush=True)
    print(f"[f2feats] low_vol_decile_flag: values={sorted(panel['low_vol_decile_flag'].unique())}",
          flush=True)

    out_path = DST_DIR / "panel_features.parquet"
    panel.to_parquet(out_path, index=False)
    print(f"[f2feats] wrote {out_path}: 29 feature cols", flush=True)

    print(f"[f2feats] fitting per-fold scalers to {SCALER_ROOT}", flush=True)
    cohorts = pd.read_parquet(DST_DIR / "cohorts.parquet")
    cohorts["date"] = pd.to_datetime(cohorts["date"])
    dates = sorted(panel["date"].unique())
    target_folds = (1, 2, 3, 4, 5) if max(dates) >= pd.Timestamp("2024-01-01") else (1, 2, 3)
    for fold in target_folds:
        train_idx, val_idx, test_idx = fold_indices(fold, dates)
        scaler = fit_fold_scaler(panel, train_idx, cohorts, fold)
        save_fold_scaler(scaler, SCALER_ROOT / f"fold{fold}/scalers.pkl")
        print(f"  fold {fold}: train={len(train_idx)} val={len(val_idx)} "
              f"test={len(test_idx)}", flush=True)

    print(f"[f2feats] DONE; panel at {DST_DIR}", flush=True)


if __name__ == "__main__":
    main()
