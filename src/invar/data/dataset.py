"""InVAR data adapter over the LATTICE S&P 500 panel.

Reads the LATTICE processed parquets, applies the per-fold pickled
``FoldScaler``, and yields per-day cross-sectional batches with shape::

    features : (N_t, L, F)   # ticker x lookback x feature
    macro    : (L, F_macro)  # macro lookback for the query day
    y_cs     : (N_t,)        # cross-sectional z-score of fwd_return_h
    mask     : (N_t,)        # bool, True for active tickers on the day
    cohort_meta : dict of (N_t,) tensors

``L = 60`` trading days, ``F = 26`` panel features, ``F_macro = 24``.
``N_t`` varies day to day; the dataset does not pad to a fixed N.

Auxiliary supervision computed train-fold-only:
- 20-day forward realised vol per (date, ticker), used by the VolHead.
- GaussianMixture(K=8) regime labels per training day, used by the
  RegimeClassifierHead.

The dataset reuses LATTICE artifacts but does not depend on
``src/lattice/training/dataloader.py``; only the pickle-compat import of
``FoldScaler`` and the ``SECTOR_TO_ID`` mapping cross the boundary.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from src.lattice.training.dataloader import SECTOR_TO_ID
from src.lattice.training.standardise import (
    apply_fold_scaler, load_fold_scaler,
)
from src.lattice.data.build_panel import (
    PANEL_FEATURE_COLS, MACRO_FEATURE_COLS,
)
from src.lattice.data.folds import FOLDS, fold_indices, EMBARGO_DAYS


PANEL_FEATURE_DIM = len(PANEL_FEATURE_COLS)
MACRO_FEATURE_DIM = len(MACRO_FEATURE_COLS)
DEFAULT_LOOKBACK = 60
DEFAULT_VOL_HORIZON = 20
N_REGIME_LABELS = 8

# Environment-variable override for the processed parquet directory and the
# scaler directory. Lets the augmented-panel build live at a sibling path
# (e.g. data/lattice_aug/processed) without forking every training script.
# Defaults reproduce the original behavior exactly.
_DEFAULT_PROCESSED_DIR = Path(
    os.environ.get("LATTICE_PROCESSED_DIR", "data/lattice/processed"),
)
_DEFAULT_SCALER_DIR = Path(
    os.environ.get("LATTICE_SCALER_DIR", "experiments/lattice"),
)


@dataclass
class InvarDayBatch:
    """One trading-day batch for INVAR.forward()."""

    features: Tensor          # (N_t, L, F)
    macro: Tensor             # (L, F_macro)
    y_cs: Tensor              # (N_t,)
    mask: Tensor              # (N_t,) bool, all True for active tickers
    sector_id: Tensor         # (N_t,) long
    size_decile: Tensor       # (N_t,) long, -1 where missing
    age_bucket: Tensor        # (N_t,) long
    regime_label: int         # GaussianMixture(K=8) cluster id for the day
    fwd_vol_20d: Tensor       # (N_t,) float, 20-day forward realised vol
    has_fwd_vol: Tensor       # (N_t,) bool, True iff 20-day forward exists
    day_index: int            # integer day index in the panel calendar
    date: pd.Timestamp        # absolute date for the query
    tickers: list[str]        # length N_t, ordered to match feature/y rows


def cross_sectional_zscore(y: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Z-score y over the active subset; returns 0 where mask is False."""
    out = np.zeros_like(y, dtype=np.float32)
    if not mask.any():
        return out
    vals = y[mask]
    finite = np.isfinite(vals)
    if not finite.any():
        return out
    vals = vals[finite]
    if vals.size < 2:
        return out
    mu = float(vals.mean())
    sigma = float(vals.std())
    if sigma < 1e-9:
        return out
    z = np.where(np.isfinite(y), (y - mu) / sigma, 0.0)
    out[mask] = z[mask].astype(np.float32)
    return out


class InvarDataset:
    """Per-day cross-sectional dataset over the LATTICE S&P 500 panel.

    Args:
        fold: 1, 2 or 3.
        split: one of {"train", "val", "test"}.
        processed_dir: directory holding the LATTICE processed parquets.
        scaler_dir: directory whose ``fold{F}/scalers.pkl`` lives.
        lookback: encoder window length (default 60 trading days).
        vol_horizon: forward window for the auxiliary realised-vol target
            (default 20 trading days).
        n_regimes: number of GaussianMixture components for the auxiliary
            regime label (default 8).
        regime_seed: seed for the GaussianMixture fit.
        require_min_active: minimum active tickers to yield a day (default 50).
    """

    def __init__(
        self,
        fold: int,
        split: str,
        processed_dir: Path = _DEFAULT_PROCESSED_DIR,
        scaler_dir: Path = _DEFAULT_SCALER_DIR,
        lookback: int = DEFAULT_LOOKBACK,
        vol_horizon: int = DEFAULT_VOL_HORIZON,
        n_regimes: int = N_REGIME_LABELS,
        regime_seed: int = 0,
        require_min_active: int = 50,
    ) -> None:
        if fold not in FOLDS:
            raise ValueError(f"fold must be in {{1, 2, 3}}, got {fold}")
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be train/val/test, got {split}")
        self.fold = fold
        self.split = split
        self.lookback = lookback
        self.vol_horizon = vol_horizon
        self.n_regimes = n_regimes
        self.require_min_active = require_min_active

        panel_path = processed_dir / "panel_features.parquet"
        cohort_path = processed_dir / "cohorts.parquet"
        macro_path = processed_dir / "macro_state.parquet"
        scaler_path = scaler_dir / f"fold{fold}/scalers.pkl"
        for p in (panel_path, cohort_path, macro_path, scaler_path):
            if not p.exists():
                raise FileNotFoundError(
                    f"Missing LATTICE artifact: {p}. See "
                    f"docs/lattice_dataset_catalogue.md for paths."
                )

        self.panel = pd.read_parquet(panel_path)
        self.cohorts = pd.read_parquet(cohort_path)
        self.macro = pd.read_parquet(macro_path)
        for df in (self.panel, self.cohorts, self.macro):
            df["date"] = pd.to_datetime(df["date"])

        # Determine the effective panel feature columns at load time. This
        # supports legacy panels (26 cols) and the 2026-05-12 augmented
        # panel (37 cols) without forking the codebase. The model's
        # n_features is then derived from `self.feature_dim`.
        self.feature_cols = [c for c in PANEL_FEATURE_COLS
                              if c in self.panel.columns]
        self.feature_dim = len(self.feature_cols)
        missing = [c for c in PANEL_FEATURE_COLS if c not in self.panel.columns]
        if missing:
            print(f"[InvarDataset] panel is missing {len(missing)} of "
                  f"{len(PANEL_FEATURE_COLS)} known feature columns; using "
                  f"effective feature_dim={self.feature_dim}. "
                  f"Missing: {missing}", flush=True)

        self.scaler = load_fold_scaler(scaler_path)
        self.panel_scaled = apply_fold_scaler(self.panel, self.cohorts, self.scaler)

        self.dates = sorted(self.panel["date"].unique())
        self.tickers_universe = sorted(self.panel["ticker"].unique())
        self.n_universe = len(self.tickers_universe)
        self.ticker_to_idx = {t: i for i, t in enumerate(self.tickers_universe)}
        self.date_to_idx = {d: i for i, d in enumerate(self.dates)}

        self._build_dense_tensors()
        self.train_idx, self.val_idx, self.test_idx = fold_indices(fold, self.dates)
        self._fit_regime_labels(regime_seed=regime_seed)
        self._build_fwd_vol()

        if split == "train":
            self.split_idx = self.train_idx
        elif split == "val":
            self.split_idx = self.val_idx
        else:
            self.split_idx = self.test_idx
        self._verify_embargo()

        self._eligible_idx = self._filter_eligible(self.split_idx)

    def _build_dense_tensors(self) -> None:
        """Build dense (T, N, F) panel + (T, F_macro) macro tensors."""
        T = len(self.dates)
        N = self.n_universe
        F = self.feature_dim
        self._panel_tensor = np.zeros((T, N, F), dtype=np.float32)
        self._panel_tensor_raw = np.zeros((T, N, F), dtype=np.float32)
        self._mask_tensor = np.zeros((T, N), dtype=bool)
        self._y_tensor = np.zeros((T, N), dtype=np.float32)
        self._cohort_sector = np.full((T, N), -1, dtype=np.int64)
        self._cohort_size = np.full((T, N), -1, dtype=np.int64)
        self._cohort_age = np.full((T, N), -1, dtype=np.int64)
        self._macro_tensor = np.zeros((T, MACRO_FEATURE_DIM), dtype=np.float32)
        self._macro_raw_tensor = np.zeros((T, MACRO_FEATURE_DIM), dtype=np.float32)

        scaled = self.panel_scaled.assign(
            di=self.panel_scaled["date"].map(self.date_to_idx),
            ti=self.panel_scaled["ticker"].map(self.ticker_to_idx),
        ).dropna(subset=["di", "ti"])
        scaled["di"] = scaled["di"].astype(int)
        scaled["ti"] = scaled["ti"].astype(int)
        self._panel_tensor[scaled["di"].values, scaled["ti"].values] = (
            scaled[self.feature_cols].to_numpy(dtype=np.float32)
        )
        self._mask_tensor[scaled["di"].values, scaled["ti"].values] = True
        self._y_tensor[scaled["di"].values, scaled["ti"].values] = (
            scaled["fwd_return_h"].to_numpy(dtype=np.float32)
        )
        raw = self.panel.assign(
            di=self.panel["date"].map(self.date_to_idx),
            ti=self.panel["ticker"].map(self.ticker_to_idx),
        ).dropna(subset=["di", "ti"])
        raw["di"] = raw["di"].astype(int)
        raw["ti"] = raw["ti"].astype(int)
        self._panel_tensor_raw[raw["di"].values, raw["ti"].values] = (
            raw[self.feature_cols].to_numpy(dtype=np.float32)
        )

        cohort = self.cohorts.assign(
            di=self.cohorts["date"].map(self.date_to_idx),
            ti=self.cohorts["ticker"].map(self.ticker_to_idx),
        ).dropna(subset=["di", "ti"])
        cohort["di"] = cohort["di"].astype(int)
        cohort["ti"] = cohort["ti"].astype(int)
        for _, row in cohort.iterrows():
            di = int(row["di"])
            ti = int(row["ti"])
            if pd.notna(row["size_decile"]):
                self._cohort_size[di, ti] = int(row["size_decile"])
            if pd.notna(row["age_bucket"]):
                self._cohort_age[di, ti] = int(row["age_bucket"])
            sec = row["sector"]
            if isinstance(sec, str) and sec in SECTOR_TO_ID:
                self._cohort_sector[di, ti] = SECTOR_TO_ID[sec]

        macro_indexed = self.macro.set_index("date")
        for di, d in enumerate(self.dates):
            if d in macro_indexed.index:
                row = macro_indexed.loc[d]
                if isinstance(row, pd.Series):
                    vals = row[MACRO_FEATURE_COLS].fillna(0.0).to_numpy(dtype=np.float32)
                    self._macro_raw_tensor[di] = vals

        # Z-score macro against TRAIN fold only (computed in fit_regime_labels).
        self._macro_tensor = self._macro_raw_tensor.copy()

    def _fit_regime_labels(self, regime_seed: int) -> None:
        """Fit GaussianMixture(K=N_REGIMES) on train-fold macro vectors.

        Also fits the train-fold-only z-score on macro and applies to all days.
        """
        train_macro = self._macro_raw_tensor[self.train_idx]
        finite_rows = np.isfinite(train_macro).all(axis=1)
        train_macro_clean = train_macro[finite_rows]
        macro_mean = train_macro_clean.mean(axis=0)
        macro_std = train_macro_clean.std(axis=0)
        macro_std = np.where(macro_std < 1e-9, 1.0, macro_std)
        self._macro_z_mean = macro_mean.astype(np.float32)
        self._macro_z_std = macro_std.astype(np.float32)
        self._macro_tensor = (self._macro_raw_tensor - macro_mean) / macro_std

        from sklearn.mixture import GaussianMixture
        gmm = GaussianMixture(
            n_components=self.n_regimes, covariance_type="full",
            random_state=regime_seed, max_iter=200, reg_covar=1e-4,
        )
        gmm.fit(self._macro_tensor[self.train_idx])
        self._gmm = gmm
        self._regime_labels = gmm.predict(self._macro_tensor).astype(np.int64)

    def _build_fwd_vol(self) -> None:
        """20-day forward realised vol per (date, ticker), train-fold only stats.

        Computed from raw log_return panel via rolling future-window std.
        Stored on every day for efficient lookup; whether a (date, ticker)
        cell has 20 valid forward days is captured in ``has_fwd_vol``.
        """
        log_ret_idx = self.feature_cols.index("log_return")
        T, N, _ = self._panel_tensor_raw.shape
        H = self.vol_horizon
        log_ret = self._panel_tensor_raw[..., log_ret_idx]
        log_ret = np.where(self._mask_tensor, log_ret, np.nan)

        fwd_vol = np.full((T, N), np.nan, dtype=np.float32)
        for t in range(T):
            t_end = min(T, t + H + 1)
            if t_end - (t + 1) < 5:
                continue
            window = log_ret[t + 1: t_end]
            with np.errstate(invalid="ignore"):
                vol = np.nanstd(window, axis=0)
            fwd_vol[t] = vol.astype(np.float32)
        self._fwd_vol_tensor = fwd_vol

        # Mark cells with at least 5 valid forward days.
        finite_count = np.zeros((T, N), dtype=np.int32)
        for t in range(T):
            t_end = min(T, t + H + 1)
            if t_end - (t + 1) < 1:
                continue
            window = log_ret[t + 1: t_end]
            finite_count[t] = (np.isfinite(window).astype(np.int32)).sum(axis=0)
        self._has_fwd_vol = (finite_count >= 5) & np.isfinite(fwd_vol)

    def _verify_embargo(self) -> None:
        """Verify the 5-day embargo at the train/val and val/test boundaries."""
        if not (len(self.train_idx) and len(self.val_idx)):
            return
        gap = int(self.val_idx[0]) - int(self.train_idx[-1])
        if gap < EMBARGO_DAYS:
            raise ValueError(
                f"Embargo violation: train/val gap {gap} < {EMBARGO_DAYS} on fold {self.fold}"
            )
        if not (len(self.val_idx) and len(self.test_idx)):
            return
        gap2 = int(self.test_idx[0]) - int(self.val_idx[-1])
        if gap2 < EMBARGO_DAYS:
            raise ValueError(
                f"Embargo violation: val/test gap {gap2} < {EMBARGO_DAYS} on fold {self.fold}"
            )

    def _filter_eligible(self, idx: np.ndarray) -> np.ndarray:
        """Drop days with insufficient lookback or insufficient active tickers."""
        keep = []
        for t in idx:
            t = int(t)
            if t < self.lookback - 1:
                continue
            n_active = int(self._mask_tensor[t].sum())
            if n_active < self.require_min_active:
                continue
            keep.append(t)
        return np.asarray(keep, dtype=np.int64)

    def __len__(self) -> int:
        return len(self._eligible_idx)

    def __iter__(self) -> Iterator[InvarDayBatch]:
        for t in self._eligible_idx:
            yield self.get(int(t))

    def get(self, day_index: int) -> InvarDayBatch:
        t = int(day_index)
        L = self.lookback
        active = self._mask_tensor[t]
        active_idx = np.where(active)[0]
        n_active = active_idx.size
        if n_active == 0:
            raise ValueError(f"No active tickers on day {t} ({self.dates[t]})")

        feat_window = self._panel_tensor[t - L + 1: t + 1, active_idx]
        feat_window = np.transpose(feat_window, (1, 0, 2))  # (N_t, L, F)
        macro_window = self._macro_tensor[t - L + 1: t + 1]

        y_raw = self._y_tensor[t]
        y_cs_full = cross_sectional_zscore(y_raw, active)
        y_cs = y_cs_full[active_idx]

        sector = self._cohort_sector[t, active_idx]
        size = self._cohort_size[t, active_idx]
        age = self._cohort_age[t, active_idx]
        fwd_vol = self._fwd_vol_tensor[t, active_idx]
        has_fwd_vol = self._has_fwd_vol[t, active_idx]

        return InvarDayBatch(
            features=torch.from_numpy(feat_window.copy()).float(),
            macro=torch.from_numpy(macro_window.copy()).float(),
            y_cs=torch.from_numpy(y_cs.astype(np.float32).copy()).float(),
            mask=torch.ones(n_active, dtype=torch.bool),
            sector_id=torch.from_numpy(sector.astype(np.int64).copy()).long(),
            size_decile=torch.from_numpy(size.astype(np.int64).copy()).long(),
            age_bucket=torch.from_numpy(age.astype(np.int64).copy()).long(),
            regime_label=int(self._regime_labels[t]),
            fwd_vol_20d=torch.from_numpy(
                np.nan_to_num(fwd_vol, nan=0.0).astype(np.float32).copy()
            ).float(),
            has_fwd_vol=torch.from_numpy(has_fwd_vol.copy()).bool(),
            day_index=t,
            date=pd.Timestamp(self.dates[t]),
            tickers=[self.tickers_universe[i] for i in active_idx],
        )


__all__ = [
    "InvarDataset", "InvarDayBatch", "cross_sectional_zscore",
    "PANEL_FEATURE_DIM", "MACRO_FEATURE_DIM", "DEFAULT_LOOKBACK",
    "DEFAULT_VOL_HORIZON", "N_REGIME_LABELS",
]
