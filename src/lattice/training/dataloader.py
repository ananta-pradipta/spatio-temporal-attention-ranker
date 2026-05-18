"""Data loading + per-day batching for LATTICE training.

Constructs the input tensors that LATTICE.forward() expects from the
parquet artifacts produced by Phase 1 + Phase 2.

The trainer typically processes one trading day at a time (B=1) because:
  - The cross-sectional ranking task ranks all active tickers within a day.
  - Cross-day batching adds complexity for the IPO retrieval which already
    handles per-(day, ticker) keys.

This module exposes:
  LatticeDayBatch: a dataclass with all tensors needed for one day's forward.
  LatticeDataPrep: a class that loads parquets once and yields day batches.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from src.lattice.data.build_panel import (
    PANEL_FEATURE_COLS, ST_FEATURE_COLS, MACRO_FEATURE_COLS,
)
from src.lattice.data.episode_keys import (
    REGIME_KEY_DIM, NOVELTY_KEY_DIM,
    build_regime_key_tensor, build_novelty_key_tensor,
    compute_first_panel_idx_per_ticker, compute_idiovol_60d_proxy,
)
from src.lattice.data.sector_projection import (
    build_or_load_sector_projection, project_sector,
)
from src.lattice.training.standardise import apply_fold_scaler, load_fold_scaler


SECTOR_TO_ID = {
    "Communication Services": 0, "Consumer Discretionary": 1,
    "Consumer Staples": 2, "Energy": 3, "Financials": 4,
    "Health Care": 5, "Industrials": 6, "Information Technology": 7,
    "Materials": 8, "Real Estate": 9, "Utilities": 10,
}


@dataclass
class LatticeDayBatch:
    """One day's batch for LATTICE.forward(). All tensors have B=1."""

    panel_features: Tensor              # [1, N, T_lookback, F_panel]
    macro_state: Tensor                  # [1, F_macro]
    cohort_size_decile: Tensor           # [1, N] long
    cohort_liquidity_decile: Tensor      # [1, N] long
    cohort_sector_id: Tensor             # [1, N] long
    cohort_age_bucket: Tensor            # [1, N] long
    regime_query_keys: Tensor            # [1, K_regime]
    novelty_query_keys: Tensor           # [1, N, K_novelty]
    novelty_sector_ids: Tensor           # [1, N] long
    active_mask: Tensor                  # [1, N] bool
    day_index: Tensor                    # [1] long
    corr_neighbor_idx: Tensor            # [1, N, K_top]
    corr_neighbor_mask: Tensor           # [1, N, K_top]
    y_target: Tensor                     # [1, N] (5-day fwd log return; for loss)
    tickers: list                        # list of N strings


class LatticeDataPrep:
    """Loads Phase 1 parquets and yields day batches with fold-scaled features."""

    def __init__(
        self, fold: int, lookback: int = 60,
        processed_dir: Path = Path("data/lattice/processed"),
        scaler_dir: Path = Path("experiments/lattice"),
        build_episode_keys: bool = True,
    ) -> None:
        self.fold = fold
        self.lookback = lookback
        self.build_episode_keys = build_episode_keys

        self.panel = pd.read_parquet(processed_dir / "panel_features.parquet")
        self.cohorts = pd.read_parquet(processed_dir / "cohorts.parquet")
        self.st = pd.read_parquet(processed_dir / "stocktwits_features.parquet")
        self.macro = pd.read_parquet(processed_dir / "macro_state.parquet")
        self.panel["date"] = pd.to_datetime(self.panel["date"])
        self.cohorts["date"] = pd.to_datetime(self.cohorts["date"])
        self.st["date"] = pd.to_datetime(self.st["date"])
        self.macro["date"] = pd.to_datetime(self.macro["date"])

        # Apply per-fold scaler to panel features
        scaler = load_fold_scaler(scaler_dir / f"fold{fold}/scalers.pkl")
        self.panel_scaled = apply_fold_scaler(self.panel, self.cohorts, scaler)

        self.dates = sorted(self.panel["date"].unique())
        self.tickers_universe = sorted(self.panel["ticker"].unique())
        self.n_universe = len(self.tickers_universe)
        self.ticker_to_idx = {t: i for i, t in enumerate(self.tickers_universe)}
        self.date_to_idx = {d: i for i, d in enumerate(self.dates)}

        # Pre-build dense [T, N, F_panel] tensor for fast slicing.
        # _panel_tensor: fold-scaled features (sector-z-scored, fold-z); used
        #   by encoder/aggregator/etc.
        # _panel_tensor_raw: pre-scaling features; used by episode-key
        #   construction so that cs-mean of returns is the actual cs-mean,
        #   not approximately zero (post-sector-z-score).
        F_panel = len(PANEL_FEATURE_COLS)
        self._panel_tensor = np.zeros((len(self.dates), self.n_universe, F_panel),
                                        dtype=np.float32)
        self._panel_tensor_raw = np.zeros(
            (len(self.dates), self.n_universe, F_panel), dtype=np.float32,
        )
        self._mask_tensor = np.zeros((len(self.dates), self.n_universe), dtype=bool)
        self._y_tensor = np.zeros((len(self.dates), self.n_universe), dtype=np.float32)

        # Vectorised fill: align both panel and panel_scaled on (date, ticker)
        # indexes to avoid per-row pandas lookup.
        panel_aligned = self.panel.assign(
            di=self.panel["date"].map(self.date_to_idx),
            ti=self.panel["ticker"].map(self.ticker_to_idx),
        ).dropna(subset=["di", "ti"])
        panel_aligned["di"] = panel_aligned["di"].astype(int)
        panel_aligned["ti"] = panel_aligned["ti"].astype(int)
        scaled_aligned = self.panel_scaled.assign(
            di=self.panel_scaled["date"].map(self.date_to_idx),
            ti=self.panel_scaled["ticker"].map(self.ticker_to_idx),
        ).dropna(subset=["di", "ti"])
        scaled_aligned["di"] = scaled_aligned["di"].astype(int)
        scaled_aligned["ti"] = scaled_aligned["ti"].astype(int)
        raw_vals = panel_aligned[PANEL_FEATURE_COLS].to_numpy(dtype=np.float32)
        scaled_vals = scaled_aligned[PANEL_FEATURE_COLS].to_numpy(dtype=np.float32)
        self._panel_tensor_raw[panel_aligned["di"].values, panel_aligned["ti"].values] = raw_vals
        self._panel_tensor[scaled_aligned["di"].values, scaled_aligned["ti"].values] = scaled_vals
        self._mask_tensor[scaled_aligned["di"].values, scaled_aligned["ti"].values] = True
        self._y_tensor[scaled_aligned["di"].values, scaled_aligned["ti"].values] = (
            scaled_aligned["fwd_return_h"].to_numpy(dtype=np.float32)
        )

        # Cohort tensor [T, N, 4]: size, liquidity, sector, age
        self._cohort_size = np.full((len(self.dates), self.n_universe), -1, dtype=np.int64)
        self._cohort_liq = np.full((len(self.dates), self.n_universe), -1, dtype=np.int64)
        self._cohort_sector = np.full((len(self.dates), self.n_universe), -1, dtype=np.int64)
        self._cohort_age = np.full((len(self.dates), self.n_universe), -1, dtype=np.int64)
        for _, row in self.cohorts.iterrows():
            di = self.date_to_idx.get(row["date"])
            ti = self.ticker_to_idx.get(row["ticker"])
            if di is None or ti is None:
                continue
            sd = row["size_decile"]
            ld = row["liquidity_decile"]
            sec = row["sector"]
            ab = row["age_bucket"]
            if pd.notna(sd): self._cohort_size[di, ti] = int(sd)
            if pd.notna(ld): self._cohort_liq[di, ti] = int(ld)
            if isinstance(sec, str) and sec in SECTOR_TO_ID:
                self._cohort_sector[di, ti] = SECTOR_TO_ID[sec]
            if pd.notna(ab): self._cohort_age[di, ti] = int(ab)

        # StockTwits per-(date, ticker) features
        F_st = len(ST_FEATURE_COLS)
        self._st_tensor = np.zeros((len(self.dates), self.n_universe, F_st), dtype=np.float32)
        st_in = self.st[self.st["ticker"].isin(self.ticker_to_idx)].copy()
        st_in["di"] = st_in["date"].map(self.date_to_idx)
        st_in["ti"] = st_in["ticker"].map(self.ticker_to_idx)
        st_in = st_in.dropna(subset=["di", "ti"])
        for _, row in st_in.iterrows():
            di = int(row["di"]); ti = int(row["ti"])
            self._st_tensor[di, ti] = row[ST_FEATURE_COLS].to_numpy(dtype=np.float32)

        # Macro state [T, F_macro]
        self._macro_tensor = np.zeros((len(self.dates), len(MACRO_FEATURE_COLS)),
                                        dtype=np.float32)
        macro_indexed = self.macro.set_index("date")
        for di, d in enumerate(self.dates):
            if d in macro_indexed.index:
                row = macro_indexed.loc[d]
                if isinstance(row, pd.Series):
                    vals = row[MACRO_FEATURE_COLS].fillna(0.0).to_numpy(dtype=np.float32)
                    self._macro_tensor[di] = vals

        # Build correlation graph top-K neighbors per day
        # (60-day reliability-shrunk Pearson, computed lazily on demand)
        self._corr_neighbor_idx_cache: dict = {}
        self._corr_neighbor_mask_cache: dict = {}

        # Phase 5b: regime + novelty retrieval-bank keys.
        # Train-fold-only z-scoring stats fitted here; pre-compute per-day
        # query keys for the full panel so make_batch is a tensor lookup.
        self._regime_keys: np.ndarray | None = None
        self._novelty_keys: np.ndarray | None = None
        self._novelty_eligible: np.ndarray | None = None
        self._first_panel_idx: np.ndarray | None = None
        self.regime_stats = None
        self.novelty_stats = None
        self.idiovol_tensor: np.ndarray | None = None
        self._sector_per_ticker: np.ndarray | None = None
        self._sector_proj_scalar_per_ticker: np.ndarray | None = None
        if self.build_episode_keys:
            self._build_episode_keys()

    def _build_episode_keys(self) -> None:
        """Phase 5b: fit train-fold stats, pre-compute regime + novelty keys."""
        from src.lattice.data.folds import fold_indices
        train_idx, _, _ = fold_indices(self.fold, self.dates)

        sector_per_ticker = np.full(self.n_universe, -1, dtype=np.int64)
        for n in range(self.n_universe):
            col = self._cohort_sector[:, n]
            valid = col[col >= 0]
            if valid.size:
                from collections import Counter
                ctr = Counter(valid.tolist())
                sector_per_ticker[n] = int(ctr.most_common(1)[0][0])
        self._sector_per_ticker = sector_per_ticker

        proj_weight = build_or_load_sector_projection()
        proj_scalars = project_sector(proj_weight, sector_per_ticker)
        self._sector_proj_scalar_per_ticker = proj_scalars

        self._first_panel_idx = compute_first_panel_idx_per_ticker(self._mask_tensor)

        self.idiovol_tensor = compute_idiovol_60d_proxy(
            self._panel_tensor_raw, self._mask_tensor, sector_per_ticker,
        )

        self._regime_keys, self.regime_stats = build_regime_key_tensor(
            self._panel_tensor_raw, self._mask_tensor, self._macro_tensor, train_idx,
        )

        self._novelty_keys, self.novelty_stats, self._novelty_eligible = (
            build_novelty_key_tensor(
                self._panel_tensor_raw, self._mask_tensor, self._st_tensor,
                sector_per_ticker, proj_scalars, self._first_panel_idx,
                train_idx, idiovol_tensor=self.idiovol_tensor,
            )
        )

    def get_corr_neighbors(
        self, t: int, top_k: int = 8, window: int = 60, tau: float = 30.0,
        min_overlap: int = 5,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute reliability-shrunk top-K correlation neighbors for day t."""
        if t in self._corr_neighbor_idx_cache:
            return (self._corr_neighbor_idx_cache[t],
                     self._corr_neighbor_mask_cache[t])
        if t < window:
            idx = np.zeros((self.n_universe, top_k), dtype=np.int64)
            msk = np.zeros((self.n_universe, top_k), dtype=bool)
            self._corr_neighbor_idx_cache[t] = idx
            self._corr_neighbor_mask_cache[t] = msk
            return idx, msk

        # log_return is feature index 0 in PANEL_FEATURE_COLS
        ret_window = self._panel_tensor[t - window + 1: t + 1, :, 0]  # [W, N]
        mask_window = self._mask_tensor[t - window + 1: t + 1]         # [W, N]
        # Mask cells where ticker isn't tradable
        n_overlap = mask_window.astype(np.int32).sum(axis=0)              # [N]
        # Pearson correlation per pair (N, N) over the window's valid cells
        ret_centered = ret_window - ret_window.mean(axis=0, keepdims=True)
        ret_centered = np.nan_to_num(ret_centered, nan=0.0)
        denom = (ret_centered ** 2).sum(axis=0)                           # [N]
        denom = np.where(denom < 1e-9, 1e-9, denom)
        cov = ret_centered.T @ ret_centered                                # [N, N]
        rho_raw = cov / np.sqrt(denom[:, None] * denom[None, :])
        # Reliability shrinkage by overlap count (use min of pair)
        n_overlap_pair = np.minimum(n_overlap[:, None], n_overlap[None, :])
        shrink = n_overlap_pair / (n_overlap_pair + tau)
        rho_shrunk = rho_raw * shrink
        # Mask self-pairs and below-min-overlap pairs
        rho_shrunk[n_overlap_pair < min_overlap] = -np.inf
        np.fill_diagonal(rho_shrunk, -np.inf)
        # Mask inactive tickers (today)
        active_today = self._mask_tensor[t]
        rho_shrunk[~active_today, :] = -np.inf
        rho_shrunk[:, ~active_today] = -np.inf
        # Top-K per row
        top_idx = np.argsort(-rho_shrunk, axis=1)[:, :top_k]              # [N, K]
        valid = np.take_along_axis(rho_shrunk, top_idx, axis=1) > -np.inf
        self._corr_neighbor_idx_cache[t] = top_idx.astype(np.int64)
        self._corr_neighbor_mask_cache[t] = valid
        return top_idx.astype(np.int64), valid

    def make_batch(
        self, day_index: int,
        regime_query_key: np.ndarray | None = None,
        novelty_query_keys: np.ndarray | None = None,
    ) -> LatticeDayBatch:
        """Build one day's batch.

        Args:
            day_index: integer index into self.dates.
            regime_query_key: [K_regime] per-day fingerprint.
                If None, uses zero-vector (regime memory will be near-noop).
            novelty_query_keys: [N, K_novelty] per-(day, ticker) signature.
                If None, uses zero-vector (novelty memory will be near-noop).
        """
        t = day_index
        T_back = self.lookback
        if t < T_back:
            # pad start of window with zeros
            window = np.zeros((T_back, self.n_universe, self._panel_tensor.shape[-1]),
                                dtype=np.float32)
            window[T_back - (t + 1):] = self._panel_tensor[: t + 1]
        else:
            window = self._panel_tensor[t - T_back + 1: t + 1]            # [T_back, N, F]

        active_today = self._mask_tensor[t]
        y_today = self._y_tensor[t]
        macro_today = self._macro_tensor[t]
        size_today = self._cohort_size[t]
        liq_today = self._cohort_liq[t]
        sec_today = self._cohort_sector[t]
        age_today = self._cohort_age[t]

        if regime_query_key is None:
            if self._regime_keys is not None:
                regime_query_key = self._regime_keys[t]
            else:
                regime_query_key = np.zeros(14, dtype=np.float32)
        if novelty_query_keys is None:
            if self._novelty_keys is not None:
                novelty_query_keys = self._novelty_keys[t]
            else:
                novelty_query_keys = np.zeros((self.n_universe, 8), dtype=np.float32)

        corr_idx, corr_mask = self.get_corr_neighbors(t)

        # window: [T_back, N, F] -> [B=1, N, T_back, F]
        panel_features = window.transpose(1, 0, 2)[None, :, :, :]
        return LatticeDayBatch(
            panel_features=torch.from_numpy(panel_features.copy()).float(),
            macro_state=torch.from_numpy(macro_today[None, :].copy()).float(),
            cohort_size_decile=torch.from_numpy(size_today[None, :].copy()).long(),
            cohort_liquidity_decile=torch.from_numpy(liq_today[None, :].copy()).long(),
            cohort_sector_id=torch.from_numpy(sec_today[None, :].copy()).long(),
            cohort_age_bucket=torch.from_numpy(age_today[None, :].copy()).long(),
            regime_query_keys=torch.from_numpy(regime_query_key[None, :].copy()).float(),
            novelty_query_keys=torch.from_numpy(novelty_query_keys[None, :, :].copy()).float(),
            novelty_sector_ids=torch.from_numpy(sec_today[None, :].copy()).long(),
            active_mask=torch.from_numpy(active_today[None, :].copy()).bool(),
            day_index=torch.tensor([t], dtype=torch.long),
            corr_neighbor_idx=torch.from_numpy(corr_idx[None, :, :].copy()).long(),
            corr_neighbor_mask=torch.from_numpy(corr_mask[None, :, :].copy()).bool(),
            y_target=torch.from_numpy(y_today[None, :].copy()).float(),
            tickers=self.tickers_universe,
        )


__all__ = ["LatticeDataPrep", "LatticeDayBatch", "SECTOR_TO_ID"]
