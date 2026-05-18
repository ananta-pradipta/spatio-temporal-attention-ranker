"""MAiT (Macro-Adaptive iTransformer) data adapter.

Wraps `InvarDataset` per-day batches without modifying the dataset class.
The adapter:
  1. Drops two panel features (catalyst_type_id, has_stocktwits) -> 24 kept.
  2. Selects 17 of the 24 macro features (drops collinear or sector-overlap
     ETF returns and gld_5d_ret; keeps market_breadth_proxy since it is
     100 percent non-null in macro_state.parquet as of 2026-05-11).
  3. Extracts a 5-vector `regime_input` at the query day for the gate MLP.
  4. Persists the subset macro scaler stats (mean and std over the 17
     kept features, train-fold-only) to
     `experiments/lattice/fold{F}/macro_scaler_mait.pkl` for
     reproducibility. The actual scaling is already done by InvarDataset
     internally during `_fit_regime_labels`; this file records what the
     adapter consumes.

Design reference: `docs/mait_design.md` Section 5 (feature selection) and
Section 2.1 (per-day batch contract).
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import os

import numpy as np
import torch
from torch import Tensor

from src.invar.data.dataset import InvarDataset, InvarDayBatch
from src.lattice.data.build_panel import (
    MACRO_FEATURE_COLS,
    PANEL_FEATURE_COLS,
)


# Panel: drop catalyst_type_id (categorical, does not z-scale) and
# has_stocktwits (low-variance flag, StockTwits features unused).
_PANEL_DROP_NAMES = ["catalyst_type_id", "has_stocktwits"]

# Two macro presets selectable via the MAIT_PRESET environment variable.
#
# "v17" (original spec, MAiT design 2026-05-10): drops 7 of 24 macros
# (qqq, iwm, xly, xlp, xlu, xlre, gld) leaving 17 kept. Gate slot-5 is
# market_breadth_proxy.
#
# "minimal" (2026-05-11 pivot per docs/macro_feature_analysis.md): drops
# 16 of 24 leaving 8 regime-discriminating macros. Gate slot-5 is
# breakeven_10y. Use the minimal preset to test the macro-feature-noise
# hypothesis.
_PRESET = os.environ.get("MAIT_PRESET", "minimal").lower()

if _PRESET == "v17":
    _MACRO_DROP_NAMES = [
        "qqq_5d_ret",
        "iwm_5d_ret",
        "xly_5d_ret",
        "xlp_5d_ret",
        "xlu_5d_ret",
        "xlre_5d_ret",
        "gld_5d_ret",
    ]
    _REGIME_NAMES = [
        "vix",
        "vix_term_slope",
        "slope_2s10s",
        "hyg_5d_ret",
        "market_breadth_proxy",
    ]
    _SCALER_FILENAME = "macro_scaler_mait.pkl"
elif _PRESET == "minimal":
    _MACRO_DROP_NAMES = [
        "slope_3m10y",          # redundant with slope_2s10s
        "dxy_5d_ret",           # weak max |z|=1.08
        "tlt_5d_ret",           # redundant with dgs10
        "gld_5d_ret",           # weak discriminator
        "spy_5d_ret",           # collinear with sector ETFs
        "qqq_5d_ret", "iwm_5d_ret",
        "xlk_5d_ret", "xlf_5d_ret", "xle_5d_ret", "xlv_5d_ret",
        "xly_5d_ret", "xlp_5d_ret", "xlu_5d_ret", "xlre_5d_ret",
        "market_breadth_proxy",
    ]
    _REGIME_NAMES = [
        "vix",
        "vix_term_slope",
        "slope_2s10s",
        "hyg_5d_ret",
        "breakeven_10y",
    ]
    _SCALER_FILENAME = "macro_scaler_mait_minimal.pkl"
else:
    raise ValueError(
        f"unknown MAIT_PRESET={_PRESET!r}; expected 'v17' or 'minimal'",
    )


def _resolve_keep_indices(panel_cols: list[str] | None = None) -> dict:
    """Compute panel and macro keep indices.

    Args:
        panel_cols: the actual panel feature column ordering provided by a
            specific dataset instance. Defaults to PANEL_FEATURE_COLS (the
            full 37-col augmented panel). Passing the dataset's
            ``feature_cols`` makes the adapter compatible with both the
            legacy 26-col panel and the 37-col augmented panel.
    """
    panel_cols = panel_cols if panel_cols is not None else list(PANEL_FEATURE_COLS)
    panel_keep_idx = [i for i, name in enumerate(panel_cols)
                      if name not in _PANEL_DROP_NAMES]
    macro_keep_idx = [i for i, name in enumerate(MACRO_FEATURE_COLS)
                      if name not in _MACRO_DROP_NAMES]
    kept_macro_names = [MACRO_FEATURE_COLS[i] for i in macro_keep_idx]
    regime_idx_in_kept = []
    for name in _REGIME_NAMES:
        if name not in kept_macro_names:
            raise ValueError(
                f"regime feature {name!r} is not in the kept macro list; "
                f"update _REGIME_NAMES or _MACRO_DROP_NAMES",
            )
        regime_idx_in_kept.append(kept_macro_names.index(name))
    return dict(
        panel_keep_idx=panel_keep_idx,
        macro_keep_idx=macro_keep_idx,
        regime_idx_in_kept=regime_idx_in_kept,
        kept_macro_names=kept_macro_names,
        kept_panel_names=[panel_cols[i] for i in panel_keep_idx],
    )


KEEP = _resolve_keep_indices()
N_PANEL_KEPT = len(KEEP["panel_keep_idx"])
N_MACRO_KEPT = len(KEEP["macro_keep_idx"])
REGIME_DIM = len(KEEP["regime_idx_in_kept"])


@dataclass
class MaitBatch:
    x_panel: Tensor          # (N_t, N_PANEL_KEPT=24, L=60)
    x_macro_lookback: Tensor  # (N_MACRO_KEPT=17, L=60)
    regime_input: Tensor     # (REGIME_DIM=5,) query-day scalars
    y_cs: Tensor             # (N_t,)
    mask: Tensor             # (N_t,) bool
    day_index: int
    date: object             # pd.Timestamp


class MaitBatchAdapter:
    """Wraps an `InvarDataset` to produce MaitBatch objects.

    The adapter does NOT modify InvarDataset; it only consumes the
    per-day InvarDayBatch. The MaitBatchAdapter also persists a subset
    of InvarDataset's train-fold-only macro z-scaler stats to
    `experiments/lattice/fold{F}/macro_scaler_mait.pkl`. The persisted
    file records the (mean, std, names) for the 17 kept macro features
    over the train split. If the file already exists, load it; do not
    refit.
    """

    def __init__(self, dataset: InvarDataset,
                 scaler_dir: Path | None = None,
                 ensure_scaler_persisted: bool = True) -> None:
        if scaler_dir is None:
            scaler_dir = Path(os.environ.get(
                "LATTICE_SCALER_DIR", "experiments/lattice",
            ))
        self.dataset = dataset
        self.fold = dataset.fold
        # Compute panel keep-idx against the dataset's actual feature
        # ordering (supports legacy 26-col + augmented 37-col panels).
        ds_cols = getattr(dataset, "feature_cols", None)
        local_keep = _resolve_keep_indices(ds_cols) if ds_cols else KEEP
        self.local_keep = local_keep
        self.panel_keep_idx = torch.tensor(local_keep["panel_keep_idx"], dtype=torch.long)
        self.macro_keep_idx = torch.tensor(local_keep["macro_keep_idx"], dtype=torch.long)
        self.regime_idx_in_kept = torch.tensor(
            local_keep["regime_idx_in_kept"], dtype=torch.long,
        )
        self.n_panel_kept = len(local_keep["panel_keep_idx"])
        self.n_macro_kept = len(local_keep["macro_keep_idx"])
        # Filename suffixed so the 8-feature minimal preset does not collide
        # with the older 17-feature scaler files persisted earlier today.
        self.scaler_path = (
            scaler_dir / f"fold{self.fold}" / _SCALER_FILENAME
        )
        self.scaler = self._load_or_save_scaler(ensure_scaler_persisted)

    def _load_or_save_scaler(self, ensure_persisted: bool) -> dict:
        if self.scaler_path.exists():
            with open(self.scaler_path, "rb") as f:
                return pickle.load(f)
        # First-time setup: take the (already-fitted, train-fold-only)
        # macro z-stats from InvarDataset and subset to the 17 kept macros.
        full_mean = self.dataset._macro_z_mean
        full_std = self.dataset._macro_z_std
        macro_keep = KEEP["macro_keep_idx"]
        scaler = dict(
            mean=full_mean[macro_keep].astype(np.float32),
            std=full_std[macro_keep].astype(np.float32),
            feature_names=KEEP["kept_macro_names"],
            regime_feature_names=_REGIME_NAMES,
            fold=self.fold,
        )
        if ensure_persisted:
            self.scaler_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.scaler_path, "wb") as f:
                pickle.dump(scaler, f)
        return scaler

    def adapt(self, batch: InvarDayBatch) -> MaitBatch:
        """Adapt one InvarDayBatch into MaitBatch shapes.

        Args:
            batch: an InvarDayBatch from `InvarDataset.__iter__` or `.get`.

        Returns:
            MaitBatch with:
              x_panel          shape (N_t, 24, L=60), float32
              x_macro_lookback shape (17, L=60), float32 (already train-fold
                                z-scored by InvarDataset)
              regime_input     shape (5,), float32, query-day values of
                                [vix, vix_term_slope, slope_2s10s,
                                 hyg_5d_ret, market_breadth_proxy]
              y_cs, mask, day_index, date: passed through unchanged.
        """
        x_panel = batch.features.permute(0, 2, 1).index_select(
            dim=1, index=self.panel_keep_idx,
        )
        macro_kept = batch.macro.index_select(dim=1, index=self.macro_keep_idx)
        x_macro_lookback = macro_kept.transpose(0, 1).contiguous()
        regime_input = x_macro_lookback.index_select(
            dim=0, index=self.regime_idx_in_kept,
        )[:, -1]
        return MaitBatch(
            x_panel=x_panel,
            x_macro_lookback=x_macro_lookback,
            regime_input=regime_input,
            y_cs=batch.y_cs,
            mask=batch.mask,
            day_index=batch.day_index,
            date=batch.date,
        )

    def iter_days(self):
        """Yield MaitBatch over the dataset's eligible days."""
        for batch in self.dataset:
            yield self.adapt(batch)
