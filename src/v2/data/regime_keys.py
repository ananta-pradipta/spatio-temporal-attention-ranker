"""Regime keys for EFR-DGraph-epiSTAR.

Per spec Section 3. Builds the per-day 12-d regime feature vector
used by EFR memory retrieval and the FactorRepricingGate. All
features are train-fold standardised; same statistics applied to
val and test.

Extends the 4-d cs_struct from CSID v1 with 8 macro/sector features.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.v2.data.cs_struct import build_cs_struct


REGIME_FEATURE_COLS = [
    "pc1_share_21d",
    "avg_pairwise_corr_60d",
    "dispersion_5d",
    "market_return_5d",
    "xbi_ret_5d",
    "xbi_ret_20d",
    "xbi_rv_20d",
    "xbi_rv_60d",
    "delta_10y_20d",
    "delta_credit_spread_20d",
    "vix",
    "vxn",
]


def build_regime_keys(
    log_returns: np.ndarray,
    tradable_mask: np.ndarray,
    avg_pairwise_corr_60d: np.ndarray,
    cs_dispersion: np.ndarray,
    xbi_close: pd.Series,
    macro_arr: np.ndarray,
    macro_cols: list[str],
    panel_dates: list[pd.Timestamp],
    train_idx: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """Build the 12-d regime keys [T, R] tensor.

    The first 4 dims come from `build_cs_struct` (already train-fold
    z-scored). The remaining 8 come from the macro panel + a derived
    delta_credit_spread_20d (mapped to the existing
    `delta_hy_spread_20d` column from BAA10Y).
    """
    cs_struct, _ = build_cs_struct(
        log_returns=log_returns, tradable_mask=tradable_mask,
        avg_pairwise_corr_60d=avg_pairwise_corr_60d, cs_dispersion=cs_dispersion,
        xbi_close=xbi_close, panel_dates=panel_dates, train_idx=train_idx,
    )
    t_total = cs_struct.shape[0]
    out = np.zeros((t_total, len(REGIME_FEATURE_COLS)), dtype=np.float32)
    out[:, 0:4] = cs_struct
    macro_map = {
        "xbi_ret_5d": "xbi_ret_5d",
        "xbi_ret_20d": "xbi_ret_20d",
        "xbi_rv_20d": "xbi_rv_20d",
        "xbi_rv_60d": "xbi_rv_60d",
        "delta_10y_20d": "delta_10y_20d",
        "delta_credit_spread_20d": "delta_hy_spread_20d",  # BAA10Y proxy
        "vix": "vix",
        "vxn": "vxn",
    }
    for j, col in enumerate(REGIME_FEATURE_COLS[4:], start=4):
        macro_col = macro_map.get(col, col)
        if macro_col in macro_cols:
            out[:, j] = macro_arr[:, macro_cols.index(macro_col)]
        else:
            print(f"[regime_keys] WARN missing macro col: {macro_col}")
    return out.astype(np.float32), list(REGIME_FEATURE_COLS)


__all__ = ["REGIME_FEATURE_COLS", "build_regime_keys"]
