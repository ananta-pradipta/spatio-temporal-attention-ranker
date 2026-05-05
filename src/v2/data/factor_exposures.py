"""Factor exposures for FC-DGraph-epiSTAR.

Per spec Part A. Builds the per-(day, ticker, factor) exposure
tensor used by the FactorCalibrator. Each factor is standardised
CROSS-SECTIONALLY per day (across active tickers only) and clipped
to [-5, 5] to avoid outliers.

The cross-sectional z-score per day is what makes the factor scores
genuinely cross-sectional and prevents day-level constants from
contaminating the calibration.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.v2.data.rolling_macro_betas import ROLLING_BETA_COLS


# Spec Part A primary factor list (14 factors).
PRIMARY_FACTOR_COLS = [
    "realized_vol_20d",
    "realized_vol_60d",
    "log_market_cap",
    "cash_runway_q",
    "cash_to_mc",
    "rd_intensity",
    "log_volume_ratio_20d",
    "st_volume_24h",
    "st_bullish_ratio",
    "st_labeled_ratio",
    "age_trading_days",
    "rolling_xbi_beta_60d",
    "rolling_rate_beta_60d",
    "rolling_credit_beta_60d",
]

# Mapping each factor to its data source.
#   "panel:N" = panel feature index N (22-feature schema)
#   "beta:NAME" = column name in ROLLING_BETA_COLS
#   "age" = age_feat[..., 0]
PANEL_FEATURE_INDEX = {
    "log_return": 0, "log_return_5d": 1, "log_return_20d": 2,
    "log_volume": 3, "log_volume_ratio_20d": 4,
    "realized_vol_20d": 5, "realized_vol_60d": 6,
    "high_low_range": 7, "close_to_high": 8,
    "st_volume_24h": 9, "st_volume_change_30d": 10,
    "st_bullish_ratio": 11, "st_sentiment_dispersion": 12, "st_labeled_ratio": 13,
    "log_market_cap": 14, "cash_runway_q": 15, "rd_intensity": 16,
    "revenue_growth_yoy": 17, "cash_to_mc": 18,
    "shares_outstanding_yoy": 19, "total_assets_growth": 20,
    "has_fundamentals": 21,
}


def build_factor_exposures(
    x_raw: np.ndarray,
    age_feat: np.ndarray,
    betas_tensor: np.ndarray,
    tradable_mask: np.ndarray,
    factor_cols: list[str] | None = None,
    clip_z: float = 5.0,
) -> tuple[np.ndarray, list[str]]:
    """Build [T, N, K] cross-sectionally standardised factor exposures.

    Args:
        x_raw: [T, N, 22] raw panel features.
        age_feat: [T, N, 8] age feature tensor (from DOW v2.3 build).
        betas_tensor: [T, N, 10] rolling beta tensor
            (`src.v2.data.rolling_macro_betas.betas_to_tensor`).
        tradable_mask: [T, N] bool active mask.
        factor_cols: which factor columns to include; defaults to
            PRIMARY_FACTOR_COLS.
        clip_z: clip standardised exposures to [-clip_z, clip_z].

    Returns:
        exposures: [T, N, K] float32 cross-sectionally z-scored
            factor exposures. Inactive cells are filled with 0.
        cols: list of factor names in order.
    """
    cols = list(factor_cols) if factor_cols is not None else list(PRIMARY_FACTOR_COLS)
    t_total, n, _ = x_raw.shape
    raw = np.zeros((t_total, n, len(cols)), dtype=np.float32)
    for k, col in enumerate(cols):
        if col in PANEL_FEATURE_INDEX:
            raw[..., k] = x_raw[..., PANEL_FEATURE_INDEX[col]]
        elif col in ROLLING_BETA_COLS:
            raw[..., k] = betas_tensor[..., ROLLING_BETA_COLS.index(col)]
        elif col == "age_trading_days":
            raw[..., k] = age_feat[..., 0]
        elif col == "history_valid_ratio_60d":
            raw[..., k] = age_feat[..., 7]
        else:
            raise ValueError(f"unknown factor column: {col}")

    # Cross-sectional standardisation per day (over active tickers only).
    out = np.zeros_like(raw)
    for t in range(t_total):
        m = tradable_mask[t]
        if m.sum() < 5:
            continue
        for k in range(len(cols)):
            vals = raw[t, m, k]
            mu = float(np.mean(vals))
            sd = float(np.std(vals))
            if sd < 1e-6:
                z = vals - mu
            else:
                z = (vals - mu) / sd
            z = np.clip(z, -clip_z, clip_z)
            out[t, m, k] = z
    return out, cols


__all__ = [
    "PRIMARY_FACTOR_COLS",
    "PANEL_FEATURE_INDEX",
    "build_factor_exposures",
]
