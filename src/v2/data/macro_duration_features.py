"""Macro-duration feature builder for DOW-epiSTAR v2.

Extends the SBP v1 macro features to the full DOW v2 feature set
defined in spec Section A. Pulls:

    Treasury yields:    DGS3MO, DGS2, DGS10           (FRED)
    Credit spread:      BAA10Y (proxy for BAMLH0A0HYM2 HY OAS, same
                        credit-stress signal, full free historical
                        access via FRED graph CSV)
    Volatility:         VIX, VXN, VVIX                (already in
                        data/processed/risk_features.parquet)
    Sector ETF:         XBI, IBB                       (yfinance)
    Broad market ETF:   QQQ, SPY                       (yfinance)

Computed features per day:

    yields and yield deltas:
        dgs3mo, dgs2, dgs10
        delta_3mo_5d, delta_3mo_20d
        delta_2y_5d, delta_2y_20d
        delta_10y_5d, delta_10y_20d
        term_10y_2y, term_10y_3m
    credit:
        hy_spread, delta_hy_spread_5d, delta_hy_spread_20d
    volatility z-scores:
        vix_z, vxn_z, vvix_z
    ETF returns and realized vols:
        xbi_ret_1d, xbi_ret_5d, xbi_ret_20d
        xbi_rv_20d, xbi_rv_60d
        ibb_ret_5d, ibb_ret_20d
        qqq_ret_5d, qqq_ret_20d
        spy_ret_5d, spy_ret_20d
    cross-sectional diagnostics (read from episode_keys helper):
        avg_pairwise_corr_60d, cross_sectional_dispersion,
        cross_sectional_skew, cross_sectional_kurtosis,
        active_count_norm

Output:
    data/processed/macro_duration_features.parquet
        Columns include both raw and z-scored variants (suffix _z) for
        the standardisable features. The z-scoring is applied at
        consume time using train-fold statistics; here we persist raw
        levels.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from src.v2.data.macro_features import _load_or_fetch_etf, _load_or_fetch_fred


@dataclass
class MacroDurationConfig:
    """Hyperparameters for macro-duration feature extraction."""

    panel_start: str = "2014-09-01"   # 60d warmup
    panel_end: str = "2023-01-15"
    output_path: Path = Path("data/processed/macro_duration_features.parquet")
    fred_cache: Path = Path("data/raw/macro_fred_full.csv")
    xbi_cache: Path = Path("data/raw/xbi_close.csv")
    ibb_cache: Path = Path("data/raw/ibb_close.csv")
    qqq_cache: Path = Path("data/raw/qqq_close.csv")
    spy_cache: Path = Path("data/raw/spy_close.csv")
    risk_features_parquet: Path = Path("data/processed/risk_features.parquet")


def _load_or_fetch_fred_full(start: str, end: str, cache: Path) -> pd.DataFrame:
    """Load full FRED set: DGS3MO, DGS2, DGS10, BAA10Y."""
    series = ["DGS3MO", "DGS2", "DGS10", "BAA10Y"]
    if cache.exists():
        df = pd.read_csv(cache, parse_dates=["date"]).set_index("date")
        if all(s in df.columns for s in series):
            if df.index.min() <= pd.Timestamp(start) and df.index.max() >= pd.Timestamp(end):
                return df
    from pandas_datareader import data as web
    parts = []
    for s in series:
        try:
            x = web.DataReader(s, "fred", start, end)
        except Exception as exc:
            warnings.warn(f"FRED fetch failed for {s}: {exc}", stacklevel=2)
            continue
        x.columns = [s]
        parts.append(x)
    if not parts:
        raise RuntimeError("FRED fetch failed for all series")
    out = pd.concat(parts, axis=1).sort_index()
    out.index.name = "date"
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache)
    return out


def build_macro_duration_features(
    cfg: MacroDurationConfig | None = None,
) -> pd.DataFrame:
    """Build the full DOW v2 macro-duration feature table.

    Persists to ``cfg.output_path`` and returns the wide-format
    DataFrame indexed by date.
    """
    cfg = cfg or MacroDurationConfig()

    fred = _load_or_fetch_fred_full(cfg.panel_start, cfg.panel_end, cfg.fred_cache)
    xbi = _load_or_fetch_etf("XBI", cfg.panel_start, cfg.panel_end, cfg.xbi_cache)
    ibb = _load_or_fetch_etf("IBB", cfg.panel_start, cfg.panel_end, cfg.ibb_cache)
    qqq = _load_or_fetch_etf("QQQ", cfg.panel_start, cfg.panel_end, cfg.qqq_cache)
    spy = _load_or_fetch_etf("SPY", cfg.panel_start, cfg.panel_end, cfg.spy_cache)

    risk = pd.read_parquet(cfg.risk_features_parquet)
    risk = risk.copy()
    risk.index = pd.to_datetime(risk.index)

    # Common date index from XBI (most restrictive, ETF trading days).
    dates = pd.DatetimeIndex(sorted(set(xbi.index) & set(ibb.index) & set(qqq.index) & set(spy.index)))
    fred_aligned = fred.reindex(dates).ffill(limit=5)
    xbi_a = xbi.reindex(dates).ffill(limit=5)
    ibb_a = ibb.reindex(dates).ffill(limit=5)
    qqq_a = qqq.reindex(dates).ffill(limit=5)
    spy_a = spy.reindex(dates).ffill(limit=5)
    risk_a = risk.reindex(dates).ffill(limit=5)

    out = pd.DataFrame(index=dates)
    out.index.name = "date"

    # Yields and deltas.
    out["dgs3mo"] = fred_aligned["DGS3MO"]
    out["dgs2"] = fred_aligned["DGS2"]
    out["dgs10"] = fred_aligned["DGS10"]
    out["delta_3mo_5d"] = fred_aligned["DGS3MO"].diff(5)
    out["delta_3mo_20d"] = fred_aligned["DGS3MO"].diff(20)
    out["delta_2y_5d"] = fred_aligned["DGS2"].diff(5)
    out["delta_2y_20d"] = fred_aligned["DGS2"].diff(20)
    out["delta_10y_5d"] = fred_aligned["DGS10"].diff(5)
    out["delta_10y_20d"] = fred_aligned["DGS10"].diff(20)
    out["term_10y_2y"] = fred_aligned["DGS10"] - fred_aligned["DGS2"]
    out["term_10y_3m"] = fred_aligned["DGS10"] - fred_aligned["DGS3MO"]

    # Credit spread.
    out["hy_spread"] = fred_aligned["BAA10Y"]
    out["delta_hy_spread_5d"] = fred_aligned["BAA10Y"].diff(5)
    out["delta_hy_spread_20d"] = fred_aligned["BAA10Y"].diff(20)

    # Volatility.
    out["vix"] = risk_a["vix"]
    out["vxn"] = risk_a["vxn"]
    out["vvix"] = risk_a["vvix"]

    # ETF returns (log).
    xbi_logret = np.log(xbi_a / xbi_a.shift(1))
    ibb_logret = np.log(ibb_a / ibb_a.shift(1))
    qqq_logret = np.log(qqq_a / qqq_a.shift(1))
    spy_logret = np.log(spy_a / spy_a.shift(1))
    out["xbi_ret_1d"] = xbi_logret
    out["xbi_ret_5d"] = np.log(xbi_a / xbi_a.shift(5))
    out["xbi_ret_20d"] = np.log(xbi_a / xbi_a.shift(20))
    out["xbi_rv_20d"] = xbi_logret.rolling(20).std()
    out["xbi_rv_60d"] = xbi_logret.rolling(60).std()
    out["ibb_ret_5d"] = np.log(ibb_a / ibb_a.shift(5))
    out["ibb_ret_20d"] = np.log(ibb_a / ibb_a.shift(20))
    out["qqq_ret_5d"] = np.log(qqq_a / qqq_a.shift(5))
    out["qqq_ret_20d"] = np.log(qqq_a / qqq_a.shift(20))
    out["spy_ret_5d"] = np.log(spy_a / spy_a.shift(5))
    out["spy_ret_20d"] = np.log(spy_a / spy_a.shift(20))

    # ---- F2-targeted regime/risk additions (pre-registered 2026-05-17).
    # All trailing-window / contemporaneous => point-in-time safe; no
    # forward information. Pre-registration and rationale: F2 (2021-22
    # rate-rotation) is the binding negative fold for every model.

    # (a-new) Yield-curve curvature (butterfly). Rate velocity and 2s10s/
    # 3m10y slope are already covered by delta_* and term_* above; only
    # curvature is net-new. Positive = humped curve, negative = bowed.
    out["curvature_butterfly"] = (
        2.0 * fred_aligned["DGS2"]
        - fred_aligned["DGS3MO"]
        - fred_aligned["DGS10"]
    )

    # (b) MOVE proxy: annualized 21d rolling std of daily 10y-yield
    # changes (Treasury-rate realized vol). Legacy InVAR move_proxy
    # definition; fully FRED-derived, trailing window.
    d10_1d = fred_aligned["DGS10"].diff(1)
    out["move_proxy"] = (
        d10_1d.rolling(21, min_periods=10).std() * np.sqrt(252.0)
    )

    # (d) VIX term-structure: VIX3M - VIX (positive = contango/calm).
    # Reuse the vetted construct already persisted in risk_features.
    out["vix_term_slope"] = risk_a["vix_term_slope"]

    # (c) Rolling 60d stock-bond correlation (signed; sign encodes
    # regime). Bond return proxied from DGS10 via -D*dy (D~=7), dy in
    # decimal. Negative = flight-to-quality (risk-off); positive =
    # joint stock+bond drawdown (the 2022 rate-shock regime, i.e. F2).
    bond_ret = -7.0 * d10_1d / 100.0
    out["stockbond_corr60"] = (
        spy_logret.rolling(60, min_periods=30).corr(bond_ret)
    )

    # Persist.
    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cfg.output_path)
    print(f"[macro_duration] wrote {cfg.output_path}: shape={out.shape}, "
          f"date range {out.index.min()} to {out.index.max()}")
    return out


MACRO_FEATURE_COLS_FULL = [
    "dgs3mo", "dgs2", "dgs10",
    "delta_3mo_5d", "delta_3mo_20d",
    "delta_2y_5d", "delta_2y_20d",
    "delta_10y_5d", "delta_10y_20d",
    "term_10y_2y", "term_10y_3m",
    "hy_spread", "delta_hy_spread_5d", "delta_hy_spread_20d",
    "vix", "vxn", "vvix",
    "xbi_ret_1d", "xbi_ret_5d", "xbi_ret_20d",
    "xbi_rv_20d", "xbi_rv_60d",
    "ibb_ret_5d", "ibb_ret_20d",
    "qqq_ret_5d", "qqq_ret_20d",
    "spy_ret_5d", "spy_ret_20d",
    # F2-targeted regime/risk additions (pre-registered 2026-05-17).
    "curvature_butterfly", "move_proxy",
    "vix_term_slope", "stockbond_corr60",
]

# Subset used for the macro_gate that drives lambda_macro (Section E
# of the spec).
MACRO_GATE_COLS = [
    "delta_10y_5d", "delta_10y_20d",
    "delta_hy_spread_20d",
    "vix", "vxn",
    "xbi_rv_20d", "xbi_ret_20d",
]


def standardize_macro_duration(
    macro: pd.DataFrame, panel_dates: list[pd.Timestamp],
    train_idx: np.ndarray,
) -> tuple[np.ndarray, list[str], dict]:
    """Align ``macro`` to ``panel_dates``, z-score using train stats.

    Returns:
        macro_arr: [T, F] float32 standardised macro features (aligned).
        cols: list of feature column names.
        stats: dict with train-fold mean and std per column for audit.
    """
    panel_index = pd.DatetimeIndex(pd.to_datetime(panel_dates).normalize())
    macro = macro.copy()
    macro.index = pd.to_datetime(macro.index).normalize()
    macro_aligned = macro.reindex(panel_index).ffill(limit=5)
    cols = [c for c in MACRO_FEATURE_COLS_FULL if c in macro_aligned.columns]
    arr = macro_aligned[cols].to_numpy(dtype=np.float32)
    arr = np.where(np.isnan(arr), 0.0, arr)
    train_arr = arr[train_idx]
    mu = train_arr.mean(axis=0)
    sd = train_arr.std(axis=0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    arr_z = ((arr - mu) / sd).astype(np.float32)
    return arr_z, cols, {"mean": mu.tolist(), "std": sd.tolist()}


__all__ = [
    "MacroDurationConfig",
    "MACRO_FEATURE_COLS_FULL",
    "MACRO_GATE_COLS",
    "build_macro_duration_features",
    "standardize_macro_duration",
]
