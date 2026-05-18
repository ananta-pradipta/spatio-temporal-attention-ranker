"""Universal macro state vector for the S&P 500 validation.

Parallel to ``src/v2/data/macro_duration_features.py`` with two
substitutions that preserve the 28-d shape:

    XBI return / vol slots  ->  XLK return / vol     (5 slots, IT sector ETF)
    IBB return slots        ->  XLF return           (2 slots, Financials sector ETF)

Output column names are kept identical to the biotech feed (xbi_ret_1d,
xbi_ret_5d, xbi_ret_20d, xbi_rv_20d, xbi_rv_60d, ibb_ret_5d, ibb_ret_20d)
so the existing model code reads transparently. Values are XLK/XLF.

Output: data/processed/macro_duration_features_sp500.parquet
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.v2.data.macro_duration_features import (
    MacroDurationConfig, MACRO_FEATURE_COLS_FULL, _load_or_fetch_fred_full,
)


@dataclass
class UniversalMacroDurationConfig(MacroDurationConfig):
    """Hyperparameters for universal macro-state extraction."""

    sector_etfs_parquet: Path = Path("data/raw/sp500/sector_etfs.parquet")
    output_path: Path = Path("data/processed/macro_duration_features_sp500.parquet")


def build_universal_macro_duration_features(
    cfg: UniversalMacroDurationConfig | None = None,
) -> pd.DataFrame:
    cfg = cfg or UniversalMacroDurationConfig()

    fred = _load_or_fetch_fred_full(cfg.panel_start, cfg.panel_end, cfg.fred_cache)
    risk = pd.read_parquet(cfg.risk_features_parquet)
    risk.index = pd.to_datetime(risk.index)

    etfs = pd.read_parquet(cfg.sector_etfs_parquet)
    etfs["date"] = pd.to_datetime(etfs["date"]).dt.normalize()
    by_etf: dict[str, pd.Series] = {
        tk: sub.set_index("date")["close"].sort_index()
        for tk, sub in etfs.groupby("ticker")
    }

    common = sorted(set(by_etf["XLK"].index) & set(by_etf["XLF"].index)
                    & set(by_etf["QQQ"].index) & set(by_etf["SPY"].index))
    dates = pd.DatetimeIndex(common)

    fred_a = fred.reindex(dates).ffill(limit=5)
    risk_a = risk.reindex(dates).ffill(limit=5)
    xlk_a  = by_etf["XLK"].reindex(dates).ffill(limit=5)
    xlf_a  = by_etf["XLF"].reindex(dates).ffill(limit=5)
    qqq_a  = by_etf["QQQ"].reindex(dates).ffill(limit=5)
    spy_a  = by_etf["SPY"].reindex(dates).ffill(limit=5)

    out = pd.DataFrame(index=dates)
    out.index.name = "date"

    out["dgs3mo"] = fred_a["DGS3MO"]
    out["dgs2"]   = fred_a["DGS2"]
    out["dgs10"]  = fred_a["DGS10"]
    out["delta_3mo_5d"]  = fred_a["DGS3MO"].diff(5)
    out["delta_3mo_20d"] = fred_a["DGS3MO"].diff(20)
    out["delta_2y_5d"]   = fred_a["DGS2"].diff(5)
    out["delta_2y_20d"]  = fred_a["DGS2"].diff(20)
    out["delta_10y_5d"]  = fred_a["DGS10"].diff(5)
    out["delta_10y_20d"] = fred_a["DGS10"].diff(20)
    out["term_10y_2y"]   = fred_a["DGS10"] - fred_a["DGS2"]
    out["term_10y_3m"]   = fred_a["DGS10"] - fred_a["DGS3MO"]
    out["hy_spread"]               = fred_a["BAA10Y"]
    out["delta_hy_spread_5d"]      = fred_a["BAA10Y"].diff(5)
    out["delta_hy_spread_20d"]     = fred_a["BAA10Y"].diff(20)
    out["vix"]  = risk_a["vix"]
    out["vxn"]  = risk_a["vxn"]
    out["vvix"] = risk_a["vvix"]

    # XBI slots populated from XLK (preserve column names)
    xlk_logret = np.log(xlk_a / xlk_a.shift(1))
    out["xbi_ret_1d"]  = xlk_logret
    out["xbi_ret_5d"]  = np.log(xlk_a / xlk_a.shift(5))
    out["xbi_ret_20d"] = np.log(xlk_a / xlk_a.shift(20))
    out["xbi_rv_20d"]  = xlk_logret.rolling(20).std()
    out["xbi_rv_60d"]  = xlk_logret.rolling(60).std()

    # IBB slots populated from XLF
    out["ibb_ret_5d"]  = np.log(xlf_a / xlf_a.shift(5))
    out["ibb_ret_20d"] = np.log(xlf_a / xlf_a.shift(20))

    out["qqq_ret_5d"]  = np.log(qqq_a / qqq_a.shift(5))
    out["qqq_ret_20d"] = np.log(qqq_a / qqq_a.shift(20))
    out["spy_ret_5d"]  = np.log(spy_a / spy_a.shift(5))
    out["spy_ret_20d"] = np.log(spy_a / spy_a.shift(20))

    # ---- F2-targeted regime/risk additions (pre-registered 2026-05-17).
    # MUST mirror src/v2/data/macro_duration_features.py exactly so the
    # universal (S&P 500 / lattice_native) panel carries the same 4
    # net-new features as the biotech panel. All trailing/contemporaneous
    # => point-in-time safe.
    out["curvature_butterfly"] = (
        2.0 * fred_a["DGS2"] - fred_a["DGS3MO"] - fred_a["DGS10"]
    )
    d10_1d = fred_a["DGS10"].diff(1)
    out["move_proxy"] = (
        d10_1d.rolling(21, min_periods=10).std() * np.sqrt(252.0)
    )
    out["vix_term_slope"] = risk_a["vix_term_slope"]
    spy_logret = np.log(spy_a / spy_a.shift(1))
    bond_ret = -7.0 * d10_1d / 100.0
    out["stockbond_corr60"] = (
        spy_logret.rolling(60, min_periods=30).corr(bond_ret)
    )

    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cfg.output_path)
    print(f"[universal_macro_duration] wrote {cfg.output_path}: "
          f"shape={out.shape}, range {out.index.min()} -> {out.index.max()}",
          flush=True)
    return out


__all__ = [
    "UniversalMacroDurationConfig",
    "build_universal_macro_duration_features",
]
