"""Build extended (2015-2026) versions of the v2 trainer's auxiliary
parquets so the RAG-STAR Universe sweep can run on LATTICE folds F4 and F5.

Outputs (next to the canonical ones, with `_ext` suffix):
    data/processed/risk_features_sp500_ext.parquet
    data/processed/macro_duration_features_sp500_ext.parquet
    data/processed/sp500_rolling_betas_ext.parquet

Uses the extended raw inputs that LATTICE already maintains:
    data/lattice/raw/macro_etfs.parquet         (-> sector_etfs.parquet extended)
    data/lattice/raw/macro_etfs_extra.parquet
    data/lattice/raw/prices_sp500.parquet        (-> prices_sp500_extended.parquet)
    data/lattice/raw/sp500_constituents_pit.parquet (extended)
    data/lattice/raw/macro_fred*.csv / cache

Run with: python -m scripts.lattice.build_v2_aux_extended
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.v2.data.macro_duration_features import (
    MACRO_FEATURE_COLS_FULL, _load_or_fetch_fred_full,
)
from src.v2.data.universal_macro_features import (
    UniversalMacroDurationConfig, build_universal_macro_duration_features,
)
from src.v2.data.universal_rolling_betas import (
    UniversalRollingBetaConfig, build_universal_rolling_betas,
)


EXT_PANEL_START = "2014-09-01"
EXT_PANEL_END = "2026-04-30"

OUT_DIR = Path("data/processed")
SECTOR_ETFS = Path("data/raw/sp500/sector_etfs.parquet")
PRICES_EXTENDED = Path("data/raw/sp500/prices_sp500_extended.parquet")
CONSTITUENTS_EXTENDED = Path(
    "data/raw/sp500/sp500_constituents_history_extended.parquet",
)


def build_extended_risk_features() -> None:
    """Extended risk_features_sp500_ext.parquet (2015-2026).

    Mirrors scripts/build_universal_risk_features.py but builds the
    biotech base from extended FRED + sector_etfs first (rather than
    reading the static 2015-2022 base parquet).
    """
    print("[ext aux] building risk_features_sp500_ext", flush=True)
    fred = _load_or_fetch_fred_full(
        EXT_PANEL_START, EXT_PANEL_END,
        Path("data/raw/macro_fred_full.csv"),
    )
    if "DATE" in fred.columns:
        fred = fred.rename(columns={"DATE": "date"})
    if "date" in fred.columns:
        fred = fred.set_index("date")
    fred.index = pd.to_datetime(fred.index)
    fred = fred.sort_index()

    etfs = pd.read_parquet(SECTOR_ETFS)
    etfs["date"] = pd.to_datetime(etfs["date"]).dt.normalize()
    xlk = etfs[etfs.ticker == "XLK"].set_index("date")["close"].sort_index()

    idx = pd.date_range(EXT_PANEL_START, EXT_PANEL_END, freq="B")
    xlk_aligned = xlk.reindex(idx).ffill(limit=5)
    vix = fred["VIXCLS"].reindex(idx).ffill() if "VIXCLS" in fred.columns else None
    vxn = fred["VXNCLS"].reindex(idx).ffill() if "VXNCLS" in fred.columns else None
    vvix = fred["VVIXCLS"].reindex(idx).ffill() if "VVIXCLS" in fred.columns else None

    logret_1d = np.log(xlk_aligned / xlk_aligned.shift(1))
    out = pd.DataFrame(index=idx)
    out["vix"] = vix
    out["vxn"] = vxn
    out["vvix"] = vvix
    if vix is not None:
        out["vix_term_slope"] = vvix - vix if vvix is not None else 0.0
        out["vix_5d_change"] = vix.diff(5)
    out["xbi_rv_20d"] = logret_1d.rolling(20).std()   # XLK substitution
    out["xbi_rv_60d"] = logret_1d.rolling(60).std()
    fwd_5d = np.log(xlk_aligned.shift(-5) / xlk_aligned).abs()
    out["xbi_fwd_abs_ret_5d"] = fwd_5d
    out = out[[
        "vix", "vxn", "vvix", "vix_term_slope",
        "xbi_rv_20d", "xbi_rv_60d", "vix_5d_change", "xbi_fwd_abs_ret_5d",
    ]]
    out_path = OUT_DIR / "risk_features_sp500_ext.parquet"
    out.to_parquet(out_path)
    print(f"  wrote {out_path}: shape={out.shape} "
          f"date range {out.index.min()} -> {out.index.max()}", flush=True)


def build_extended_macro_duration() -> None:
    print("[ext aux] building macro_duration_features_sp500_ext", flush=True)
    cfg = UniversalMacroDurationConfig(
        panel_start=EXT_PANEL_START,
        panel_end=EXT_PANEL_END,
        fred_cache=Path("data/raw/macro_fred_full.csv"),
        sector_etfs_parquet=SECTOR_ETFS,
        risk_features_parquet=OUT_DIR / "risk_features_sp500_ext.parquet",
        output_path=OUT_DIR / "macro_duration_features_sp500_ext.parquet",
    )
    df = build_universal_macro_duration_features(cfg)
    print(f"  wrote {cfg.output_path}: shape={df.shape}", flush=True)


def build_extended_rolling_betas() -> None:
    print("[ext aux] building sp500_rolling_betas_ext", flush=True)
    cfg = UniversalRollingBetaConfig(
        panel_start=EXT_PANEL_START,
        panel_end=EXT_PANEL_END,
        raw_prices_parquet=PRICES_EXTENDED,
        sector_etfs_parquet=SECTOR_ETFS,
        constituents_parquet=CONSTITUENTS_EXTENDED,
        macro_duration_parquet=OUT_DIR / "macro_duration_features_sp500_ext.parquet",
        output_path=OUT_DIR / "sp500_rolling_betas_ext.parquet",
    )
    df = build_universal_rolling_betas(cfg)
    print(f"  wrote {cfg.output_path}: shape={df.shape}", flush=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    build_extended_risk_features()
    build_extended_macro_duration()
    build_extended_rolling_betas()
    print("[ext aux] DONE", flush=True)


if __name__ == "__main__":
    main()
