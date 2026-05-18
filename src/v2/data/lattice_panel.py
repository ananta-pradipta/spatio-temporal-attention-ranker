"""LATTICE panel adapter for the v2 RAG-STAR trainer.

Produces a v2-schema panel DataFrame (22 FEATURE_COLS) from LATTICE
artifacts so the RAG-STAR architecture can train and evaluate on the
same dataset that SWA-InVAR and MAiT consume.

Inputs:
    data/lattice/processed/panel_features.parquet  (26 LATTICE panel cols)
    data/processed/stocktwits_features_sp500.parquet (optional, 2014-2022 cap)

Outputs:
    pandas DataFrame with FEATURE_COLS (22) + ticker + date + fwd_return_h
    tickers list, dates list

Substitutions vs the v2 biotech panel (matches the spirit of the existing
``src/v2/data/universal_panel.py`` mappings):

    Position 9-13 (ST_COLS): joined from stocktwits_features_sp500 if
        available; zero-filled otherwise. v2 ST col names are mapped to
        LATTICE ST col names where possible.

    Position 14-20 (FUND_COLS):
        cash_runway_q          <- interest_coverage  (distress proxy)
        rd_intensity           <- rd_to_sales
        revenue_growth_yoy     <- asset_growth_yoy   (growth proxy)
        cash_to_mc             <- book_to_market     (liquidity-to-market)
        shares_outstanding_yoy <- capex_to_sales     (dilution proxy)
        total_assets_growth    <- asset_growth_yoy
        log_market_cap         <- log_market_cap     (direct)

    Position 21 (FLAG_COLS):
        has_fundamentals       <- has_fundamentals   (direct)

PRICE_COLS positions 0-8 are direct since LATTICE inherited the same
price-feature definitions; only close_to_high_5d is renamed to
close_to_high.

The fwd_return_h target column carries through directly from LATTICE
(both pipelines use np.log(close.shift(-5) / close)).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.mtgn.training.panel_enriched import (
    PRICE_COLS, ST_COLS, FUND_COLS, FLAG_COLS, FEATURE_COLS,
)


@dataclass
class LatticePanelConfig:
    """LATTICE-to-v2-panel adapter config."""

    lattice_dir: Path = Path("data/lattice/processed")
    stocktwits_features_parquet: Path = Path(
        "data/processed/stocktwits_features_sp500.parquet",
    )
    start_date: str = "2015-01-09"
    end_date: str = "2025-12-31"
    horizon_days: int = 5


# v2 -> LATTICE col-name mappings used during the build.
_PRICE_RENAME = {"close_to_high_5d": "close_to_high"}

_FUND_PROXY = {
    "cash_runway_q":          "interest_coverage",
    "rd_intensity":           "rd_to_sales",
    "revenue_growth_yoy":     "asset_growth_yoy",
    "cash_to_mc":             "book_to_market",
    "shares_outstanding_yoy": "capex_to_sales",
    "total_assets_growth":    "asset_growth_yoy",
    "log_market_cap":         "log_market_cap",
}

# LATTICE ST col names from src/lattice/data/build_panel.py ST_FEATURE_COLS:
#   st_volume_24h_log, st_volume_abnormal_z60d, st_sentiment_dispersion,
#   st_labeled_ratio, st_bullish_ratio_demeaned.
_ST_MAP = {
    "st_volume_24h":          "st_volume_24h_log",
    "st_volume_change_30d":   "st_volume_abnormal_z60d",
    "st_bullish_ratio":       "st_bullish_ratio_demeaned",
    "st_sentiment_dispersion":"st_sentiment_dispersion",
    "st_labeled_ratio":       "st_labeled_ratio",
}


def build_lattice_panel(
    cfg: LatticePanelConfig | None = None,
) -> tuple[pd.DataFrame, list, list]:
    """Build a v2-schema panel from LATTICE inputs.

    Returns:
        panel: pandas DataFrame with FEATURE_COLS (22) + 'ticker' + 'date'
               + 'fwd_return_h'.
        tickers: sorted unique ticker list.
        dates: sorted unique date list.
    """
    cfg = cfg or LatticePanelConfig()

    panel_path = cfg.lattice_dir / "panel_features.parquet"
    if not panel_path.exists():
        raise FileNotFoundError(f"LATTICE panel not found: {panel_path}")
    panel = pd.read_parquet(panel_path)
    panel["date"] = pd.to_datetime(panel["date"])
    panel["ticker"] = panel["ticker"].astype(str).str.upper()
    panel = panel[
        (panel["date"] >= pd.Timestamp(cfg.start_date))
        & (panel["date"] <= pd.Timestamp(cfg.end_date))
    ].copy()

    panel = panel.rename(columns=_PRICE_RENAME)

    for v2_name, lattice_name in _FUND_PROXY.items():
        if lattice_name in panel.columns:
            panel[v2_name] = panel[lattice_name]
        else:
            panel[v2_name] = 0.0

    for c in ST_COLS:
        panel[c] = 0.0
    if cfg.stocktwits_features_parquet.exists():
        st = pd.read_parquet(cfg.stocktwits_features_parquet)
        st["date"] = pd.to_datetime(st["date"])
        st["ticker"] = st["ticker"].astype(str).str.upper()
        # The SP500 StockTwits parquet was generated with v2-schema column
        # names directly (st_volume_24h, st_bullish_ratio, etc.). Try those
        # first; fall back to the LATTICE-name aliases in _ST_MAP if a v2
        # name is absent.
        st_cols_direct = [c for c in ST_COLS if c in st.columns]
        st_cols_alias = [
            c for c in _ST_MAP.values()
            if c not in ST_COLS and c in st.columns
        ]
        merge_cols = sorted(set(st_cols_direct + st_cols_alias))
        if merge_cols:
            st_subset = st[["ticker", "date"] + merge_cols].copy()
            panel = panel.merge(
                st_subset, how="left", on=["ticker", "date"],
                suffixes=("", "_st"),
            )
            populated = 0
            for v2c in ST_COLS:
                if v2c in panel.columns and v2c in st_cols_direct:
                    src_col = v2c + "_st" if v2c + "_st" in panel.columns else v2c
                    panel[v2c] = pd.to_numeric(
                        panel[src_col], errors="coerce",
                    ).fillna(0.0)
                    populated += 1
                else:
                    alias = _ST_MAP.get(v2c)
                    if alias and alias in panel.columns:
                        panel[v2c] = pd.to_numeric(
                            panel[alias], errors="coerce",
                        ).fillna(0.0)
                        populated += 1
            print(
                f"[lattice_panel] stocktwits joined: "
                f"{populated}/{len(ST_COLS)} cols populated "
                f"(direct={len(st_cols_direct)}, alias={len(st_cols_alias)})",
            )
    else:
        print(
            f"[lattice_panel] stocktwits parquet missing at "
            f"{cfg.stocktwits_features_parquet}; ST cols zero-filled",
        )

    if "has_fundamentals" not in panel.columns:
        panel["has_fundamentals"] = 0.0

    # Ablation B (2026-05-12): if LATTICE_PANEL_ZERO_ST=1, force all 5
    # StockTwits cols to zero AFTER the join. Tests whether retail-sentiment
    # explains RAG-STAR Universe v1's F2 lift relative to v2.
    if os.environ.get("LATTICE_PANEL_ZERO_ST", "0") == "1":
        for c in ST_COLS:
            panel[c] = 0.0
        print(
            "[lattice_panel] ABLATION B: ST cols force-zeroed "
            "(LATTICE_PANEL_ZERO_ST=1)",
        )

    for c in FEATURE_COLS:
        if c not in panel.columns:
            panel[c] = 0.0
        panel[c] = (
            pd.to_numeric(panel[c], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )

    panel["fwd_return_h"] = (
        pd.to_numeric(panel["fwd_return_h"], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )
    panel = panel.dropna(subset=["fwd_return_h"]).reset_index(drop=True)

    panel = (
        panel[["ticker", "date", "fwd_return_h"] + FEATURE_COLS]
        .sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )
    dates = sorted(panel["date"].unique().tolist())
    tickers = sorted(panel["ticker"].unique().tolist())
    return panel, tickers, dates


__all__ = ["LatticePanelConfig", "build_lattice_panel"]
