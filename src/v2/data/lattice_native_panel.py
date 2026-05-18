"""LATTICE-native panel adapter for the v2 RAG-STAR trainer (RAG-STAR
Universe v2).

Returns the LATTICE 26-column panel verbatim, no substitutions or
column-name aliasing. This is the cleanest interpretation of the
"same dataset and features as SWA-InVAR / MAiT" request because the
trainer ingests exactly the panel that InvarDataset consumes.

Output columns (in order, 26 panel features):

    PRICE_VOL (10): log_return, log_return_5d, log_return_20d,
                    log_volume, log_volume_ratio_20d,
                    realized_vol_20d, realized_vol_60d,
                    high_low_range, close_to_high_5d,
                    amihud_illiquidity_20d
    DISTRESS  (4):  interest_coverage, net_debt_to_ebitda,
                    fcf_yield, current_ratio
    INTANG    (4):  rd_to_sales, sga_to_sales, gross_profitability,
                    capex_to_sales
    OTHER_FUND(3):  log_market_cap, book_to_market, asset_growth_yoy
    CATALYST  (3):  days_to_next_catalyst_sin, _cos, catalyst_type_id
    FLAG      (2):  has_fundamentals, has_stocktwits

The v2 model's hardcoded DURATION_PANEL_COL_IDX = [14..21, 5,6,4,7, 9..13]
now picks LATTICE-native cols: 8 LATTICE fund-block cols + 4 price/vol
cols + 5 LATTICE distress cols. That is a semantically reasonable
duration-head input (fundamentals + risk/liquidity + distress), even
though the exact column identities differ from the original biotech v2.

The trainer config must set ``feature_dim: 26`` to size the input linear
projection correctly.

Target column ``fwd_return_h`` passes through directly from LATTICE.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.lattice.data.build_panel import PANEL_FEATURE_COLS


def _default_lattice_dir() -> Path:
    """Read LATTICE_PROCESSED_DIR env var so sibling panels
    (data/lattice_catalyst/processed, data/lattice_aug/processed, etc.)
    can override at runtime without editing config."""
    return Path(os.environ.get("LATTICE_PROCESSED_DIR",
                                "data/lattice/processed"))


@dataclass
class LatticeNativePanelConfig:
    lattice_dir: Path = field(default_factory=_default_lattice_dir)
    start_date: str = "2015-01-09"
    end_date: str = "2026-04-30"
    horizon_days: int = 5


def build_lattice_native_panel(
    cfg: LatticeNativePanelConfig | None = None,
) -> tuple[pd.DataFrame, list, list]:
    """Build the LATTICE-native 26-col panel for RAG-STAR Universe v2."""
    cfg = cfg or LatticeNativePanelConfig()

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

    # Stable 26-col ordering matching the on-disk Universal Ticker panel
    # schema. NOT `PANEL_FEATURE_COLS[:26]` because the upstream constant
    # was extended in May 2026 with 11 Tier A1+A2 features that shift the
    # slice. Hardcoded here so the v2 RAG-STAR Universe contract stays
    # stable independent of constant churn.
    target_cols = [
        "log_return", "log_return_5d", "log_return_20d",
        "log_volume", "log_volume_ratio_20d",
        "realized_vol_20d", "realized_vol_60d",
        "high_low_range", "close_to_high_5d", "amihud_illiquidity_20d",
        "interest_coverage", "net_debt_to_ebitda",
        "fcf_yield", "current_ratio",
        "rd_to_sales", "sga_to_sales", "gross_profitability", "capex_to_sales",
        "log_market_cap", "book_to_market", "asset_growth_yoy",
        "days_to_next_catalyst_sin", "days_to_next_catalyst_cos",
        "catalyst_type_id",
        "has_fundamentals", "has_stocktwits",
    ]
    assert len(target_cols) == 26
    missing = [c for c in target_cols if c not in panel.columns]
    if missing:
        print(
            f"[lattice_native_panel] WARNING: panel missing "
            f"{len(missing)} cols; zero-filled. Missing: {missing}",
            flush=True,
        )
    for c in target_cols:
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
        panel[["ticker", "date", "fwd_return_h"] + target_cols]
        .sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )
    dates = sorted(panel["date"].unique().tolist())
    tickers = sorted(panel["ticker"].unique().tolist())
    return panel, tickers, dates


# Module-level FEATURE_COLS that the v2 trainer's panel_to_tensors path
# can import in lieu of the biotech 22-col list. Hardcoded to the
# canonical on-disk LATTICE 26-col ordering so the v2 RAG-STAR Universe
# contract is stable independent of upstream PANEL_FEATURE_COLS churn.
FEATURE_COLS = [
    "log_return", "log_return_5d", "log_return_20d",
    "log_volume", "log_volume_ratio_20d",
    "realized_vol_20d", "realized_vol_60d",
    "high_low_range", "close_to_high_5d", "amihud_illiquidity_20d",
    "interest_coverage", "net_debt_to_ebitda",
    "fcf_yield", "current_ratio",
    "rd_to_sales", "sga_to_sales", "gross_profitability", "capex_to_sales",
    "log_market_cap", "book_to_market", "asset_growth_yoy",
    "days_to_next_catalyst_sin", "days_to_next_catalyst_cos",
    "catalyst_type_id",
    "has_fundamentals", "has_stocktwits",
]
assert len(FEATURE_COLS) == 26


__all__ = [
    "LatticeNativePanelConfig", "build_lattice_native_panel",
    "FEATURE_COLS",
]
