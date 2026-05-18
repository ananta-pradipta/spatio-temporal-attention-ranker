"""LATTICE-native F2-feature panel adapter for the v2 trainer.

Reads `data/lattice_f2feats/processed/panel_features.parquet` which has the
canonical 26 cols plus 3 F2-targeted features appended:

  26 canonical cols (price/volume, distress, intangibles, other fund,
                     catalyst, flag) ordered identical to lattice_native_panel
  +  rate_beta_60d
  +  delta_rv_20d
  +  low_vol_decile_flag

Output panel shape: (T, N, F=29). The v2 RAG-STAR architecture's input
projection grows from 26 to 29 dims via the standard linear layer; no
other code changes required. F2-specific predictive signal documented in
the rank-IC analysis on 2026-05-13.

Selected via `panel_kind: lattice_native_f2` in the trainer config. Does
not interfere with the canonical lattice_native code path.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


def _default_lattice_dir() -> Path:
    return Path(os.environ.get("LATTICE_PROCESSED_DIR",
                                "data/lattice_f2feats/processed"))


@dataclass
class LatticeNativeF2PanelConfig:
    lattice_dir: Path = field(default_factory=_default_lattice_dir)
    start_date: str = "2015-01-09"
    end_date: str = "2026-04-30"
    horizon_days: int = 5


# 29-col schema: canonical 26 + 3 F2-targeted additions. Order is locked.
FEATURE_COLS = [
    # 10 PRICE_VOL
    "log_return", "log_return_5d", "log_return_20d",
    "log_volume", "log_volume_ratio_20d",
    "realized_vol_20d", "realized_vol_60d",
    "high_low_range", "close_to_high_5d", "amihud_illiquidity_20d",
    # 4 DISTRESS
    "interest_coverage", "net_debt_to_ebitda",
    "fcf_yield", "current_ratio",
    # 4 INTANGIBLES
    "rd_to_sales", "sga_to_sales", "gross_profitability", "capex_to_sales",
    # 3 OTHER FUND
    "log_market_cap", "book_to_market", "asset_growth_yoy",
    # 3 CATALYST
    "days_to_next_catalyst_sin", "days_to_next_catalyst_cos",
    "catalyst_type_id",
    # 2 FLAG
    "has_fundamentals", "has_stocktwits",
    # 3 F2-targeted additions (2026-05-13)
    "rate_beta_60d", "delta_rv_20d", "low_vol_decile_flag",
]
assert len(FEATURE_COLS) == 29


def build_lattice_native_f2_panel(
    cfg: LatticeNativeF2PanelConfig | None = None,
) -> tuple[pd.DataFrame, list, list]:
    """Build the 29-col F2-feature panel for RAG-STAR Universe."""
    cfg = cfg or LatticeNativeF2PanelConfig()

    panel_path = cfg.lattice_dir / "panel_features.parquet"
    if not panel_path.exists():
        raise FileNotFoundError(f"F2-feature panel not found: {panel_path}")
    panel = pd.read_parquet(panel_path)
    panel["date"] = pd.to_datetime(panel["date"])
    panel["ticker"] = panel["ticker"].astype(str).str.upper()
    panel = panel[
        (panel["date"] >= pd.Timestamp(cfg.start_date))
        & (panel["date"] <= pd.Timestamp(cfg.end_date))
    ].copy()

    missing = [c for c in FEATURE_COLS if c not in panel.columns]
    if missing:
        print(
            f"[lattice_native_f2_panel] WARNING: panel missing "
            f"{len(missing)} cols; zero-filled. Missing: {missing}",
            flush=True,
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


__all__ = [
    "LatticeNativeF2PanelConfig",
    "build_lattice_native_f2_panel",
    "FEATURE_COLS",
]
