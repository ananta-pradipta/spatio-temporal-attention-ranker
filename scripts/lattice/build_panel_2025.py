"""Rebuild the processed panel and macro state with extended 2015-01 to 2025-12 coverage.

Reads the EXTENDED raw symlinks (prices_sp500.parquet ->
prices_sp500_extended.parquet, sp500_constituents_pit.parquet ->
sp500_constituents_history_extended.parquet) and rebuilds:
  data/lattice/processed/panel_features.parquet
  data/lattice/processed/cohorts.parquet
  data/lattice/processed/active_mask.parquet
  data/lattice/processed/macro_state.parquet

per the 2026-05-11 Scenario A dataset extension. Calls the existing
build_phase1 (panel features + cohorts + active mask) and
build_macro_state (24-feature macro state) entry points with
panel_end="2025-12-31".
"""
from __future__ import annotations

from pathlib import Path

from src.lattice.data.build_panel import LatticePhase1Config, build_phase1
from src.lattice.data.build_macro import build_macro_state


def main() -> None:
    print("[build_panel_2025] starting panel rebuild", flush=True)
    cfg = LatticePhase1Config(
        raw_dir=Path("data/lattice/raw"),
        out_dir=Path("data/lattice/processed"),
        panel_start="2015-01-09",
        panel_end="2025-12-31",
    )
    summary = build_phase1(cfg)
    print(f"[build_panel_2025] phase1 summary: {summary}", flush=True)

    print("[build_panel_2025] building macro state", flush=True)
    build_macro_state(
        panel_start="2015-01-09",
        panel_end="2025-12-31",
    )
    print("[build_panel_2025] DONE", flush=True)


if __name__ == "__main__":
    main()
