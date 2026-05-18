"""Print per-feature statistics for the RAG-STAR Universe panel.

Writes a markdown table to docs/rag_star_universe_features.md so the
user can compare feature provenance, NaN/zero density, and value
ranges across the 22 v2 FEATURE_COLS.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.v2.data.lattice_panel import (
    LatticePanelConfig, build_lattice_panel,
    _PRICE_RENAME, _FUND_PROXY, _ST_MAP,
)
from src.mtgn.training.panel_enriched import (
    PRICE_COLS, ST_COLS, FUND_COLS, FLAG_COLS, FEATURE_COLS,
)


def _source(col: str) -> str:
    if col in PRICE_COLS:
        if col in _PRICE_RENAME.values():
            src_name = [k for k, v in _PRICE_RENAME.items() if v == col][0]
            return f"PRICE LATTICE rename <- {src_name}"
        return "PRICE LATTICE direct"
    if col in ST_COLS:
        mapped = _ST_MAP.get(col)
        return f"ST joined <- {mapped}"
    if col in FUND_COLS:
        return f"FUND LATTICE <- {_FUND_PROXY.get(col, col)}"
    if col in FLAG_COLS:
        return "FLAG LATTICE direct"
    return "UNKNOWN"


def main() -> None:
    print("[inventory] building RAG-STAR Universe panel ...", flush=True)
    cfg = LatticePanelConfig(end_date="2023-12-31")
    panel, tickers, dates = build_lattice_panel(cfg)
    print(f"[inventory] panel rows={len(panel):,} tickers={len(tickers)} "
          f"dates={len(dates)}", flush=True)

    out_lines = [
        "# RAG-STAR Universe panel: feature inventory",
        "",
        f"Built {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}. "
        f"Panel: T={len(dates)} dates, N={len(tickers)} tickers, "
        f"F=22 (v2 schema), {len(panel):,} non-empty rows.",
        f"Date range: {pd.Timestamp(min(dates)).strftime('%Y-%m-%d')} to "
        f"{pd.Timestamp(max(dates)).strftime('%Y-%m-%d')}.",
        "",
        "All NaN cells are filled with 0 by the adapter "
        "(`pd.to_numeric(...).fillna(0.0)`), so the %zero column "
        "subsumes the missingness signal.",
        "",
        "| # | feature | source | %zero | mean | std | min | max |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for i, c in enumerate(FEATURE_COLS):
        v = pd.to_numeric(panel[c], errors="coerce").astype(float)
        n_zero = int((v == 0).sum())
        pct_zero = 100.0 * n_zero / len(v)
        nonzero = v[v != 0]
        if len(nonzero) > 0:
            mu = float(nonzero.mean()); sd = float(nonzero.std())
            mn = float(nonzero.min()); mx = float(nonzero.max())
        else:
            mu = sd = mn = mx = 0.0
        out_lines.append(
            f"| {i} | `{c}` | {_source(c)} | "
            f"{pct_zero:.1f}% | {mu:+.4f} | {sd:.4f} | "
            f"{mn:+.4f} | {mx:+.4f} |"
        )

    fwd = pd.to_numeric(panel["fwd_return_h"], errors="coerce").astype(float)
    n_fwd_nan = int(fwd.isna().sum())
    out_lines += [
        "",
        f"Target `fwd_return_h`: NaN={n_fwd_nan} "
        f"mean={fwd.mean():+.4f} std={fwd.std():.4f} "
        f"min={fwd.min():+.4f} max={fwd.max():+.4f}",
        "",
        "**Notes**",
        "",
        f"- StockTwits join: {sum(1 for c in ST_COLS if _ST_MAP.get(c) in panel.columns or panel[c].std() > 0)} "
        f"of 5 v2 ST cols populated; the rest are zero-filled because LATTICE "
        f"either lacks the column name or the parquet didn't cover the row.",
        "- The v2 22-col schema is preserved bit-for-bit so the existing "
        "RAG-STAR architecture reads transparently. Substitutions: "
        "`cash_runway_q <- interest_coverage`, `rd_intensity <- rd_to_sales`, "
        "`revenue_growth_yoy <- asset_growth_yoy`, `cash_to_mc <- book_to_market`, "
        "`shares_outstanding_yoy <- capex_to_sales`, "
        "`total_assets_growth <- asset_growth_yoy`.",
    ]

    out_path = Path("docs/rag_star_universe_features.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out_lines))
    print(f"[inventory] wrote {out_path}", flush=True)
    for line in out_lines:
        print(line)


if __name__ == "__main__":
    main()
