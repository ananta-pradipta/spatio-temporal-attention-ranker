"""Phase 3 driver: build the universal S&P 500 panel and emit a smoke-test
summary plus a tensor snapshot.

Output:
    data/processed/sp500_snapshots.pt   torch dict {x, y, mask, tickers, dates}
    logs/universal_validation/phase3_panel.json  summary stats
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.v2.data.universal_panel import (
    UniversalPanelConfig, build_universal_panel, universal_panel_to_tensors,
)


def main() -> None:
    print("[universal-panel] starting build", flush=True)
    t0 = time.time()
    cfg = UniversalPanelConfig()
    panel, tickers, dates = build_universal_panel(cfg)
    print(f"[universal-panel] build wall-clock: {time.time()-t0:.1f}s", flush=True)
    print(f"[universal-panel] panel rows: {len(panel):,}  tickers: {len(tickers)}  dates: {len(dates)}", flush=True)

    tens = universal_panel_to_tensors(panel, tickers, dates)
    x, y, m = tens["x"], tens["y"], tens["mask"]
    print(f"[universal-panel] x shape: {x.shape}  y shape: {y.shape}  mask sum: {int(m.sum()):,}", flush=True)

    out = Path("data/processed/sp500_snapshots.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "x": torch.from_numpy(x), "y": torch.from_numpy(y), "mask": torch.from_numpy(m),
        "tickers": tickers, "dates": [str(d.date()) for d in dates],
    }, out)
    print(f"[universal-panel] wrote {out}", flush=True)

    # Summary stats
    summary = {
        "build_seconds": time.time() - t0,
        "panel_rows": len(panel), "tickers": len(tickers), "dates": len(dates),
        "shape_T_N_F": list(x.shape),
        "active_cells": int(m.sum()),
        "active_density": float(m.mean()),
        "feature_describe": {
            "altman_z (col=cash_runway_q)":   panel.cash_runway_q.describe().to_dict(),
            "capex_intensity (col=rd_intensity)": panel.rd_intensity.describe().to_dict(),
            "log_market_cap":                 panel.log_market_cap.describe().to_dict(),
            "realized_vol_60d":               panel.realized_vol_60d.describe().to_dict(),
        },
    }
    log_path = Path("logs/universal_validation/phase3_panel.json")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"[universal-panel] wrote {log_path}", flush=True)


if __name__ == "__main__":
    main()
