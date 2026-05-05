"""Precompute the per-day DySTAGE graph cache once and save to disk.

The graph cache (adjacency edge index, edge features, shortest paths)
depends only on log returns, tradable mask, and the graph hyperparameters,
none of which vary across folds or seeds. Running this script once saves
~8.5 minutes per (fold, seed) downstream training run.

Cache format: list of dicts, one per trading day, with keys
    edge_index      : long tensor [2, E]
    edge_weight     : float tensor [E]
    edge_feat       : float tensor [N, N, edge_scale]
    shortest_path_len : long tensor [N, N]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from src.baselines.dystage_adapter import (
    DySTAGEGraphConfig,
    build_day_graph,
)
from src.baselines.v2_runner import build_masks, build_panel, V2BaselineConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/processed/dystage_graph_cache.pt",
                        help="Output cache path")
    parser.add_argument("--corr-window", type=int, default=60)
    parser.add_argument("--corr-threshold", type=float, default=0.3)
    args = parser.parse_args()

    cfg = V2BaselineConfig()
    cache_cfg = DySTAGEGraphConfig(
        corr_window=args.corr_window,
        corr_threshold=args.corr_threshold,
    )
    print(f"[graph-cache] config: window={cache_cfg.corr_window} "
          f"threshold={cache_cfg.corr_threshold} scales={cache_cfg.edge_scales}")

    x_raw, _, _, dates = build_panel(cfg)
    T, N, _ = x_raw.shape
    mm = build_masks(cfg, dates, ["dummy"] * N)
    tradable = mm["tradable_mask"]
    log_ret = x_raw[..., 0].astype(np.float32)
    print(f"[graph-cache] panel: T={T} N={N}")

    # Use a placeholder for x; downstream loads x per fold from
    # the standardised features, not from the cache.
    placeholder_x = np.zeros((T, N, 1), dtype=np.float32)

    cache = []
    t0 = time.time()
    for t in range(T):
        if t % 200 == 0 and t > 0:
            print(f"[graph-cache] t={t}/{T} ({time.time()-t0:.0f}s)")
        g = build_day_graph(t, log_ret, tradable, placeholder_x, cache_cfg)
        cache.append({
            "edge_index": g.edge_index.detach().clone(),
            "edge_weight": g.edge_weight.detach().clone(),
            "edge_feat": g.edge_feat.detach().clone(),
            "shortest_path_len": g.shortest_path_len.detach().clone(),
        })
    print(f"[graph-cache] built in {time.time()-t0:.0f}s")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "cache": cache,
        "config": {
            "corr_window": cache_cfg.corr_window,
            "corr_threshold": cache_cfg.corr_threshold,
            "edge_scales": list(cache_cfg.edge_scales),
            "shortest_path_cap": cache_cfg.shortest_path_cap,
        },
        "n_panel": (T, N),
    }, out)
    print(f"[graph-cache] wrote {out}")


if __name__ == "__main__":
    main()
