"""DySTAGE baseline trainer using the v2 protocol (matches RAG-STAR).

Same panel, masks, fold definitions, embargo, seeds, loss, and metrics
as ``src.v2.training.train_dow_epistar``. Wraps the vendored DySTAGE
architecture (NJIT-Fintech-Lab, ICAIF 2024, Gu et al.).

Run:
    python -m src.baselines.train_dystage_v2 --fold 1 --seed 42

Output: results/baselines_244/dystage_v2/fold{F}_seed{S}.json (+ npz).
"""
from __future__ import annotations

import argparse
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from src.baselines.dystage_adapter import (
    DySTAGEArgs,
    DySTAGEGraphConfig,
    build_day_graph,
)
from src.baselines.v2_runner import (
    V2BaselineConfig,
    build_age_features,
    build_masks,
    build_panel,
    cs_mse_loss,
    evaluate_predictions,
    fold_split,
    save_result,
    set_seeds,
    standardize_features,
    warmup_cosine_lr,
)
from src.baselines.vendored.dystage_models.DySTAGE import DySTAGE


@dataclass
class DySTAGEV2Config(V2BaselineConfig):
    output_dir: str = "results/baselines_244/dystage_v2"
    # DySTAGE-specific
    hist_time_steps: int = 12        # paper default
    n_heads: int = 4
    node_dim: int = 64
    attention_layers: int = 2
    temporal_head_config: str = "4"
    temporal_layer_config: str = "64"
    temporal_drop: float = 0.3
    residual: bool = True
    # Graph
    corr_window: int = 60
    corr_threshold: float = 0.3
    # Training
    learning_rate: float = 1e-4
    weight_decay: float = 5e-4       # paper default


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, choices=[1, 2, 3, 4, 5], required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--panel_kind", type=str, default="biotech",
                   choices=["biotech", "lattice_native"])
    p.add_argument("--two_regime_val", action="store_true")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--panel_end", type=str, default=None)
    args = p.parse_args()

    cfg = DySTAGEV2Config(fold=args.fold, seed=args.seed)
    if args.smoke:
        cfg.epochs = 2
    cfg.panel_kind = args.panel_kind
    cfg.two_regime_val = args.two_regime_val
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.panel_end:
        cfg.panel_end = args.panel_end
    elif args.panel_kind == "lattice_native":
        cfg.panel_end = "2025-12-31"

    set_seeds(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DySTAGE-v2] fold={cfg.fold} seed={cfg.seed} device={device}")

    x_raw, y, tickers, dates = build_panel(cfg)
    T, N, Fdim = x_raw.shape
    print(f"[DySTAGE-v2] panel: T={T} N={N} F={Fdim}")
    if N < 50:
        raise RuntimeError("Panel too small")

    mm = build_masks(cfg, dates, tickers)
    tradable = mm["tradable_mask"]
    loss_mask = mm["loss_mask"]
    hist20 = mm["history_valid_20d"]
    hist60 = mm["history_valid_60d"]

    train_idx, val_idx, test_idx = fold_split(cfg, dates)
    print(f"[DySTAGE-v2] fold {cfg.fold}: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    x = standardize_features(x_raw, tradable, train_idx)
    age_feat = build_age_features(tradable, hist20, hist60)
    age_days = age_feat[..., 0].astype(np.int64)

    # Build or load the per-day graph cache. Adjacency, edge features,
    # and shortest-path arrays depend only on log returns + mask, not on
    # the standardised features, so we can share them across folds and
    # seeds via a disk cache populated by `dystage_graph_cache.py`.
    graph_cfg = DySTAGEGraphConfig(
        corr_window=cfg.corr_window,
        corr_threshold=cfg.corr_threshold,
    )
    log_ret = x_raw[..., 0].astype(np.float32)
    # Cache is keyed by panel_kind so biotech and universal panels each get
    # their own. The biotech cache (T=2014, N=244) would crash on the
    # universal panel (T~2755, N~600); panel-kind-aware paths prevent that.
    cache_suffix = "_lattice_native" if cfg.panel_kind == "lattice_native" else ""
    # Prefer compute-node local /tmp (no quota, ~800GB) so DyStAGE runs
    # within one sbatch share cache without consuming home quota. Falls back
    # to home (data/processed/) if /tmp is not writable. Cache rebuild is
    # ~3 hours on universal panel, so even within-job sharing is a big win.
    import os as _os
    _user = _os.environ.get("USER", "adp232")
    tmp_dir = Path(f"/tmp/{_user}/dystage_cache")
    cache_path_tmp = tmp_dir / f"dystage_graph_cache{cache_suffix}.pt"
    cache_path_home = Path(f"data/processed/dystage_graph_cache{cache_suffix}.pt")
    # Use /tmp if writable, else home.
    try:
        tmp_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_path_tmp
    except Exception:
        cache_path = cache_path_home
    # Backwards-compat: if /tmp doesn't have the cache but home does, use home.
    if not cache_path.exists() and cache_path_home.exists():
        cache_path = cache_path_home
    print(f"[DySTAGE-v2] cache path: {cache_path}", flush=True)
    from torch_geometric.data import Data as _PygData
    graph_cache: list = []
    if cache_path.exists():
        print(f"[DySTAGE-v2] loading graph cache from {cache_path}")
        loaded = torch.load(cache_path, weights_only=False)
        cache_meta = loaded["config"]
        if (cache_meta["corr_window"] != graph_cfg.corr_window
                or cache_meta["corr_threshold"] != graph_cfg.corr_threshold):
            print("[DySTAGE-v2] cache config mismatches current cfg; rebuilding")
            cache_path = None
        else:
            for t in range(T):
                entry = loaded["cache"][t]
                graph_cache.append(_PygData(
                    x=torch.from_numpy(x[t]).float(),
                    edge_index=entry["edge_index"],
                    edge_weight=entry["edge_weight"],
                    edge_feat=entry["edge_feat"],
                    shortest_path_len=entry["shortest_path_len"],
                ))
    if not graph_cache:
        print(f"[DySTAGE-v2] precomputing graphs for T={T} days (this is slow)...")
        t_graph0 = time.time()
        for t in range(T):
            if t % 200 == 0 and t > 0:
                print(f"[DySTAGE-v2] graph cache: t={t}/{T} ({time.time()-t_graph0:.0f}s)")
            graph_cache.append(build_day_graph(t, log_ret, tradable, x, graph_cfg))
        print(f"[DySTAGE-v2] graph cache built in {time.time()-t_graph0:.0f}s")
        # Persist the cache so subsequent runs (other seeds, other folds)
        # don't repeat the multi-hour rebuild. Format matches the biotech
        # cache loader: dict with "config" and "cache" keys, where each
        # cache entry stores edge_index/weight/feat/shortest_path_len.
        cache_payload = {
            "config": {
                "corr_window": graph_cfg.corr_window,
                "corr_threshold": graph_cfg.corr_threshold,
                "panel_kind": cfg.panel_kind,
                "T": int(T),
                "N": int(graph_cache[0].x.shape[0]) if T > 0 else 0,
            },
            "cache": [
                {
                    "edge_index": g.edge_index,
                    "edge_weight": g.edge_weight,
                    "edge_feat": g.edge_feat,
                    "shortest_path_len": g.shortest_path_len,
                }
                for g in graph_cache
            ],
        }
        # Use a temp path then rename, so concurrent runs don't see a
        # partially-written file. Use the legacy zip-free pickle format
        # because the new torch zipfile serialiser can fail on >5GB cache
        # files (observed: "unexpected pos NNN vs NNN-104" at 5GB on GPFS).
        tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            torch.save(cache_payload, tmp_path,
                       _use_new_zipfile_serialization=False)
            tmp_path.replace(cache_path)
            print(f"[DySTAGE-v2] graph cache saved to {cache_path} "
                  f"({cache_path.stat().st_size/1e9:.1f} GB)")
        except Exception as e:
            print(f"[DySTAGE-v2] graph cache save FAILED: {e}; "
                  "this run will use in-memory cache only.")
            if tmp_path.exists():
                tmp_path.unlink()

    valid_feat_idx = torch.arange(Fdim, device=device)
    dy_args = DySTAGEArgs(
        hist_time_steps=cfg.hist_time_steps,
        n_heads=cfg.n_heads,
        node_dim=cfg.node_dim,
        attention_layers=cfg.attention_layers,
        temporal_head_config=cfg.temporal_head_config,
        temporal_layer_config=cfg.temporal_layer_config,
        temporal_drop=cfg.temporal_drop,
        residual=cfg.residual,
    )
    model = DySTAGE(
        args=dy_args, num_nodes=N, num_features=Fdim,
        edge_scale=len(graph_cfg.edge_scales),
        valid_feat_idx=valid_feat_idx,
    ).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    total_steps = cfg.epochs * max(1, len(train_idx))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optim, lr_lambda=lambda s: warmup_cosine_lr(s, cfg.warmup_steps, total_steps)
    )

    H = cfg.hist_time_steps

    def graphs_for(t: int) -> list:
        """Return the H-window of graph snapshots ending at t (exclusive of label t)."""
        return [graph_cache[k].to(device) for k in range(t - H, t)]

    def run_split(idx: np.ndarray, train_: bool) -> tuple[float, np.ndarray, np.ndarray]:
        model.train(train_)
        losses = []
        y_hat_all = np.zeros((T, N), dtype=np.float32)
        emask = np.zeros((T, N), dtype=bool)
        for t in idx:
            t = int(t)
            if t < H or loss_mask[t].sum() < 5:
                continue
            graphs = graphs_for(t)
            y_pred = model(graphs)            # [N]
            y_full = y_pred.float()
            l = cs_mse_loss(y_full, torch.from_numpy(y[t]).to(device),
                            torch.from_numpy(loss_mask[t]).to(device))
            if train_:
                optim.zero_grad()
                l.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optim.step()
                scheduler.step()
            losses.append(float(l.item()))
            y_hat_all[t] = y_full.detach().cpu().numpy()
            emask[t] = loss_mask[t]
        return (float(np.mean(losses)) if losses else float("nan"), y_hat_all, emask)

    history: list = []
    best_val_ic = -1e9
    best_state = None
    patience = 0
    for epoch in range(cfg.epochs):
        t0 = time.time()
        np.random.seed(cfg.seed + epoch)
        perm = np.random.permutation(train_idx)
        train_loss, _, _ = run_split(perm, train_=True)
        val_loss, val_yhat, val_mask = run_split(val_idx, train_=False)
        val_metrics = evaluate_predictions(val_yhat, y, val_mask, age_days)
        dt = time.time() - t0
        improved = val_metrics["ic"] > best_val_ic + 1e-5
        print(f"[DySTAGE-v2] epoch {epoch}: train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} val_ic={val_metrics['ic']:+.4f} "
              f"({dt:.1f}s)" + ("  *best*" if improved else ""))
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_ic": val_metrics["ic"],
            "val_rank_ic": val_metrics["rank_ic"],
            "time_sec": round(dt, 2),
        })
        if improved:
            best_val_ic = val_metrics["ic"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= cfg.early_stop_patience:
                print(f"[DySTAGE-v2] early stop epoch {epoch} best_val_ic={best_val_ic:+.4f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    _, test_yhat, test_mask = run_split(test_idx, train_=False)
    test_metrics = evaluate_predictions(test_yhat, y, test_mask, age_days)
    val_metrics_final = evaluate_predictions(val_yhat, y, val_mask, age_days)
    print(f"[DySTAGE-v2] TEST ic={test_metrics['ic']:+.4f} rank_ic={test_metrics['rank_ic']:+.4f}")

    out_path = save_result(
        out_dir=Path(cfg.output_dir),
        fold=cfg.fold, seed=cfg.seed,
        model_name="DySTAGE (v2 protocol)",
        test_metrics=test_metrics,
        val_metrics=val_metrics_final,
        test_y_hat=test_yhat,
        test_eval_mask=test_mask,
        history=history,
        config=asdict(cfg),
        n_panel=(T, N, Fdim),
        n_train=len(train_idx), n_val=len(val_idx), n_test=len(test_idx),
        y_true=y, tickers=tickers, dates=dates,
        age_days=age_days, tradable_mask=tradable,
    )
    print(f"[DySTAGE-v2] wrote {out_path}")


if __name__ == "__main__":
    main()
