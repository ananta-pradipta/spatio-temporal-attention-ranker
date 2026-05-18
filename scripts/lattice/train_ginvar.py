"""Train G-InVAR on the LATTICE S&P 500 panel.

Reuses the InvarDataset (with lookback=20 instead of 60) and the v1
trainer scaffolding. Per spec section 8 the loss is daily cross-sectional
z-scored MSE; metrics include daily IC, rank IC, and NDCG@10 / NDCG@50.

Usage::

    PYTHONPATH=$PWD python3 -u -m scripts.lattice.train_ginvar \\
        --fold 1 --seed 42 --epochs 10 \\
        --graph-mode dense \\
        --output-dir experiments/ginvar/baseline
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from src.invar.data.dataset import InvarDataset, PANEL_FEATURE_DIM, MACRO_FEATURE_DIM
from src.invar.evaluation.metrics import daily_ic, daily_rank_ic, ndcg_at_k, long_short_sharpe
from src.lattice.models.ginvar.model import GInVAR, GInVARConfig, count_parameters
from src.lattice.models.ginvar.losses import cs_zscored_mse_loss
from src.lattice.models.ginvar.graph_builders import (
    build_sector_graph_per_day, row_normalise,
    build_correlation_graph_per_day, build_factor_graph_per_day,
    build_social_graph_per_day,
    compute_stress_per_day, regime_blend_weights, blend_graphs,
)


@dataclass
class GInVARTrainConfig:
    fold: int = 1
    seed: int = 42
    lr: float = 1.0e-4
    weight_decay: float = 1.0e-5
    warmup_steps: int = 500
    grad_clip: float = 1.0
    epochs: int = 10
    early_stop_patience: int = 3
    output_dir: str = "experiments/ginvar/baseline"
    save_predictions: bool = True

    graph_mode: str = "dense"
    top_k: int = 16
    use_corr_graph: bool = False
    use_sector_graph: bool = False
    use_factor_graph: bool = False
    use_social_graph: bool = False
    use_beta_graph: bool = False
    lookback: int = 20
    n_layers: int = 2


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def warmup_cosine(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return max(1, step) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))


def to_batch_inputs(
    invar_batch, A_full: torch.Tensor | None, day_index: int,
    n_universe: int, active_idx: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Adapt an InvarDayBatch to G-InVAR's (B=1) input shapes."""
    # InvarDayBatch.features is (N_t, L, F); we need (1, L, N_t, F).
    features = invar_batch.features.permute(1, 0, 2).unsqueeze(0)
    mask = invar_batch.mask.unsqueeze(0).bool()
    A: torch.Tensor | None = None
    if A_full is not None:
        # A_full is (T, N_universe, N_universe). Slice to active subset.
        A_day = A_full[day_index][np.ix_(active_idx, active_idx)]
        A = A_day.unsqueeze(0)
    return features, A, mask


def evaluate(
    model: GInVAR, dataset: InvarDataset, A_full: torch.Tensor | None,
    device: torch.device, collect_predictions: bool = False,
) -> dict:
    model.eval()
    pred_rows = []
    with torch.no_grad():
        for batch in dataset:
            t = int(batch.day_index)
            mask_full = dataset._mask_tensor[t]
            active_idx = np.where(mask_full)[0]
            features, A, mask = to_batch_inputs(
                batch, A_full, t, dataset.n_universe, active_idx,
            )
            features = features.to(device)
            mask = mask.to(device)
            if A is not None:
                A = A.to(device)
            scores, _ = model(features, A, mask, return_attn=False)
            yh = scores[0].cpu().numpy()
            yt = batch.y_cs.numpy()
            sec = batch.sector_id.numpy()
            sd = batch.size_decile.numpy()
            ab = batch.age_bucket.numpy()
            date_str = batch.date.strftime("%Y-%m-%d")
            for i in range(batch.features.shape[0]):
                pred_rows.append({
                    "date": date_str, "ticker": batch.tickers[i],
                    "y_hat": float(yh[i]), "y_true": float(yt[i]),
                    "sector_id": int(sec[i]),
                    "size_decile": int(sd[i]),
                    "age_bucket": int(ab[i]),
                })
    if not pred_rows:
        return {"ic": float("nan"), "rank_ic": float("nan"),
                "ndcg10": float("nan"), "ndcg50": float("nan"),
                "sharpe": float("nan"), "predictions": []}
    df = pd.DataFrame(pred_rows)
    ic = daily_ic(df)
    rank = daily_rank_ic(df)
    ndcg10 = ndcg_at_k(df, k=10)
    ndcg50 = ndcg_at_k(df, k=50)
    sharpe = long_short_sharpe(df)
    return {
        "ic": ic["mean"], "rank_ic": rank["mean"],
        "ndcg10": ndcg10["mean"], "ndcg50": ndcg50["mean"],
        "sharpe": sharpe["sharpe"],
        "n_days": ic["n_days"],
        "predictions": pred_rows if collect_predictions else None,
    }


def build_full_graph(
    dataset: InvarDataset,
    use_sector: bool, use_corr: bool, use_factor: bool, use_social: bool,
    train_idx: np.ndarray,
) -> torch.Tensor | None:
    """Build all enabled graphs, compute regime stress + blend, return A_graph.

    Each individual graph is row-normalised; the blend is the per-day
    weighted sum (weights deterministic from the macro stress index per
    spec section 3); the final blended graph is row-normalised again so
    each row sums to 1 for the attention prior.
    """
    if not (use_sector or use_corr or use_factor or use_social):
        return None
    from src.lattice.data.build_panel import (
        PANEL_FEATURE_COLS, MACRO_FEATURE_COLS, ST_FEATURE_COLS,
    )

    graphs: dict[str, np.ndarray] = {}

    if use_sector:
        constituents_path = Path("data/lattice/raw/sp500_constituents_pit.parquet")
        if not constituents_path.exists():
            raise FileNotFoundError(f"Missing PIT constituents: {constituents_path}")
        print(f"[ginvar] sector graph over "
              f"{dataset.n_universe} tickers x {len(dataset.dates)} days...",
              flush=True)
        A_sec = build_sector_graph_per_day(
            constituents_path, dataset.tickers_universe,
            [pd.Timestamp(d) for d in dataset.dates],
        )
        graphs["sector"] = row_normalise(A_sec)

    if use_corr or use_factor or use_social:
        log_ret_idx = PANEL_FEATURE_COLS.index("log_return")
        log_returns = dataset._panel_tensor_raw[..., log_ret_idx]
        log_returns = np.where(dataset._mask_tensor, log_returns, np.nan)

    if use_corr:
        print(f"[ginvar] correlation graph (60-day, reliability-shrunk)...", flush=True)
        A_corr = build_correlation_graph_per_day(
            log_returns, dataset._mask_tensor,
            window=60, min_overlap=20, tau=30.0,
        )
        graphs["corr"] = row_normalise(A_corr)

    if use_factor:
        print(f"[ginvar] factor-similarity graph...", flush=True)
        factor_cols = [
            "log_market_cap", "log_volume", "log_volume_ratio_20d",
            "amihud_illiquidity_20d",
            "realized_vol_20d", "realized_vol_60d", "high_low_range",
            "book_to_market", "fcf_yield", "gross_profitability",
            "asset_growth_yoy",
            "interest_coverage", "net_debt_to_ebitda", "current_ratio",
            "rd_to_sales", "sga_to_sales", "capex_to_sales",
            "days_to_next_catalyst_sin", "days_to_next_catalyst_cos",
            "catalyst_type_id",
        ]
        idx = [PANEL_FEATURE_COLS.index(c) for c in factor_cols]
        # Use raw (pre-scaling) values so the per-day cs standardisation
        # within build_factor_graph_per_day is a faithful regime read.
        factor_features = dataset._panel_tensor_raw[..., idx]
        A_fac = build_factor_graph_per_day(
            factor_features, dataset._mask_tensor,
        )
        graphs["factor"] = row_normalise(A_fac)

    if use_social:
        print(f"[ginvar] social graph (StockTwits cosine)...", flush=True)
        # Load stocktwits features parquet directly.
        st_df = pd.read_parquet("data/lattice/processed/stocktwits_features.parquet")
        st_df["date"] = pd.to_datetime(st_df["date"])
        date_to_idx = {d: i for i, d in enumerate(dataset.dates)}
        ticker_to_idx = {t: i for i, t in enumerate(dataset.tickers_universe)}
        T, N = len(dataset.dates), dataset.n_universe
        st_tensor = np.zeros((T, N, len(ST_FEATURE_COLS)), dtype=np.float32)
        has_st = np.zeros((T, N), dtype=bool)
        st_in = st_df.assign(
            di=st_df["date"].map(date_to_idx),
            ti=st_df["ticker"].map(ticker_to_idx),
        ).dropna(subset=["di", "ti"])
        st_in["di"] = st_in["di"].astype(int)
        st_in["ti"] = st_in["ti"].astype(int)
        st_tensor[st_in["di"].values, st_in["ti"].values] = (
            st_in[ST_FEATURE_COLS].to_numpy(dtype=np.float32)
        )
        has_st[st_in["di"].values, st_in["ti"].values] = True
        A_soc = build_social_graph_per_day(
            st_tensor, has_st, dataset._mask_tensor,
        )
        graphs["social"] = row_normalise(A_soc)

    # Compute stress per day and regime-adaptive blend weights.
    print(f"[ginvar] computing stress index + regime blend weights...", flush=True)
    log_ret_idx = PANEL_FEATURE_COLS.index("log_return")
    panel_log_ret = dataset._panel_tensor_raw[..., log_ret_idx]
    stress = compute_stress_per_day(
        dataset._macro_raw_tensor, panel_log_ret, dataset._mask_tensor,
        list(MACRO_FEATURE_COLS), train_idx,
    )
    weights = regime_blend_weights(
        stress, use_corr=use_corr, use_sector=use_sector,
        use_factor=use_factor, use_social=use_social,
    )
    A_blended = blend_graphs(graphs, weights)
    A_blended = row_normalise(A_blended)
    print(f"[ginvar] blended graph: shape {A_blended.shape}, "
          f"sources {list(graphs.keys())}, stress range "
          f"[{stress.min():+.2f}, {stress.max():+.2f}]", flush=True)
    return torch.from_numpy(A_blended).float()


def train_one(cfg: GInVARTrainConfig) -> dict:
    set_seeds(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ginvar] fold={cfg.fold} seed={cfg.seed} device={device} "
          f"graph_mode={cfg.graph_mode}", flush=True)

    train_ds = InvarDataset(fold=cfg.fold, split="train", lookback=cfg.lookback)
    val_ds = InvarDataset(fold=cfg.fold, split="val", lookback=cfg.lookback)
    test_ds = InvarDataset(fold=cfg.fold, split="test", lookback=cfg.lookback)
    print(f"[ginvar] train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}",
          flush=True)

    A_full: torch.Tensor | None = None
    if cfg.graph_mode != "dense":
        # Default to sector if no explicit selection, to preserve
        # backwards-compat with the Phase 1 sbatches.
        if not (cfg.use_sector_graph or cfg.use_corr_graph
                  or cfg.use_factor_graph or cfg.use_social_graph):
            cfg.use_sector_graph = True
        A_full = build_full_graph(
            train_ds,
            use_sector=cfg.use_sector_graph,
            use_corr=cfg.use_corr_graph,
            use_factor=cfg.use_factor_graph,
            use_social=cfg.use_social_graph,
            train_idx=train_ds.train_idx,
        )

    model_cfg = GInVARConfig(
        n_features=PANEL_FEATURE_DIM, lookback=cfg.lookback,
        macro_dim=MACRO_FEATURE_DIM, graph_mode=cfg.graph_mode,
        top_k=cfg.top_k, n_layers=cfg.n_layers,
    )
    model = GInVAR(model_cfg).to(device)
    n_params = count_parameters(model)
    print(f"[ginvar] params={n_params:,}", flush=True)

    optim = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = max(len(train_ds), 1) * cfg.epochs
    scheduler = LambdaLR(optim,
                          lambda s: warmup_cosine(s, cfg.warmup_steps, total_steps))
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    out_dir = Path(cfg.output_dir) / f"fold{cfg.fold}/seed{cfg.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    best_val_ic = -1.0
    best_epoch = -1
    epochs_no_improve = 0

    for epoch in range(cfg.epochs):
        model.train()
        epoch_losses = []
        eligible = list(train_ds._eligible_idx)
        random.shuffle(eligible)
        for t in eligible:
            batch = train_ds.get(int(t))
            mask_full = train_ds._mask_tensor[int(t)]
            active_idx = np.where(mask_full)[0]
            features, A, mask = to_batch_inputs(
                batch, A_full, int(t), train_ds.n_universe, active_idx,
            )
            features = features.to(device)
            mask = mask.to(device)
            if A is not None:
                A = A.to(device)
            target = batch.y_cs.unsqueeze(0).to(device)

            optim.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
                scores, _ = model(features, A, mask, return_attn=False)
                loss = cs_zscored_mse_loss(scores, target, mask)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optim)
            scaler.update()
            scheduler.step()
            epoch_losses.append(float(loss.item()))

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        val_metrics = evaluate(model, val_ds, A_full, device)
        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "val_ic": val_metrics["ic"], "val_rank_ic": val_metrics["rank_ic"],
            "val_ndcg10": val_metrics["ndcg10"],
        })
        print(f"[ginvar] epoch {epoch}: loss={train_loss:.4f} "
              f"val_ic={val_metrics['ic']:+.4f} "
              f"val_rank_ic={val_metrics['rank_ic']:+.4f} "
              f"val_ndcg10={val_metrics['ndcg10']:.4f}", flush=True)

        if np.isfinite(val_metrics["ic"]) and val_metrics["ic"] > best_val_ic:
            best_val_ic = val_metrics["ic"]
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save({"model_state": model.state_dict(),
                          "best_epoch": epoch}, out_dir / "ckpt.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.early_stop_patience:
                print(f"[ginvar] early stop at epoch {epoch}", flush=True)
                break

    ckpt = torch.load(out_dir / "ckpt.pt", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    test_metrics = evaluate(model, test_ds, A_full, device,
                              collect_predictions=cfg.save_predictions)
    pred_rows = test_metrics.pop("predictions", None)
    if cfg.save_predictions and pred_rows:
        pd.DataFrame(pred_rows).to_parquet(
            out_dir / "predictions.parquet", index=False,
        )

    print(f"[ginvar] test_ic={test_metrics['ic']:+.4f} "
          f"rank_ic={test_metrics['rank_ic']:+.4f} "
          f"ndcg10={test_metrics['ndcg10']:.4f} "
          f"sharpe={test_metrics['sharpe']:.3f}", flush=True)

    result = {
        "config": asdict(cfg),
        "model_config": model_cfg.__dict__,
        "n_params": int(n_params),
        "best_val_ic": float(best_val_ic),
        "best_epoch": int(best_epoch),
        "test_ic": float(test_metrics["ic"]),
        "test_rank_ic": float(test_metrics["rank_ic"]),
        "test_ndcg10": float(test_metrics["ndcg10"]),
        "test_ndcg50": float(test_metrics["ndcg50"]),
        "test_sharpe": float(test_metrics["sharpe"]),
        "history": history,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--output-dir", type=str, default="experiments/ginvar/baseline")
    p.add_argument("--graph-mode", type=str, default="dense",
                    choices=["dense", "graph_bias", "graph_mask",
                             "graph_bias_and_mask"])
    p.add_argument("--top-k", type=int, default=16)
    p.add_argument("--use-sector-graph", action="store_true")
    p.add_argument("--use-corr-graph", action="store_true")
    p.add_argument("--use-factor-graph", action="store_true")
    p.add_argument("--use-social-graph", action="store_true")
    p.add_argument("--use-beta-graph", action="store_true")
    p.add_argument("--lookback", type=int, default=20)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--save-predictions", action="store_true", default=True)
    args = p.parse_args()
    cfg = GInVARTrainConfig(
        fold=args.fold, seed=args.seed, epochs=args.epochs,
        output_dir=args.output_dir, graph_mode=args.graph_mode,
        top_k=args.top_k, use_sector_graph=args.use_sector_graph,
        use_corr_graph=args.use_corr_graph,
        use_factor_graph=args.use_factor_graph,
        use_social_graph=args.use_social_graph,
        use_beta_graph=args.use_beta_graph,
        lookback=args.lookback, n_layers=args.n_layers,
        save_predictions=args.save_predictions,
    )
    train_one(cfg)


if __name__ == "__main__":
    main()
