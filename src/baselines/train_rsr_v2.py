"""RSR baseline trainer using the v2 protocol (matches RAG-STAR).

Same panel, masks, fold definitions, embargo, seeds, and metrics as
``src.v2.training.train_dow_epistar``. The only difference from
RAG-STAR is the model: this script wraps the vendored RSR
architecture (Feng et al., TOIS 2019).

Reference code source: github.com/fulifeng/Temporal_Relational_Stock_Ranking
(TensorFlow 1.x, MIT-licensed). We ported the LSTM + Explicit
relation-rank attention modules to PyTorch under
``src/baselines/vendored/rsr/``.

Hyperparameters: we keep the AdamW + warmup-cosine schedule, fp16
autocast, and gradient clipping conventions of every other v2 baseline
so that fairness only requires controlling the architecture. The
RSR-specific knobs default to the TOIS 2019 paper's reported NASDAQ
configuration: LSTM hidden = 64, single layer, leaky slope 0.2.

Loss: cross-sectional MSE on z-scored 5d forward log returns (the v2
protocol's ``cs_mse_loss``). The original RSR paper uses a pairwise
margin ranking loss; for v2 fairness we deliberately swap to
``cs_mse_loss`` so that every baseline is trained against the same
loss surface and any IC differences are attributable to architecture.

Run:
    python -m src.baselines.train_rsr_v2 --fold 1 --seed 42

Output: results/baselines_244/rsr_v2/fold{F}_seed{S}.json (+ npz).
"""
from __future__ import annotations

import argparse
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from src.baselines.rsr_adapter import RSRAdapter, RSRHyperparams
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


def _relation_graph_path_for_fold(fold: int) -> Path:
    """Return the per-fold relation-graph path.

    The graph is rebuilt per fold using only data available at the
    fold's train-end date; see ``src/baselines/build_rsr_relation_graph.py``.
    """
    return Path(f"data/processed/rsr_relation_graph_fold{fold}.pt")


@dataclass
class RSRV2Config(V2BaselineConfig):
    """Top-level config bundling the v2 protocol + RSR-specific knobs.

    RSR hyperparameter rationale:
      - hidden_size = 64 matches the TOIS 2019 paper's NASDAQ config.
      - num_layers = 1 matches the paper's default.
      - leaky_slope = 0.2 matches the upstream TF1 implementation.
      - head_hidden = 64 mirrors hidden_size; the paper's score head
        is a thin linear projection so 64 is on the upper end.
    """

    output_dir: str = "results/baselines_244/rsr_v2"
    # RSR architecture.
    d_model: int = 64                # LSTM hidden + attention width
    num_layers: int = 1              # LSTM depth
    head_hidden: int = 64            # MLP head hidden width
    leaky_slope: float = 0.2         # LeakyReLU slope in attention scoring
    dropout: float = 0.0
    # Per-fold relation graph; populated in main() once we know the fold.
    relation_graph_path: str = ""


def _load_relation_graph(
    cfg: RSRV2Config, panel_tickers: list[str]
) -> tuple[torch.Tensor, dict]:
    """Load the saved adjacency and align rows/cols to the panel ticker order.

    Returns:
        (A_panel, info) where A_panel is a (N_panel, N_panel) uint8
        tensor in panel ticker order, and info is a dict with edge
        counts and density on the panel-restricted graph.
    """
    path = Path(cfg.relation_graph_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Relation graph not found at {path}. Build it first via "
            f"`python -m src.baselines.build_rsr_relation_graph`."
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    A_full = payload["A"].numpy()
    src_tickers = [str(t).upper() for t in payload["tickers"]]
    src_to_idx = {t: i for i, t in enumerate(src_tickers)}

    n_panel = len(panel_tickers)
    A_panel = np.zeros((n_panel, n_panel), dtype=np.uint8)
    n_missing = 0
    for i, ti in enumerate(panel_tickers):
        ti_u = str(ti).upper()
        if ti_u not in src_to_idx:
            n_missing += 1
            continue
        si = src_to_idx[ti_u]
        for j, tj in enumerate(panel_tickers):
            if i == j:
                continue
            tj_u = str(tj).upper()
            if tj_u not in src_to_idx:
                continue
            sj = src_to_idx[tj_u]
            A_panel[i, j] = A_full[si, sj]
    n_edges = int(A_panel.sum())
    density = float(n_edges / (n_panel * (n_panel - 1))) if n_panel > 1 else 0.0
    info = {
        "n_panel": n_panel,
        "n_missing_classification": n_missing,
        "n_edges_directed": n_edges,
        "density": density,
        "mean_out_degree": float(A_panel.sum(axis=1).mean()),
    }
    return torch.from_numpy(A_panel), info


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, choices=[1, 2, 3], required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true",
                   help="Limit to 2 epochs and abbreviated output.")
    p.add_argument("--max_epochs", type=int, default=None,
                   help="Override config.epochs (e.g. 1 for a smoke check).")
    args = p.parse_args()

    cfg = RSRV2Config(fold=args.fold, seed=args.seed)
    cfg.relation_graph_path = str(_relation_graph_path_for_fold(args.fold))
    if args.smoke:
        cfg.epochs = 2
    if args.max_epochs is not None:
        cfg.epochs = int(args.max_epochs)

    set_seeds(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[RSR-v2] fold={cfg.fold} seed={cfg.seed} device={device}")
    print(f"[RSR-v2] relation_graph_path={cfg.relation_graph_path}")

    x_raw, y, tickers, dates = build_panel(cfg)
    T, N, Fdim = x_raw.shape
    print(f"[RSR-v2] panel: T={T} N={N} F={Fdim}")
    if N < 50:
        raise RuntimeError("Panel too small")

    A_panel, graph_info = _load_relation_graph(cfg, tickers)
    print(
        f"[RSR-v2] relation graph: edges={graph_info['n_edges_directed']} "
        f"density={graph_info['density']:.4f} "
        f"mean_deg={graph_info['mean_out_degree']:.1f} "
        f"missing_class={graph_info['n_missing_classification']}"
    )

    mm = build_masks(cfg, dates, tickers)
    tradable = mm["tradable_mask"]
    loss_mask = mm["loss_mask"]
    hist20 = mm["history_valid_20d"]
    hist60 = mm["history_valid_60d"]

    train_idx, val_idx, test_idx = fold_split(cfg, dates)
    print(f"[RSR-v2] fold {cfg.fold}: "
          f"train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    x = standardize_features(x_raw, tradable, train_idx)
    age_feat = build_age_features(tradable, hist20, hist60)
    age_days = age_feat[..., 0].astype(np.int64)

    x_t = torch.from_numpy(x).to(device)
    y_t = torch.from_numpy(y).to(device)

    W = cfg.temporal_window

    hp = RSRHyperparams(
        d_feat=Fdim,
        hidden_size=cfg.d_model,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        head_hidden=cfg.head_hidden,
        leaky_slope=cfg.leaky_slope,
    )
    model = RSRAdapter(hp, A_panel).to(device)
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    total_steps = cfg.epochs * max(1, len(train_idx))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optim, lr_lambda=lambda s: warmup_cosine_lr(s, cfg.warmup_steps, total_steps)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    def run_split(idx: np.ndarray, train_: bool) -> tuple[float, np.ndarray, np.ndarray]:
        model.train(train_)
        losses = []
        y_hat_all = np.zeros((T, N), dtype=np.float32)
        emask = np.zeros((T, N), dtype=bool)
        for t in idx:
            t = int(t)
            if t < W - 1:
                continue
            m_np = tradable[t]
            if m_np.sum() < 8:
                continue
            active_idx = np.flatnonzero(m_np)
            active_t = torch.from_numpy(active_idx).to(device)
            x_win = x_t[t - W + 1: t + 1, active_t, :].transpose(0, 1)  # (A, W, F)
            y_target_full = y_t[t]                                      # (N,)
            lmask_t = torch.from_numpy(loss_mask[t]).to(device)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                y_hat_active = model(x_win, active_t)
                y_full = torch.zeros(N, device=device, dtype=y_hat_active.dtype)
                y_full[active_t] = y_hat_active
                cs_loss = cs_mse_loss(y_full, y_target_full, lmask_t)
                total_loss = cs_loss

            if train_:
                optim.zero_grad()
                scaler.scale(total_loss).backward()
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optim)
                scaler.update()
                scheduler.step()
            losses.append(float(total_loss.item()))
            y_hat_all[t] = y_full.detach().float().cpu().numpy()
            emask[t] = loss_mask[t]
        return (float(np.mean(losses)) if losses else float("nan"),
                y_hat_all, emask)

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
        print(f"[RSR-v2] epoch {epoch}: train_loss={train_loss:.4f} "
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
                print(f"[RSR-v2] early stop epoch {epoch} "
                      f"best_val_ic={best_val_ic:+.4f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    _, test_yhat, test_mask = run_split(test_idx, train_=False)
    test_metrics = evaluate_predictions(test_yhat, y, test_mask, age_days)
    val_metrics_final = evaluate_predictions(val_yhat, y, val_mask, age_days)

    print(f"[RSR-v2] TEST ic={test_metrics['ic']:+.4f} "
          f"rank_ic={test_metrics['rank_ic']:+.4f} "
          f"ndcg10={test_metrics['ndcg10']:.4f} "
          f"ndcg50={test_metrics['ndcg50']:.4f}")

    config_dict = asdict(cfg)
    config_dict["graph_info"] = graph_info
    out_path = save_result(
        out_dir=Path(cfg.output_dir),
        fold=cfg.fold, seed=cfg.seed,
        model_name="RSR (v2 protocol)",
        test_metrics=test_metrics,
        val_metrics=val_metrics_final,
        test_y_hat=test_yhat,
        test_eval_mask=test_mask,
        history=history,
        config=config_dict,
        n_panel=(T, N, Fdim),
        n_train=len(train_idx), n_val=len(val_idx), n_test=len(test_idx),
        y_true=y, tickers=tickers, dates=dates,
        age_days=age_days, tradable_mask=tradable,
    )
    print(f"[RSR-v2] wrote {out_path}")


if __name__ == "__main__":
    main()
