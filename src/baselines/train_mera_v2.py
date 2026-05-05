"""MERA baseline trainer using the v2 protocol (matches RAG-STAR).

Same panel, masks, fold definitions, embargo, seeds, and metrics as
``src.v2.training.train_dow_epistar``. The only difference from
RAG-STAR is the model: this script wraps the vendored MERA
architecture (Liu, Song, Liu, Li, Dai, Bao, Jiang, Xia, WWW 2025).

Reference code source vendored at ``src/baselines/vendored/mera/``
is adapted from https://github.com/chenchen1104/MERA. The upstream
fmoe-based MoE block was rewritten in plain PyTorch (M=4 GRU experts,
Top-K=1) so this baseline runs in the same env as every other v2
baseline (MASTER, StockMixer, DySTAGE, FactorVAE, PatchTST). The
masked-autoencoder pre-training described in the paper is implemented
here; upstream only loads pre-trained weights from disk.

Two-phase training (per the paper):

  Phase 1: 1-2 epochs of masked-AE pre-training on training-fold
           windows. Random mask ratio 0.5 over T=20 timesteps. Stops
           early on reconstruction-loss plateau (3 consecutive non-
           improving epochs).
  Phase 2: Up to 10 epochs with early stopping on val IC (3-epoch
           patience). Backbone frozen. Retrieval pool is rebuilt at
           the start of Phase 2 from the post-Phase-1 backbone and
           kept fixed throughout Phase 2 (paper does not re-build).
           Loss = ``cs_mse_loss`` only. AdamW lr=1e-4,
           weight_decay=1e-5, 500-step warmup + cosine, fp16 autocast,
           gradient clip norm 1.0.

Run:
    python3 -m src.baselines.train_mera_v2 --fold 1 --seed 42

Smoke:
    python3 -m src.baselines.train_mera_v2 --fold 1 --seed 42 --max_epochs 1

Output: results/baselines_244/mera_v2/fold{F}_seed{S}.json (+ npz),
schema identical to FactorVAE's so ``compute_metrics.py`` produces
IC / rank IC / NDCG@10 / NDCG@50 with no special-casing.
"""
from __future__ import annotations

import argparse
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from src.baselines.mera_adapter import MERAAdapter, MERAHyperparams
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


# ---------------------------------------------------------------------------
# Config.
# ---------------------------------------------------------------------------


@dataclass
class MERAV2Config(V2BaselineConfig):
    """Top-level config bundling the v2 protocol + MERA-specific knobs.

    MERA hyperparameter rationale:

      - d_model=128, num_layers=2, num_heads=4 follow the WWW 2025
        paper's CSI 300/500/1000 default Transformer config and the
        upstream ``model_moe_attn.Transformer`` defaults.
      - n_classes=10 matches the paper's B=10 quantile bins for the
        label discretisation used by the retrieval label embedding.
      - top_n=10 matches the paper's TopN=10 nearest neighbours.
      - num_experts=4, top_k=1 are the paper's M and K defaults.
      - mask_ratio=0.5 follows the user's task brief (the paper says
        "random" without committing to a specific value; 0.5 is a
        common MAE / SimMTM default).
      - phase1_epochs=2 (paper says 1-2). We default to 2.
      - We additionally cap retrieval-pool size to 200 000 by
        downsampling if needed; with our ~240K training (day, ticker)
        pairs per fold this is a near no-op but keeps memory bounded.
    """
    output_dir: str = "results/baselines_244/mera_v2"
    # MERA architecture.
    d_model: int = 128
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.1
    n_classes: int = 10
    top_n: int = 10
    d_label: int = 16
    num_experts: int = 4
    top_k: int = 1
    expert_hidden: int = 64
    # Phase-1 self-supervised pre-training.
    # Default 8 epochs (was 2) so the frozen Transformer backbone produces
    # richer features for Phase 2 on our smaller 244-ticker / ~990-day
    # panel. Paper used CSI 300/500/1000 with much more data.
    phase1_epochs: int = 8
    phase1_patience: int = 3
    mask_ratio: float = 0.5
    # Phase-2 specific overrides. Backbone is frozen and the SMoE + head
    # is a small param count, so the v2 default lr=1e-4 / wd=1e-5 / patience=3
    # overfits within one epoch on our panel. We lower lr to 3e-5, raise
    # weight_decay to 1e-4 and drop patience to 1 (val IC peaks at epoch 0).
    phase2_lr: float = 3e-5
    phase2_weight_decay: float = 1e-4
    phase2_patience: int = 1
    # Retrieval pool.
    pool_max_size: int = 200_000
    retrieval_backend: str = "faiss"   # 'faiss' or 'sklearn'


# ---------------------------------------------------------------------------
# Retrieval pool helpers.
# ---------------------------------------------------------------------------


def _quantile_bin(y: np.ndarray, n_classes: int) -> np.ndarray:
    """Return per-sample integer class index by quantile binning.

    Ties broken by ``np.quantile`` -> linear interpolation; we map
    each y to ``floor(rank / N * n_classes)`` clipped to
    ``[0, n_classes - 1]`` which is robust to repeat values.
    """
    if y.size == 0:
        return np.zeros(0, dtype=np.int64)
    order = np.argsort(np.argsort(y))             # 0..N-1 ranks
    bins = np.floor(order * n_classes / max(1, len(y))).astype(np.int64)
    return np.clip(bins, 0, n_classes - 1)


def _try_build_faiss(dim: int, vectors: np.ndarray):
    """Return a flat L2 FAISS index, or None if FAISS unavailable."""
    try:
        import faiss  # type: ignore
    except Exception:
        return None
    index = faiss.IndexFlatL2(dim)
    index.add(np.ascontiguousarray(vectors.astype(np.float32)))
    return index


def _retrieve_topn_faiss(index, queries: np.ndarray, k: int) -> np.ndarray:
    """Return (B, k) int64 neighbour indices using a FAISS flat-L2 index."""
    qs = np.ascontiguousarray(queries.astype(np.float32))
    _, ids = index.search(qs, k)
    return ids.astype(np.int64)


def _retrieve_topn_sklearn(pool: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    """Sklearn fallback for environments without FAISS."""
    from sklearn.neighbors import NearestNeighbors  # type: ignore
    nn = NearestNeighbors(n_neighbors=k, algorithm="auto", metric="euclidean")
    nn.fit(pool)
    _, ids = nn.kneighbors(queries, n_neighbors=k, return_distance=True)
    return ids.astype(np.int64)


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, choices=[1, 2, 3], required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true",
                   help="Limit to 2 epochs and abbreviated output.")
    p.add_argument("--max_epochs", type=int, default=None,
                   help="Override config.epochs (e.g. 1 for a smoke check).")
    p.add_argument("--phase1_epochs", type=int, default=None,
                   help="Override config.phase1_epochs (default 8).")
    p.add_argument("--phase2_lr", type=float, default=None,
                   help="Override Phase 2 learning rate (default 3e-5). "
                        "Phase 1 always uses the v2 default lr=1e-4.")
    p.add_argument("--phase2_patience", type=int, default=None,
                   help="Override Phase 2 early-stop patience (default 1).")
    p.add_argument("--phase2_weight_decay", type=float, default=None,
                   help="Override Phase 2 weight decay (default 1e-4).")
    args = p.parse_args()

    cfg = MERAV2Config(fold=args.fold, seed=args.seed)
    if args.smoke:
        cfg.epochs = 2
        cfg.phase1_epochs = 1
    if args.max_epochs is not None:
        cfg.epochs = int(args.max_epochs)
    if args.phase1_epochs is not None:
        cfg.phase1_epochs = int(args.phase1_epochs)
    if args.phase2_lr is not None:
        cfg.phase2_lr = float(args.phase2_lr)
    if args.phase2_patience is not None:
        cfg.phase2_patience = int(args.phase2_patience)
    if args.phase2_weight_decay is not None:
        cfg.phase2_weight_decay = float(args.phase2_weight_decay)
    # NB: do NOT clamp phase1_epochs to epochs. Phase 1 is masked-AE
    # pretraining and is independent of Phase 2's epoch budget. A
    # smoke run with --max_epochs 1 --phase1_epochs 5 still gets 5
    # Phase-1 epochs + 1 Phase-2 epoch, as intended.

    set_seeds(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[MERA-v2] fold={cfg.fold} seed={cfg.seed} device={device}")

    x_raw, y, tickers, dates = build_panel(cfg)
    T, N, Fdim = x_raw.shape
    print(f"[MERA-v2] panel: T={T} N={N} F={Fdim}")
    if N < 50:
        raise RuntimeError("Panel too small")

    mm = build_masks(cfg, dates, tickers)
    tradable = mm["tradable_mask"]
    loss_mask = mm["loss_mask"]
    hist20 = mm["history_valid_20d"]
    hist60 = mm["history_valid_60d"]

    train_idx, val_idx, test_idx = fold_split(cfg, dates)
    print(f"[MERA-v2] fold {cfg.fold}: "
          f"train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    x = standardize_features(x_raw, tradable, train_idx)
    age_feat = build_age_features(tradable, hist20, hist60)
    age_days = age_feat[..., 0].astype(np.int64)

    x_t = torch.from_numpy(x).to(device)
    y_t = torch.from_numpy(y).to(device)

    W = cfg.temporal_window

    # -- Build the model -------------------------------------------------------
    hp = MERAHyperparams(
        d_feat=Fdim,
        context_window=W,
        d_model=cfg.d_model,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        dropout=cfg.dropout,
        n_classes=cfg.n_classes,
        top_n=cfg.top_n,
        d_label=cfg.d_label,
        num_experts=cfg.num_experts,
        top_k=cfg.top_k,
        expert_hidden=cfg.expert_hidden,
        mask_ratio=cfg.mask_ratio,
    )
    model = MERAAdapter(hp).to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # =========================================================================
    # Phase 1: masked-AE pre-training on the backbone.
    # =========================================================================
    print(f"[MERA-v2] Phase 1: masked-AE pre-training "
          f"(epochs={cfg.phase1_epochs}, mask_ratio={cfg.mask_ratio})")

    pre_optim = torch.optim.AdamW(
        list(model.backbone.parameters()) + list(model.mae_head.decoder.parameters())
        + [model.mae_head.mask_token],
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    best_pre_loss = float("inf")
    pre_patience = 0
    for epoch in range(cfg.phase1_epochs):
        t0 = time.time()
        model.train()
        np.random.seed(cfg.seed + epoch)
        perm = np.random.permutation(train_idx)
        losses = []
        for t in perm:
            t = int(t)
            if t < W - 1:
                continue
            m_np = tradable[t]
            if m_np.sum() < 8:
                continue
            active_idx = np.flatnonzero(m_np)
            active_t = torch.from_numpy(active_idx).to(device)
            x_win = x_t[t - W + 1: t + 1, active_t, :].transpose(0, 1)  # (B, T, F)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                rec_loss, _ = model.masked_ae_loss(x_win)

            pre_optim.zero_grad()
            scaler.scale(rec_loss).backward()
            scaler.unscale_(pre_optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(pre_optim)
            scaler.update()
            losses.append(float(rec_loss.item()))

        epoch_loss = float(np.mean(losses)) if losses else float("nan")
        dt = time.time() - t0
        improved = epoch_loss < best_pre_loss - 1e-5
        print(f"[MERA-v2]   pre-epoch {epoch}: rec_loss={epoch_loss:.5f} "
              f"({dt:.1f}s)" + ("  *best*" if improved else ""))
        if improved:
            best_pre_loss = epoch_loss
            pre_patience = 0
        else:
            pre_patience += 1
            if pre_patience >= cfg.phase1_patience:
                print(f"[MERA-v2]   Phase-1 early stop at epoch {epoch}")
                break

    # =========================================================================
    # Build retrieval pool from train-fold (day, ticker) pairs.
    # =========================================================================
    # Freeze backbone now (Phase 2 trains only aggregator/SMoE/head).
    for prm in model.backbone.parameters():
        prm.requires_grad_(False)

    print(f"[MERA-v2] Building retrieval pool from train fold "
          f"(top_n={cfg.top_n}, n_classes={cfg.n_classes})")
    pool_keys: list[np.ndarray] = []
    pool_labels: list[np.ndarray] = []
    pool_dayidx: list[np.ndarray] = []
    pool_tickeridx: list[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for t in train_idx:
            t = int(t)
            if t < W - 1:
                continue
            m_np = tradable[t] & loss_mask[t]   # need a label to bin
            if m_np.sum() < 2:
                continue
            active_idx = np.flatnonzero(m_np)
            active_t = torch.from_numpy(active_idx).to(device)
            x_win = x_t[t - W + 1: t + 1, active_t, :].transpose(0, 1)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                emb = model.backbone_embed(x_win)              # (b, d_model)
            emb_np = emb.float().cpu().numpy()
            y_active = y[t, active_idx].astype(np.float32)
            classes = _quantile_bin(y_active, cfg.n_classes)
            pool_keys.append(emb_np)
            pool_labels.append(classes)
            pool_dayidx.append(np.full(emb_np.shape[0], t, dtype=np.int64))
            pool_tickeridx.append(active_idx.astype(np.int64))

    pool_keys_np = np.concatenate(pool_keys, axis=0) if pool_keys else np.zeros((0, cfg.d_model), dtype=np.float32)
    pool_labels_np = np.concatenate(pool_labels, axis=0) if pool_labels else np.zeros((0,), dtype=np.int64)
    pool_dayidx_np = np.concatenate(pool_dayidx, axis=0) if pool_dayidx else np.zeros((0,), dtype=np.int64)
    print(f"[MERA-v2] pool size: {pool_keys_np.shape[0]} entries, dim={pool_keys_np.shape[1]}")

    # Cap pool size if needed (uniform random downsample by pool index;
    # keeps the (day, ticker) coverage roughly proportional).
    if pool_keys_np.shape[0] > cfg.pool_max_size:
        rng = np.random.default_rng(cfg.seed)
        keep = rng.choice(pool_keys_np.shape[0], size=cfg.pool_max_size, replace=False)
        keep.sort()
        pool_keys_np = pool_keys_np[keep]
        pool_labels_np = pool_labels_np[keep]
        pool_dayidx_np = pool_dayidx_np[keep]
        print(f"[MERA-v2] downsampled pool to {pool_keys_np.shape[0]}")

    if pool_keys_np.shape[0] < max(cfg.top_n, 8):
        raise RuntimeError(
            f"Retrieval pool too small ({pool_keys_np.shape[0]}) "
            f"for top_n={cfg.top_n}"
        )

    # Build retrieval index. FAISS preferred for speed; sklearn fallback.
    faiss_index = None
    if cfg.retrieval_backend == "faiss":
        faiss_index = _try_build_faiss(cfg.d_model, pool_keys_np)
        if faiss_index is None:
            print("[MERA-v2] FAISS unavailable; falling back to sklearn NearestNeighbors")
    if faiss_index is None:
        # Sklearn fallback: just keep the raw pool array; we call
        # _retrieve_topn_sklearn at query time. Building a single
        # NearestNeighbors object once and reusing it is faster.
        from sklearn.neighbors import NearestNeighbors  # type: ignore
        sk_nn = NearestNeighbors(n_neighbors=cfg.top_n, algorithm="auto", metric="euclidean")
        sk_nn.fit(pool_keys_np)
    else:
        sk_nn = None

    pool_keys_t = torch.from_numpy(pool_keys_np).to(device)         # (P, d_model)
    pool_labels_t = torch.from_numpy(pool_labels_np).to(device)     # (P,)

    def retrieve(query_emb_np: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """Run TopN retrieval, gather pool entries to GPU tensors.

        Args:
            query_emb_np: (B, d_model) numpy.
        Returns:
            (retrieved_feat, retrieved_class):
              retrieved_feat:  (B, top_n, d_model) float on device.
              retrieved_class: (B, top_n) long on device.
        """
        if faiss_index is not None:
            ids = _retrieve_topn_faiss(faiss_index, query_emb_np, cfg.top_n)
        else:
            _, ids = sk_nn.kneighbors(query_emb_np, n_neighbors=cfg.top_n, return_distance=True)
            ids = ids.astype(np.int64)
        ids_t = torch.from_numpy(ids).to(device)
        feats = pool_keys_t[ids_t]              # (B, top_n, d_model)
        classes = pool_labels_t[ids_t]          # (B, top_n)
        return feats, classes

    # =========================================================================
    # Phase 2: train aggregator + SMoE + predict head with cs_mse_loss.
    # =========================================================================
    print(f"[MERA-v2] Phase 2: train aggregator + SMoE + head "
          f"(epochs={cfg.epochs}, frozen backbone, "
          f"lr={cfg.phase2_lr:.2e}, wd={cfg.phase2_weight_decay:.2e}, "
          f"patience={cfg.phase2_patience})")

    optim = torch.optim.AdamW(
        list(model.phase2_parameters()),
        lr=cfg.phase2_lr,
        weight_decay=cfg.phase2_weight_decay,
    )
    total_steps = cfg.epochs * max(1, len(train_idx))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optim, lr_lambda=lambda s: warmup_cosine_lr(s, cfg.warmup_steps, total_steps)
    )

    def run_split(idx: np.ndarray, train_: bool) -> tuple[float, np.ndarray, np.ndarray]:
        # Backbone is frozen and stays in eval mode for stable BN stats;
        # the rest of the model toggles train/eval normally.
        model.aggregator.train(train_)
        model.smoe.train(train_)
        model.predict_head.train(train_)
        model.backbone.eval()

        losses = []
        y_hat_all = np.zeros((T, N), dtype=np.float32)
        emask = np.zeros((T, N), dtype=bool)
        for t in idx:
            t = int(t)
            if t < W - 1:
                continue
            m_np = tradable[t]
            if m_np.sum() < 3:
                continue
            active_idx = np.flatnonzero(m_np)
            active_t = torch.from_numpy(active_idx).to(device)
            x_win = x_t[t - W + 1: t + 1, active_t, :].transpose(0, 1)  # (B, T, F)
            y_target_full = y_t[t]
            lmask_t = torch.from_numpy(loss_mask[t]).to(device)

            # Build query embeddings under the frozen backbone, retrieve.
            with torch.no_grad(), torch.amp.autocast(
                "cuda", enabled=(device.type == "cuda")
            ):
                q_emb = model.backbone_embed(x_win)
            q_emb_np = q_emb.float().cpu().numpy()
            retrieved_feat, retrieved_class = retrieve(q_emb_np)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                y_hat_active = model(x_win, retrieved_feat, retrieved_class)  # (B,)
                y_full = torch.zeros(N, device=device, dtype=y_hat_active.dtype)
                y_full[active_t] = y_hat_active
                cs_loss = cs_mse_loss(y_full, y_target_full, lmask_t)

            if train_:
                optim.zero_grad()
                scaler.scale(cs_loss).backward()
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.phase2_parameters()], cfg.grad_clip
                )
                scaler.step(optim)
                scaler.update()
                scheduler.step()
            losses.append(float(cs_loss.item()))
            y_hat_all[t] = y_full.detach().float().cpu().numpy()
            emask[t] = loss_mask[t]
        return (float(np.mean(losses)) if losses else float("nan"),
                y_hat_all, emask)

    history: list = []
    best_val_ic = -1e9
    best_state = None
    patience = 0
    val_yhat = None
    val_mask = None
    for epoch in range(cfg.epochs):
        t0 = time.time()
        np.random.seed(cfg.seed + 1000 + epoch)
        perm = np.random.permutation(train_idx)
        train_loss, _, _ = run_split(perm, train_=True)
        val_loss, val_yhat, val_mask = run_split(val_idx, train_=False)
        val_metrics = evaluate_predictions(val_yhat, y, val_mask, age_days)
        dt = time.time() - t0
        improved = val_metrics["ic"] > best_val_ic + 1e-5
        print(f"[MERA-v2] epoch {epoch}: train_loss={train_loss:.4f} "
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
            if patience >= cfg.phase2_patience:
                print(f"[MERA-v2] early stop epoch {epoch} "
                      f"best_val_ic={best_val_ic:+.4f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    _, test_yhat, test_mask = run_split(test_idx, train_=False)
    test_metrics = evaluate_predictions(test_yhat, y, test_mask, age_days)
    val_metrics_final = evaluate_predictions(val_yhat, y, val_mask, age_days)

    print(f"[MERA-v2] TEST ic={test_metrics['ic']:+.4f} "
          f"rank_ic={test_metrics['rank_ic']:+.4f} "
          f"ndcg10={test_metrics['ndcg10']:.4f} "
          f"ndcg50={test_metrics['ndcg50']:.4f}")

    out_path = save_result(
        out_dir=Path(cfg.output_dir),
        fold=cfg.fold, seed=cfg.seed,
        model_name="MERA-v2",
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
    print(f"[MERA-v2] wrote {out_path}")


if __name__ == "__main__":
    main()
