"""Two-stage fold-causal CONTRASTIVE pretrain -> finetune for the
canonical BANKLESS InVAR (EXPERIMENT-ONLY; never in any paper).

NOT a paper baseline. This trainer is a COPY of
``src.baselines.train_invar_pretrain_v2`` with EXACTLY ONE change: the
Stage-1 self-supervised objective is REGIME-CONTRASTIVE instead of
masked-window reconstruction. Everything else (the fold-causal corpus
restriction ``pretrain_idx = fold_split(cfg, dates)[0]``, the
``_assert_pretrain_causal`` hard leakage guard, train-fold
standardisation, the Stage-2 finetune that loads the fold encoder ckpt
into the canonical BANKLESS InVAR with a layer-wise LR + two-regime val
+ SWA + cs_mse_loss, the JSON-only disk-safe write, the argparse, the
bankless asserts, the encoder-ckpt path/key convention) is reused
BYTE-IDENTICAL so the leakage discipline is preserved and the new bug
surface is the Stage-1 loss only.

Design
------
Stage 1 (REGIME-CONTRASTIVE SSL of the per-ticker temporal encoder):
    A small pretrain wrapper holds the SAME ``PerTickerTemporalEncoder``
    submodules as the canonical model plus a SimCLR-style 2-layer MLP
    projection head (``d -> d -> proj_dim``). For each training day t
    (restricted to ``pretrain_idx`` ONLY) the encoder runs over that
    day's active-ticker ``(N, T, F)`` windows; the per-ticker ``(N, d)``
    last-step outputs are masked-mean-pooled (active tickers only) into
    a single day embedding, projected through the head, and
    L2-normalised -> ``z_t``.

    POSITIVES: each training day carries the SAME 14-d regime
    fingerprint that the day-memory uses (8-dim risk + 6-dim
    cross-sectional diagnostics from ``build_episode_keys``, built
    causally from the panel and restricted to ``pretrain_idx`` training
    days; standardised with TRAIN-day stats only). Within a minibatch of
    training days the positives of an anchor day are the ``P`` other
    in-batch days whose standardised regime key is nearest (L2) to the
    anchor's, ``P = ceil(pos_frac * batch)``. NEGATIVES are the other
    in-batch days. The loss is a supervised-contrastive / InfoNCE
    objective with cosine similarity and temperature ``tau``:
        L = -mean_i log( sum_{p in pos(i)} exp(sim(i,p)/tau)
                          / sum_{a != i} exp(sim(i,a)/tau) ).
    The temporal encoder + projection head are trained with AdamW +
    ``warmup_cosine_lr`` (fp16 ok). ``cfg.pretrain_epochs`` (default 10).

    LEAKAGE: the pretrain corpus is EXACTLY the fold's training days
    ``train_idx = fold_split(cfg, dates)[0]``; ``val_idx`` /
    ``test_idx`` are never read. The regime fingerprint is standardised
    with TRAIN-day stats only and the (optional) k-means clustering is
    fit on TRAIN days only. ``_assert_pretrain_causal`` enforces the
    subset/disjoint invariant, and every day index used inside Stage-1
    is asserted to lie in ``pretrain_idx``.

    Only the temporal-encoder ``state_dict`` is saved (the projection
    head is discarded) to the SAME fold-keyed checkpoint
    ``results/<out>/_ckpt/foldF_encoder.pt`` with the SAME key
    convention as the masked-recon trainer, so Stage-2's strict load is
    unchanged.

Stage 2 (finetune the full BANKLESS InVAR per (fold, seed)):
    BYTE-IDENTICAL to ``train_invar_pretrain_v2.run_stage2_finetune``
    (same v2_runner calls, same SWA EMA loop, same early-stop, same JSON
    schema, same strict ckpt load, same layer-wise LR).

Disk: experiment-only. Stage 2 writes ONLY the
``fold{F}_seed{S}.json`` (history entries contain ``"epoch"`` so the
sbatch skip-if-done test passes). NO predictions npz. Stage 1 writes
ONLY the small encoder checkpoint .pt.

Run (1-fold smoke):
    # Stage 1: contrastive-pretrain fold 1 (1 epoch), save ckpt, exit.
    python -m src.baselines.train_invar_clpretrain_v2 --fold 1 --seed 42 \
        --panel_kind lattice_native --two_regime_val \
        --pretrain_only --pretrain_epochs 1 \
        --output_dir results/invar_clpretrain
    # Stage 2: finetune fold 1 seed 42 (1 epoch) from the checkpoint.
    python -m src.baselines.train_invar_clpretrain_v2 --fold 1 --seed 42 \
        --panel_kind lattice_native --two_regime_val \
        --skip_pretrain --finetune_epochs 1 \
        --output_dir results/invar_clpretrain
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn

from src.baselines.v2_runner import (
    build_age_features,
    build_masks,
    build_panel,
    cs_mse_loss,
    evaluate_predictions,
    fold_split,
    set_seeds,
    standardize_features,
    warmup_cosine_lr,
)
from src.baselines.train_invar_stx_v2 import (
    InvarSTXModel,
    InvarSTXV2Config,
    PerTickerTemporalEncoder,
)
from src.v2.data.episode_keys import (
    EPISODE_KEY_COLS,
    EpisodeKeyConfig,
    build_episode_keys,
)
from src.v2.data.macro_duration_features import (
    MACRO_GATE_COLS,
    build_macro_duration_features,
    standardize_macro_duration,
)
from src.v2.data.rolling_macro_betas import (
    betas_to_tensor,
    build_rolling_betas,
)
from src.v2.training.train_dow_epistar import (
    resolve_duration_indices,
    _gather_or_zero,
)


# ============================================================================
# STAGE 1: REGIME-CONTRASTIVE self-supervised pretrain of the per-ticker
# temporal encoder.
# ============================================================================

# Contrastive-pretrain hyperparameters (Stage-1 only; the masked-recon
# trainer used a 0.5 mask ratio here instead). InfoNCE temperature, the
# SimCLR projection-head width, and the fraction of the in-batch days
# treated as positives for each anchor (nearest in standardised
# regime-fingerprint space).
CL_PROJ_DIM = 128
CL_TEMPERATURE = 0.1
CL_POS_FRAC = 0.1
CL_BATCH_DAYS = 64


class TemporalEncoderContrastivePretrainer(nn.Module):
    """Regime-contrastive (SimCLR / SupCon-style) wrapper around the
    canonical ``PerTickerTemporalEncoder``.

    Holds its OWN ``PerTickerTemporalEncoder`` instance (built with the
    SAME constructor args the finetune model will use, so its
    ``state_dict`` keys match ``model.temporal_encoder`` for the Stage-2
    strict load) plus a 2-layer MLP projection head
    (``d -> d -> proj_dim``). The pretext produces ONE projected,
    L2-normalised embedding per training day from that day's
    active-ticker windows; the contrastive loss lives outside this
    module. After pretrain, ``encoder.state_dict()`` is the artefact
    loaded into the finetune model's ``temporal_encoder`` submodule
    (strict key match); the projection head is discarded.
    """

    def __init__(
        self,
        n_features: int,
        temporal_window: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        e_layers: int,
        dropout: float,
        activation: str,
        proj_dim: int,
    ) -> None:
        super().__init__()
        self.encoder = PerTickerTemporalEncoder(
            n_features=n_features,
            temporal_window=temporal_window,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            e_layers=e_layers,
            dropout=dropout,
            activation=activation,
        )
        # SimCLR-style 2-layer projection head (d -> d -> proj_dim).
        self.proj_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, proj_dim),
        )

    def day_embedding(self, x_window: Tensor) -> Tensor:
        """Encode one day's active-ticker windows into ONE day vector.

        Args:
            x_window: ``(N_active, T, F)`` per-ticker lookback windows
                for the active tickers of a single training day.

        Returns:
            ``(proj_dim,)`` L2-normalised projected day embedding.
        """
        # Per-ticker last-step encodings via the canonical encoder
        # forward (returns (N_active, d_model); the active mask was
        # already applied by selecting active tickers upstream, so a
        # plain mean over the ticker axis is the masked mean-pool).
        per_ticker = self.encoder(x_window)                    # (N, d)
        day_vec = per_ticker.mean(dim=0)                       # (d,)
        z = self.proj_head(day_vec)                            # (proj_dim,)
        z = torch.nn.functional.normalize(z, dim=-1)
        return z


def _supcon_infonce_loss(
    z: Tensor,
    pos_mask: Tensor,
    tau: float,
) -> Tensor:
    """Supervised-contrastive / InfoNCE loss over a minibatch of days.

    Args:
        z: ``(B, proj_dim)`` L2-normalised day embeddings.
        pos_mask: ``(B, B)`` bool; ``pos_mask[i, j]`` True iff day j is a
            positive for anchor i (always False on the diagonal).
        tau: softmax temperature.

    Returns:
        Scalar loss
        ``-mean_i log( sum_{p in pos(i)} exp(sim(i,p)/tau)
                       / sum_{a != i} exp(sim(i,a)/tau) )``
        with ``sim`` = cosine similarity (z is already L2-normalised so
        ``z @ z.T`` is cosine). Anchors with no positive are skipped.
    """
    b = z.shape[0]
    sim = (z @ z.t()) / max(tau, 1e-6)                         # (B, B)
    self_mask = torch.eye(b, dtype=torch.bool, device=z.device)
    # Numerically-stable log-sum-exp over all non-self entries.
    sim_masked = sim.masked_fill(self_mask, float("-inf"))
    logits_max = sim_masked.max(dim=1, keepdim=True).values.detach()
    exp_sim = torch.exp(sim_masked - logits_max)
    denom = exp_sim.sum(dim=1)                                 # (B,)
    log_prob = (sim - logits_max.squeeze(1).unsqueeze(1)
                - torch.log(denom.clamp_min(1e-12)).unsqueeze(1))
    pos = pos_mask & (~self_mask)
    pos_counts = pos.sum(dim=1)                                # (B,)
    valid = pos_counts > 0
    if not bool(valid.any()):
        return torch.zeros((), device=z.device, dtype=z.dtype)
    # Mean log-likelihood of the positives per anchor, averaged over
    # anchors that actually have at least one positive.
    pos_log_prob = (log_prob * pos.float()).sum(dim=1)
    per_anchor = pos_log_prob[valid] / pos_counts[valid].clamp_min(1).float()
    return -per_anchor.mean()


def run_stage1_pretrain(
    cfg: InvarSTXV2Config,
    pretrain_epochs: int,
    device: torch.device,
    ckpt_path: Path,
) -> None:
    """Fold-causal REGIME-CONTRASTIVE self-supervised pretrain of the
    temporal encoder.

    The pretrain corpus is restricted to ``train_idx`` ONLY (the fold's
    training days). ``val_idx`` / ``test_idx`` are never read; the
    regime fingerprint is standardised with TRAIN-day stats only; an
    explicit leakage assertion plus a per-day in-corpus assertion
    enforce this.
    """
    set_seeds(cfg.seed)

    # ---- v2_runner data / fold calls: SAME args, SAME order as
    # train_invar_pretrain_v2.py / train_invar_stx_v2.py. ----
    x_raw, y, tickers, dates = build_panel(cfg)
    T, N, Fdim = x_raw.shape
    print(f"[InVAR-clpretrain S1] panel: T={T} N={N} F={Fdim}")
    if N < 50:
        raise RuntimeError("Panel too small")

    mm = build_masks(cfg, dates, tickers)
    tradable = mm["tradable_mask"]
    loss_mask = mm["loss_mask"]

    train_idx, val_idx, test_idx = fold_split(cfg, dates)
    print(f"[InVAR-clpretrain S1] fold {cfg.fold}: "
          f"train={len(train_idx)} val={len(val_idx)} "
          f"test={len(test_idx)}")

    # ---- LEAKAGE GUARD. The pretrain corpus is EXACTLY train_idx. ----
    pretrain_idx = np.asarray(train_idx).astype(np.int64)   # corpus == train_idx
    _assert_pretrain_causal(pretrain_idx, train_idx, val_idx, test_idx)
    pretrain_set = set(int(i) for i in pretrain_idx.tolist())

    # Train-fold standardisation stats only (val/test never used here).
    x = standardize_features(x_raw, tradable, train_idx)
    x_t = torch.from_numpy(x).to(device)

    # ---- 14-d regime fingerprint: the SAME key the day-memory uses
    # (8 risk + 6 cross-sectional diagnostics). Built causally from the
    # panel exactly as train_invar_pretrain_v2 / episode_keys do, then
    # STANDARDISED WITH TRAIN-DAY STATS ONLY (val/test never touched).
    day_keys, _ = build_episode_keys(
        dates=dates, log_returns=x_raw[..., 0], mask=tradable,
        cfg=EpisodeKeyConfig(),
    )
    key_tr = day_keys[train_idx]
    key_mu = key_tr.mean(axis=0)
    key_sd = key_tr.std(axis=0)
    key_sd = np.where(key_sd < 1e-6, 1.0, key_sd)
    day_keys_z = ((day_keys - key_mu) / key_sd).astype(np.float32)
    print(f"[InVAR-clpretrain S1] regime fingerprint: "
          f"{day_keys_z.shape[1]} dims ({len(EPISODE_KEY_COLS)} cols), "
          f"train-day standardised")

    W = cfg.temporal_window
    model = TemporalEncoderContrastivePretrainer(
        n_features=Fdim,
        temporal_window=W,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        d_ff=cfg.d_ff,
        e_layers=cfg.temporal_e_layers,
        dropout=cfg.dropout,
        activation=cfg.activation,
        proj_dim=CL_PROJ_DIM,
    ).to(device)

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    # Days usable as anchors: in the pretrain corpus, with a full
    # lookback window and >=3 active tickers.
    valid_days = [
        int(t) for t in pretrain_idx
        if int(t) >= W - 1 and tradable[int(t)].sum() >= 3
    ]
    batch_days = max(2, int(CL_BATCH_DAYS))
    steps_per_epoch = max(1, len(valid_days) // batch_days)
    total_steps = pretrain_epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optim,
        lr_lambda=lambda s: warmup_cosine_lr(
            s, cfg.warmup_steps, total_steps
        ),
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=(device.type == "cuda")
    )
    n_pos = max(1, int(np.ceil(CL_POS_FRAC * batch_days)))
    print(f"[InVAR-clpretrain S1] {len(valid_days)} usable train days, "
          f"batch={batch_days}, pos/anchor={n_pos}, tau={CL_TEMPERATURE}")

    model.train()
    for epoch in range(pretrain_epochs):
        t0 = time.time()
        rng = np.random.RandomState(cfg.seed + epoch)
        perm = rng.permutation(np.asarray(valid_days, dtype=np.int64))
        losses = []
        for b0 in range(0, len(perm) - batch_days + 1, batch_days):
            batch = perm[b0: b0 + batch_days]
            # ---- LEAKAGE ASSERT: every Stage-1 day index is in the
            # pretrain (train) corpus; never val/test. ----
            for _t in batch:
                if int(_t) not in pretrain_set:
                    raise RuntimeError(
                        f"LEAKAGE: Stage-1 used day {int(_t)} not in "
                        "pretrain_idx (train corpus)."
                    )

            # Per-day standardised regime fingerprints for this batch.
            keys_b = torch.from_numpy(
                day_keys_z[batch]
            ).float().to(device)                               # (B, 14)
            # Positives = the n_pos nearest in-batch days by L2 in the
            # standardised regime-fingerprint space (self excluded).
            with torch.no_grad():
                kd = torch.cdist(keys_b, keys_b)               # (B, B)
                bb = kd.shape[0]
                eye = torch.eye(
                    bb, dtype=torch.bool, device=device
                )
                kd = kd.masked_fill(eye, float("inf"))
                k = min(n_pos, bb - 1)
                nn_idx = torch.topk(
                    kd, k=k, dim=1, largest=False
                ).indices                                      # (B, k)
                pos_mask = torch.zeros(
                    bb, bb, dtype=torch.bool, device=device
                )
                pos_mask.scatter_(1, nn_idx, True)
                pos_mask = pos_mask & (~eye)

            with torch.amp.autocast(
                "cuda", enabled=(device.type == "cuda")
            ):
                z_list = []
                for _t in batch:
                    t = int(_t)
                    m_np = tradable[t]
                    active_idx = np.flatnonzero(m_np)
                    active_t = torch.from_numpy(active_idx).to(device)
                    # (N_active, T, F) lookback window (SAME slicing as
                    # the finetune loop in run_split).
                    x_win = x_t[
                        t - W + 1: t + 1, active_t, :
                    ].transpose(0, 1)
                    z_list.append(model.day_embedding(x_win))
                z = torch.stack(z_list, dim=0)                 # (B, proj)
                cl_loss = _supcon_infonce_loss(
                    z, pos_mask, CL_TEMPERATURE
                )

            optim.zero_grad()
            scaler.scale(cl_loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.grad_clip
            )
            scaler.step(optim)
            scaler.update()
            scheduler.step()
            losses.append(float(cl_loss.item()))
        dt = time.time() - t0
        mean_loss = float(np.mean(losses)) if losses else float("nan")
        print(f"[InVAR-clpretrain S1] epoch {epoch}: "
              f"infonce={mean_loss:.5f} "
              f"({len(losses)} batches, {dt:.1f}s)")

    # ---- Save the temporal-encoder state_dict ONLY (the projection
    # head is discarded). SAME key/path convention as the masked-recon
    # trainer so Stage-2's strict load works unchanged. ----
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "fold": cfg.fold,
            "seed": cfg.seed,
            "pretrain_epochs": pretrain_epochs,
            "panel_kind": cfg.panel_kind,
            "encoder_state_dict": model.encoder.state_dict(),
        },
        ckpt_path,
    )
    print(f"[InVAR-clpretrain S1] saved encoder ckpt -> {ckpt_path}")


def _assert_pretrain_causal(
    pretrain_idx: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> None:
    """Hard leakage guard: pretrain corpus must be a subset of
    ``train_idx`` and disjoint from ``val_idx | test_idx``."""
    p = set(int(i) for i in np.asarray(pretrain_idx).tolist())
    tr = set(int(i) for i in np.asarray(train_idx).tolist())
    va = set(int(i) for i in np.asarray(val_idx).tolist())
    te = set(int(i) for i in np.asarray(test_idx).tolist())
    if not p.issubset(tr):
        raise RuntimeError(
            "LEAKAGE: pretrain corpus is NOT a subset of train_idx "
            f"({len(p - tr)} day(s) outside train_idx)."
        )
    if p & va:
        raise RuntimeError(
            f"LEAKAGE: pretrain corpus intersects val_idx "
            f"({len(p & va)} day(s))."
        )
    if p & te:
        raise RuntimeError(
            f"LEAKAGE: pretrain corpus intersects test_idx "
            f"({len(p & te)} day(s))."
        )
    print(f"[InVAR-clpretrain] LEAKAGE-CHECK OK: |pretrain|={len(p)} "
          f"subset of |train|={len(tr)}; "
          f"intersect(val)={len(p & va)} intersect(test)={len(p & te)}")


# ============================================================================
# STAGE 2: finetune the full BANKLESS InVAR (canonical harness path).
# BYTE-IDENTICAL to train_invar_pretrain_v2.run_stage2_finetune.
# ============================================================================


def run_stage2_finetune(
    cfg: InvarSTXV2Config,
    finetune_epochs: int,
    device: torch.device,
    ckpt_path: Path,
) -> None:
    """Finetune the canonical bankless InVAR with the pretrained
    temporal encoder loaded in and a layer-wise LR.

    The data / fold / eval body is byte-identical to
    ``train_invar_stx_v2.main`` (same v2_runner calls, same SWA EMA loop,
    same early-stop, same JSON schema). The ONLY differences are:
    (1) the pretrained encoder weights are loaded into
    ``model.temporal_encoder`` with a strict key match + assertion,
    (2) two AdamW param groups give the pretrained encoder 0.25x LR.
    """
    cfg.epochs = int(finetune_epochs)
    if cfg.swa_warmup_epochs >= cfg.epochs:
        cfg.swa_warmup_epochs = max(0, cfg.epochs - 1)

    set_seeds(cfg.seed)
    print(f"[InVAR-clpretrain S2] fold={cfg.fold} seed={cfg.seed} "
          f"device={device}")

    # ---- v2_runner data / fold / eval calls: BYTE-IDENTICAL to
    # train_invar_stx_v2.py (same args, same order). ----
    x_raw, y, tickers, dates = build_panel(cfg)
    T, N, Fdim = x_raw.shape
    print(f"[InVAR-clpretrain S2] panel: T={T} N={N} F={Fdim}")
    if N < 50:
        raise RuntimeError("Panel too small")

    mm = build_masks(cfg, dates, tickers)
    tradable = mm["tradable_mask"]
    loss_mask = mm["loss_mask"]
    hist20 = mm["history_valid_20d"]
    hist60 = mm["history_valid_60d"]

    train_idx, val_idx, test_idx = fold_split(cfg, dates)
    print(f"[InVAR-clpretrain S2] fold {cfg.fold}: "
          f"train={len(train_idx)} val={len(val_idx)} "
          f"test={len(test_idx)}")

    x = standardize_features(x_raw, tradable, train_idx)
    age_feat = build_age_features(tradable, hist20, hist60)
    age_days = age_feat[..., 0].astype(np.int64)

    x_t = torch.from_numpy(x).to(device)
    y_t = torch.from_numpy(y).to(device)

    W = cfg.temporal_window

    # ---- Day-level regime memory keys + values (verbatim from
    # train_invar_stx_v2.py). ----
    day_keys, _ = build_episode_keys(
        dates=dates, log_returns=x_raw[..., 0], mask=tradable,
        cfg=EpisodeKeyConfig(),
    )
    feature_idx = [0, 1, 5, 6]
    n_summary = 2 * len(feature_idx) + 1
    day_values = np.zeros(
        (len(dates), day_keys.shape[1] + n_summary), dtype=np.float32,
    )
    day_values[:, : day_keys.shape[1]] = day_keys
    for t in range(len(dates)):
        m = tradable[t]
        if m.sum() < 5:
            continue
        for j, fi in enumerate(feature_idx):
            v = x_raw[t, m, fi]
            day_values[t, day_keys.shape[1] + 2 * j] = float(np.mean(v))
            day_values[t, day_keys.shape[1] + 2 * j + 1] = float(np.std(v))
        day_values[t, -1] = float(m.sum()) / 250.0
    cfg.day_value_dim = day_values.shape[1]

    # ---- Macro input + macro-gate input (verbatim). ----
    if cfg.panel_kind == "lattice_native":
        macro_path = Path(cfg.universal_macro_duration_parquet)
    else:
        macro_path = Path(cfg.biotech_macro_duration_parquet)
    if not macro_path.exists():
        print("[InVAR-clpretrain S2] macro parquet missing; building...")
        build_macro_duration_features()
    macro = pd.read_parquet(macro_path)
    macro_arr, macro_cols, _ = standardize_macro_duration(
        macro, dates, train_idx,
    )
    print(f"[InVAR-clpretrain S2] macro features: {len(macro_cols)} dims")

    gate_indices = [macro_cols.index(c) for c in MACRO_GATE_COLS
                    if c in macro_cols]
    if len(gate_indices) != len(MACRO_GATE_COLS):
        missing = [c for c in MACRO_GATE_COLS if c not in macro_cols]
        print(f"[InVAR-clpretrain S2] WARN missing gate cols: {missing}")
    macro_gate_macro = macro_arr[:, gate_indices].astype(np.float32)
    avg_corr_idx = EPISODE_KEY_COLS.index("cs_avg_pairwise_corr_60d")
    cs_disp_idx = EPISODE_KEY_COLS.index("cs_dispersion")
    avg_corr = day_keys[:, avg_corr_idx].astype(np.float32)
    cs_disp = day_keys[:, cs_disp_idx].astype(np.float32)
    avg_corr_tr = avg_corr[train_idx]
    cs_disp_tr = cs_disp[train_idx]
    avg_corr_z = ((avg_corr - avg_corr_tr.mean())
                  / max(avg_corr_tr.std(), 1e-6)).astype(np.float32)
    cs_disp_z = ((cs_disp - cs_disp_tr.mean())
                 / max(cs_disp_tr.std(), 1e-6)).astype(np.float32)
    macro_gate_arr = np.concatenate(
        [macro_gate_macro, avg_corr_z[:, None], cs_disp_z[:, None]],
        axis=1,
    ).astype(np.float32)
    print(f"[InVAR-clpretrain S2] macro_gate input: "
          f"{macro_gate_arr.shape[1]} dims")

    # ---- Per-ticker duration input (verbatim). ----
    duration_indices = resolve_duration_indices(cfg.panel_kind)
    duration_panel_block = _gather_or_zero(x, duration_indices).astype(
        np.float32
    )
    if cfg.panel_kind == "lattice_native":
        betas_path = Path(cfg.universal_rolling_betas_parquet)
    else:
        betas_path = Path(cfg.biotech_rolling_betas_parquet)
    if not betas_path.exists():
        print("[InVAR-clpretrain S2] rolling betas parquet missing; "
              "building...")
        build_rolling_betas()
        betas_path = Path(cfg.biotech_rolling_betas_parquet)
    betas_long = pd.read_parquet(betas_path)
    betas_tensor = betas_to_tensor(betas_long, dates, tickers)
    bt_train = betas_tensor[train_idx]
    train_mask = tradable[train_idx]
    betas_std = np.zeros_like(betas_tensor)
    for fi in range(betas_tensor.shape[-1]):
        vals = bt_train[..., fi][train_mask]
        if vals.size < 2:
            mu, sd = 0.0, 1.0
        else:
            mu = float(np.mean(vals))
            sd = float(np.std(vals))
            if sd < 1e-6:
                sd = 1.0
        betas_std[..., fi] = (betas_tensor[..., fi] - mu) / sd
    betas_std = (betas_std * tradable[..., None]).astype(np.float32)
    duration_input_full = np.concatenate(
        [duration_panel_block, age_feat, betas_std], axis=-1,
    ).astype(np.float32)
    duration_input_dim = duration_input_full.shape[-1]
    print(f"[InVAR-clpretrain S2] duration input dim: "
          f"{duration_input_dim}")

    # ---- Build the canonical BANKLESS InVAR EXACTLY as
    # train_invar_stx_v2.py (enable_retrieval_bank stays False). ----
    assert cfg.enable_retrieval_bank is False, (
        "Canonical InVAR is BANKLESS; enable_retrieval_bank must be "
        "False for the pretrain protocol."
    )
    model = InvarSTXModel(
        cfg,
        n_features=Fdim,
        day_key_dim=day_keys.shape[1],
        duration_input_dim=duration_input_dim,
        macro_input_dim=macro_arr.shape[1],
        macro_gate_in_dim=macro_gate_arr.shape[1],
    ).to(device)
    model.day_memory.populate(
        keys=day_keys, values=day_values,
        day_indices=np.arange(len(dates)),
        train_day_indices=train_idx,
    )
    model.day_memory.to(device)

    # ---- Load the fold's pretrained temporal-encoder weights with a
    # STRICT key match into model.temporal_encoder; assert it loaded. ----
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Pretrained encoder ckpt not found: {ckpt_path}. Run "
            "stage 1 (--pretrain_only) for this fold first."
        )
    ckpt = torch.load(ckpt_path, map_location=device)
    enc_state = ckpt["encoder_state_dict"]
    target_keys = set(model.temporal_encoder.state_dict().keys())
    ckpt_keys = set(enc_state.keys())
    if target_keys != ckpt_keys:
        raise RuntimeError(
            "Pretrained encoder key mismatch with model.temporal_encoder. "
            f"missing={sorted(target_keys - ckpt_keys)} "
            f"unexpected={sorted(ckpt_keys - target_keys)}"
        )
    incompat = model.temporal_encoder.load_state_dict(
        enc_state, strict=True
    )
    assert not incompat.missing_keys and not incompat.unexpected_keys, (
        f"strict load failed: {incompat}"
    )
    # Verify at least one parameter actually changed to the loaded value
    # (defends against a silent no-op load).
    a_name, a_param = next(iter(model.temporal_encoder.named_parameters()))
    assert torch.allclose(
        a_param.detach().cpu(),
        enc_state[a_name].detach().cpu().to(a_param.dtype),
    ), f"pretrained weights NOT loaded into temporal_encoder.{a_name}"
    print(f"[InVAR-clpretrain S2] loaded pretrained temporal encoder "
          f"({len(ckpt_keys)} tensors, fold={ckpt.get('fold')}, "
          f"strict key match OK) from {ckpt_path}")

    allowed_train = torch.from_numpy(train_idx).long().to(device)

    # ---- LAYER-WISE LR: pretrained temporal-encoder params at
    # 0.25 * base; everything else at base (two AdamW param groups). ----
    enc_param_ids = {
        id(p) for p in model.temporal_encoder.parameters()
    }
    enc_params = [p for p in model.parameters()
                  if id(p) in enc_param_ids]
    other_params = [p for p in model.parameters()
                    if id(p) not in enc_param_ids]
    base_lr = cfg.learning_rate
    optim = torch.optim.AdamW(
        [
            {"params": other_params, "lr": base_lr},
            {"params": enc_params, "lr": 0.25 * base_lr},
        ],
        lr=base_lr,
        weight_decay=cfg.weight_decay,
    )
    print(f"[InVAR-clpretrain S2] layer-wise LR: encoder={0.25 * base_lr:.2e} "
          f"({len(enc_params)} tensors) other={base_lr:.2e} "
          f"({len(other_params)} tensors)")
    total_steps = cfg.epochs * max(1, len(train_idx))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optim,
        lr_lambda=lambda s: warmup_cosine_lr(
            s, cfg.warmup_steps, total_steps
        ),
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=(device.type == "cuda")
    )

    macro_arr_t = macro_arr.astype(np.float32)
    macro_gate_t = macro_gate_arr.astype(np.float32)

    def run_split(idx: np.ndarray, train_: bool):
        """Run one pass over ``idx`` days. Byte-identical body to
        train_invar_stx_v2.run_split."""
        model.train(train_)
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
            x_win = x_t[t - W + 1: t + 1, active_t, :].transpose(0, 1)
            y_target_full = y_t[t]
            lmask_t = torch.from_numpy(loss_mask[t]).to(device)

            day_query_key = torch.from_numpy(
                day_keys[t]
            ).float().to(device)
            regime_scalars = model.day_memory.standardize_query(
                day_query_key
            )[[0, 9]].clone()
            if torch.isnan(regime_scalars).any():
                regime_scalars = torch.zeros(2, device=device)

            dur_in = torch.from_numpy(
                duration_input_full[t, active_idx]
            ).float().to(device)
            macro_in = torch.from_numpy(
                macro_arr_t[t]
            ).float().to(device)
            macro_gate_in = torch.from_numpy(
                macro_gate_t[t]
            ).float().to(device)

            with torch.amp.autocast(
                "cuda", enabled=(device.type == "cuda")
            ):
                y_hat_active = model(
                    x_win,
                    day_query_key=day_query_key,
                    query_day_idx=t,
                    allowed_day_indices=allowed_train,
                    regime_scalars=regime_scalars,
                    duration_input=dur_in,
                    macro_input=macro_in,
                    macro_gate_input=macro_gate_in,
                )
                y_full = torch.zeros(N, device=device,
                                     dtype=y_hat_active.dtype)
                y_full[active_t] = y_hat_active
                cs_loss = cs_mse_loss(y_full, y_target_full, lmask_t)

            if train_:
                optim.zero_grad()
                scaler.scale(cs_loss).backward()
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               cfg.grad_clip)
                scaler.step(optim)
                scaler.update()
                scheduler.step()
                _maybe_update_swa()
            losses.append(float(cs_loss.item()))
            y_hat_all[t] = y_full.detach().float().cpu().numpy()
            emask[t] = loss_mask[t]
        return (float(np.mean(losses)) if losses else float("nan"),
                y_hat_all, emask)

    # ---- SWA EMA state (byte-identical, incl. is_floating_point guard). ----
    ema_state: dict[str, torch.Tensor] | None = None
    swa_epoch_ref = {"epoch": 0}

    def _maybe_update_swa() -> None:
        nonlocal ema_state
        if not cfg.use_swa or swa_epoch_ref["epoch"] < cfg.swa_warmup_epochs:
            return
        with torch.no_grad():
            sd = model.state_dict()
            if ema_state is None:
                ema_state = {k: v.detach().clone() for k, v in sd.items()}
            else:
                d = float(cfg.swa_decay)
                for k in ema_state:
                    cur = sd[k].detach()
                    if torch.is_floating_point(ema_state[k]):
                        ema_state[k].mul_(d).add_(cur, alpha=1.0 - d)
                    else:
                        ema_state[k].copy_(cur)

    def _eval_split(idx: np.ndarray):
        if cfg.use_swa and ema_state is not None:
            saved = {k: v.detach().clone()
                     for k, v in model.state_dict().items()}
            model.load_state_dict(ema_state)
            res = run_split(idx, train_=False)
            model.load_state_dict(saved)
            return res
        return run_split(idx, train_=False)

    history: list = []
    best_val_ic = -1e9
    best_state = None
    patience = 0
    for epoch in range(cfg.epochs):
        t0 = time.time()
        swa_epoch_ref["epoch"] = epoch
        np.random.seed(cfg.seed + epoch)
        perm = np.random.permutation(train_idx)
        train_loss, _, _ = run_split(perm, train_=True)
        val_loss, val_yhat, val_mask = _eval_split(val_idx)
        val_metrics = evaluate_predictions(val_yhat, y, val_mask, age_days)
        dt = time.time() - t0
        improved = val_metrics["ic"] > best_val_ic + 1e-5
        print(f"[InVAR-clpretrain S2] epoch {epoch}: "
              f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"val_ic={val_metrics['ic']:+.4f} ({dt:.1f}s)"
              + ("  *best*" if improved else ""))
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
            src_state = (
                ema_state if (cfg.use_swa and ema_state is not None)
                else model.state_dict()
            )
            best_state = {k: v.detach().cpu().clone()
                          for k, v in src_state.items()}
            patience = 0
        else:
            patience += 1
            if patience >= cfg.early_stop_patience:
                print(f"[InVAR-clpretrain S2] early stop epoch {epoch} "
                      f"best_val_ic={best_val_ic:+.4f}")
                break

    if cfg.use_swa and ema_state is not None:
        final_state = {k: v.detach().cpu().clone()
                       for k, v in ema_state.items()}
        print("[InVAR-clpretrain S2] SWA: using final EMA state for test")
    elif best_state is not None:
        final_state = best_state
    else:
        final_state = {k: v.detach().cpu().clone()
                       for k, v in model.state_dict().items()}
    model.load_state_dict(final_state)

    _, test_yhat, test_mask = run_split(test_idx, train_=False)
    test_metrics = evaluate_predictions(test_yhat, y, test_mask, age_days)
    val_metrics_final = evaluate_predictions(
        val_yhat, y, val_mask, age_days
    )

    print(f"[InVAR-clpretrain S2] TEST ic={test_metrics['ic']:+.4f} "
          f"rank_ic={test_metrics['rank_ic']:+.4f} "
          f"ndcg10={test_metrics['ndcg10']:.4f} "
          f"ndcg50={test_metrics['ndcg50']:.4f}")

    # ---- Disk-safe write. NO predictions npz. Same JSON schema keys as
    # train_invar_stx_v2 (history entries contain "epoch" so the sbatch
    # skip-if-done test passes). ----
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"fold{cfg.fold}_seed{cfg.seed}.json"
    payload = {
        "fold": cfg.fold,
        "seed": cfg.seed,
        "model": "InVAR-clpretrain (v2 protocol)",
        "panel_T": int(T),
        "panel_N": int(N),
        "panel_F": int(Fdim),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "ic": test_metrics["ic"],
        "rank_ic": test_metrics["rank_ic"],
        "ndcg10": test_metrics["ndcg10"],
        "ndcg50": test_metrics["ndcg50"],
        "test_cohort_ic": test_metrics["cohort_ic"],
        "val_ic": val_metrics_final["ic"],
        "val_rank_ic": val_metrics_final["rank_ic"],
        "val_cohort_ic": val_metrics_final["cohort_ic"],
        "history": history,
        "config": asdict(cfg),
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"[InVAR-clpretrain S2] wrote {out_path} (no npz; disk-safe)")


def main() -> None:
    """CLI entry point. Two-stage contrastive-pretrain -> finetune for
    canonical BANKLESS InVAR."""
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, choices=[1, 2, 3, 4, 5],
                   required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--panel_kind", type=str, default="lattice_native",
                   choices=["biotech", "lattice_native"])
    p.add_argument("--two_regime_val", action="store_true")
    p.add_argument("--output_dir", type=str,
                   default="results/invar_clpretrain")
    p.add_argument("--panel_end", type=str, default=None)
    p.add_argument("--pretrain_epochs", type=int, default=10)
    p.add_argument("--finetune_epochs", type=int, default=10)
    p.add_argument("--pretrain_only", action="store_true",
                   help="Run stage 1 (build+save ckpt) then exit.")
    p.add_argument("--skip_pretrain", action="store_true",
                   help="Load the existing ckpt; finetune only.")
    args = p.parse_args()

    if args.pretrain_only and args.skip_pretrain:
        raise SystemExit(
            "--pretrain_only and --skip_pretrain are mutually exclusive."
        )

    # Default config = BANKLESS canonical (enable_retrieval_bank stays
    # at its False default; never enabled anywhere in this trainer).
    cfg = InvarSTXV2Config(fold=args.fold, seed=args.seed)
    cfg.panel_kind = args.panel_kind
    cfg.two_regime_val = args.two_regime_val
    cfg.output_dir = args.output_dir
    if args.panel_end:
        cfg.panel_end = args.panel_end
    elif args.panel_kind == "lattice_native":
        cfg.panel_end = "2025-12-31"
    assert cfg.enable_retrieval_bank is False, (
        "BANKLESS canonical invariant violated."
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"[InVAR-clpretrain] fold={cfg.fold} seed={cfg.seed} "
          f"panel={cfg.panel_kind} two_regime_val={cfg.two_regime_val} "
          f"device={device}")

    # Fold-keyed encoder checkpoint path (one pretrain per fold; shared
    # across all finetune seeds for that fold).
    ckpt_dir = Path(cfg.output_dir) / "_ckpt"
    ckpt_path = ckpt_dir / f"fold{cfg.fold}_encoder.pt"

    if args.skip_pretrain:
        run_stage2_finetune(
            cfg, args.finetune_epochs, device, ckpt_path
        )
        return

    # Stage 1 contrastive pretrain (always seeded with cfg.seed; the
    # sbatch passes --seed 42 for the single per-fold pretrain).
    run_stage1_pretrain(cfg, args.pretrain_epochs, device, ckpt_path)
    if args.pretrain_only:
        print("[InVAR-clpretrain] --pretrain_only: stage 1 done; "
              "exiting.")
        return

    # Single-process path: pretrain then finetune this (fold, seed).
    run_stage2_finetune(cfg, args.finetune_epochs, device, ckpt_path)


if __name__ == "__main__":
    main()
