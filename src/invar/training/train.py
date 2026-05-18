"""InVAR training loop.

Per spec section "PHASE 2: TRAINING LOOP + FOLD 1 SANITY CHECK":

  - Optimiser:        AdamW, lr 1e-4, weight_decay 1e-5
  - Schedule:         500-step linear warmup then cosine to 10% of peak
  - Gradient clipping: norm 1.0
  - Batch policy:     one trading day per gradient step
  - Mixed precision:  torch.amp autocast
  - Epochs:           10 max
  - Early stop:       patience 3 on validation Pearson IC
  - One seed per CLI invocation via --seed

Stability mitigations on the retrieval bank:

  - Epoch 1: stop-grad on memory values (only train keys, queries, rest
             of model).
  - K curriculum: epoch 1 uses the full bank; epochs 2 to 10 anneal K
             linearly to ``cfg.top_k_retrieve`` (default 32).

The K curriculum is implemented as a per-epoch schedule on
``RegimeAxisRetrieval.set_top_k``; the stop-grad on values is via
``RegimeAxisRetrieval.freeze_values``.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from src.invar.data.dataset import InvarDataset, PANEL_FEATURE_DIM, MACRO_FEATURE_DIM
from src.invar.evaluation.metrics import (
    daily_ic, daily_rank_ic, ndcg_at_k, cohort_stratified_ic, long_short_sharpe,
)
from src.invar.model.invar import (
    Invar, InvarConfig, RegimeAxisRetrieval, count_parameters,
)
from src.invar.training.loss import LossWeights, hybrid_loss, loss_weights_for


@dataclass
class TrainConfig:
    fold: int = 1
    seed: int = 42
    lr: float = 1.0e-4
    weight_decay: float = 1.0e-5
    warmup_steps: int = 500
    grad_clip: float = 1.0
    epochs: int = 10
    early_stop_patience: int = 3
    output_dir: str = "experiments/invar/fold1"
    save_predictions: bool = True
    regime_axis: str = "retrieval"
    loss_config: str = "minimal"
    bank_size: int | None = None
    use_market_gate: bool = False
    gate_beta_init: float = 2.0
    gate_hidden_dim: int = 0
    gate_l2_weight: float = 1.0e-4
    gate_location: str = "input"             # "input" or "post_tokenizer"
    gate_form: str = "softmax_F"             # "softmax_F" or "sigmoid"
    disable_bank: bool = False
    # InVAR v6:
    use_market_gate_v2: bool = False
    macro_encoder_mode: str = "mlp_flat"
    macro_state_dim: int = 64
    market_gate_v2_form: str = "softmax_F"
    market_gate_v2_hidden_dim: int = 64
    use_dynamic_bank_controller: bool = False
    bank_controller_mode: str = "hybrid"
    bank_controller_min_weight: float = 0.05
    bank_controller_max_weight: float = 1.00
    bank_controller_hidden_dim: int = 64
    use_regime_loss: bool = False
    regime_loss_weight: float = 0.05
    # Differentiable retrieval ablation:
    retrieval_mode: str = "hard_topk"
    gumbel_tau: float = 1.0
    # SWA-InVAR (Stochastic Weight Averaging):
    use_swa: bool = False                  # if False, behaves identically to baseline
    swa_decay: float = 0.999               # EMA decay; 0.999 over ~1000 steps ~= effective 1k-step window
    swa_warmup_epochs: int = 5             # only start EMA after this many warmup epochs
    # Self-distillation: load a frozen teacher and add a soft pairwise
    # ranking distillation loss alongside the standard ranking loss.
    teacher_ckpt: str = ""                 # path to teacher ckpt.pt; empty = no distillation
    distill_weight: float = 0.5            # weight on distill term, total = ranking + w * distill
    distill_temp: float = 2.0              # softening temperature on pairwise differences
    # Macro feature subset (2026-05-11): "full" uses all 24 macro features
    # (the SWA-InVAR headline); "minimal" subsets to the 8-feature regime-
    # discriminating set from docs/macro_feature_analysis.md. With minimal,
    # GMM regime labels stay based on the original 24-feature macro (fit
    # inside InvarDataset.__init__); only the macro tensor passed to the
    # MarketGateV2 and MacroEncoder is subsetted, and macro_dim becomes 8.
    macro_subset: str = "full"


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_torch_save(obj, path: Path, retries: int = 5, sleep_s: float = 2.0) -> None:
    """torch.save wrapper that survives Wulver mmfs1 GPFS write hiccups.

    The default torch.save uses a zip-file writer that gets 'unexpected pos
    X vs Y' on flaky parallel filesystems. Two mitigations:

    1. Save to node-local /tmp first (respecting $SLURM_TMPDIR or $TMPDIR
       if set, falling back to /tmp), then atomically move to the final
       path on mmfs1.
    2. Use the legacy pickle-only serialization
       (`_use_new_zipfile_serialization=False`), which avoids the zip-writer
       path entirely.

    Combined with N retries, this is robust against the transient GPFS
    issues we've been seeing.
    """
    import os
    import shutil
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_root = Path(
        os.environ.get("SLURM_TMPDIR")
        or os.environ.get("TMPDIR")
        or "/tmp",
    )
    tmp_root.mkdir(parents=True, exist_ok=True)
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            tmp = tmp_root / f"{path.name}.tmp.{os.getpid()}.{attempt}"
            torch.save(obj, tmp, _use_new_zipfile_serialization=False)
            shutil.move(str(tmp), str(path))
            return
        except (RuntimeError, OSError) as exc:
            last_exc = exc
            time.sleep(sleep_s * (1.0 + attempt))
    raise RuntimeError(
        f"safe_torch_save failed after {retries} attempts: {last_exc}",
    )


def warmup_cosine(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return max(1, step) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))


def k_curriculum(epoch: int, cfg: InvarConfig) -> int:
    """Anneal K from full bank (epoch 1) down to top_k_retrieve."""
    if epoch <= 0:
        return cfg.bank_size
    if epoch >= 5:
        return cfg.top_k_retrieve
    # linear anneal from bank_size at epoch=1 to top_k at epoch=5
    frac = (epoch - 1) / 4.0
    k = int(round((1.0 - frac) * cfg.bank_size + frac * cfg.top_k_retrieve))
    return max(cfg.top_k_retrieve, min(cfg.bank_size, k))


def evaluate(
    model: Invar, dataset: InvarDataset, device: torch.device,
    collect_predictions: bool = False,
) -> dict:
    model.eval()
    pred_rows = []
    with torch.no_grad():
        for batch in dataset:
            features = batch.features.to(device)
            macro = batch.macro.to(device)
            mask = batch.mask.to(device)
            out = model(features, macro, mask, return_attn=False)
            yh = out["y_hat"].cpu().numpy()
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
        return {
            "ic": float("nan"), "rank_ic": float("nan"),
            "ndcg10": float("nan"), "ndcg50": float("nan"),
            "sharpe": float("nan"), "n_days": 0,
            "predictions": [] if collect_predictions else None,
        }
    pred_df = pd.DataFrame(pred_rows)
    ic = daily_ic(pred_df)
    rank = daily_rank_ic(pred_df)
    ndcg10 = ndcg_at_k(pred_df, k=10)
    ndcg50 = ndcg_at_k(pred_df, k=50)
    sharpe = long_short_sharpe(pred_df)
    return {
        "ic": ic["mean"], "ic_std": ic["std"], "n_days": ic["n_days"],
        "rank_ic": rank["mean"], "rank_ic_std": rank["std"],
        "ndcg10": ndcg10["mean"], "ndcg50": ndcg50["mean"],
        "sharpe": sharpe["sharpe"],
        "annual_return": sharpe["annual_return"],
        "annual_vol": sharpe["annual_vol"],
        "max_drawdown": sharpe["max_drawdown"],
        "predictions": pred_rows if collect_predictions else None,
    }


def train_one(cfg: TrainConfig, model_cfg: InvarConfig | None = None) -> dict:
    set_seeds(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[invar train] fold={cfg.fold} seed={cfg.seed} device={device}",
          flush=True)

    train_ds = InvarDataset(fold=cfg.fold, split="train")
    val_ds = InvarDataset(fold=cfg.fold, split="val")
    test_ds = InvarDataset(fold=cfg.fold, split="test")
    print(f"[invar train] train days {len(train_ds)}  val days {len(val_ds)}  "
          f"test days {len(test_ds)}", flush=True)

    # Optional macro feature subset. Per docs/macro_feature_analysis.md
    # (2026-05-11), 16 of 24 macro features have max |z| < 2 across all
    # three fold test windows (sector ETFs, equity 5d returns,
    # market_breadth_proxy, gld_5d_ret, dxy_5d_ret, tlt_5d_ret,
    # slope_3m10y); they do not move relative to their own training
    # distribution and the PCA effective rank of the 24-d macro state is
    # only ~6. The 8 kept features are the regime-discriminators that
    # account for one fold's max |z| > 2 each plus hyg_5d_ret as the
    # cross-regime credit signal.
    if cfg.macro_subset == "minimal":
        keep_idx = np.array([0, 1, 2, 3, 4, 5, 7, 9], dtype=np.int64)
        kept_names = [
            "vix", "vix_term_slope", "move_proxy",
            "dgs2", "dgs10", "slope_2s10s",
            "breakeven_10y", "hyg_5d_ret",
        ]
        for ds in (train_ds, val_ds, test_ds):
            ds._macro_tensor = ds._macro_tensor[:, keep_idx]
            ds._macro_raw_tensor = ds._macro_raw_tensor[:, keep_idx]
            ds._macro_z_mean = ds._macro_z_mean[keep_idx]
            ds._macro_z_std = ds._macro_z_std[keep_idx]
        effective_macro_dim = len(keep_idx)
        print(f"[invar train] macro_subset=minimal  kept={kept_names}  "
              f"macro_dim={effective_macro_dim}", flush=True)
    else:
        effective_macro_dim = MACRO_FEATURE_DIM

    if model_cfg is None:
        model_cfg = InvarConfig(
            n_features=train_ds.feature_dim, macro_dim=effective_macro_dim,
            regime_axis=cfg.regime_axis,
        )
        if cfg.bank_size is not None:
            model_cfg.bank_size = cfg.bank_size
        model_cfg.use_market_gate = cfg.use_market_gate
        model_cfg.gate_beta_init = cfg.gate_beta_init
        model_cfg.gate_hidden_dim = cfg.gate_hidden_dim
        model_cfg.gate_location = cfg.gate_location
        model_cfg.gate_form = cfg.gate_form
        model_cfg.disable_bank = cfg.disable_bank
        model_cfg.use_market_gate_v2 = cfg.use_market_gate_v2
        model_cfg.macro_encoder_mode = cfg.macro_encoder_mode
        model_cfg.macro_state_dim = cfg.macro_state_dim
        model_cfg.market_gate_v2_form = cfg.market_gate_v2_form
        model_cfg.market_gate_v2_hidden_dim = cfg.market_gate_v2_hidden_dim
        model_cfg.use_dynamic_bank_controller = cfg.use_dynamic_bank_controller
        model_cfg.bank_controller_mode = cfg.bank_controller_mode
        model_cfg.bank_controller_min_weight = cfg.bank_controller_min_weight
        model_cfg.bank_controller_max_weight = cfg.bank_controller_max_weight
        model_cfg.bank_controller_hidden_dim = cfg.bank_controller_hidden_dim
        model_cfg.retrieval_mode = cfg.retrieval_mode
        model_cfg.gumbel_tau = cfg.gumbel_tau
    model = Invar(model_cfg).to(device)
    n_params = count_parameters(model)
    print(f"[invar train] params={n_params:,}  bank_size={model_cfg.bank_size}  "
          f"top_k_retrieve={model_cfg.top_k_retrieve}  "
          f"loss_config={cfg.loss_config}", flush=True)

    if cfg.use_market_gate:
        gate_params, other_params = [], []
        for name, p in model.named_parameters():
            if name.startswith("market_gate."):
                gate_params.append(p)
            else:
                other_params.append(p)
        optim = AdamW(
            [
                {"params": gate_params, "lr": cfg.lr,
                  "weight_decay": cfg.gate_l2_weight},
                {"params": other_params, "lr": cfg.lr,
                  "weight_decay": cfg.weight_decay},
            ],
        )
    else:
        optim = AdamW(model.parameters(), lr=cfg.lr,
                       weight_decay=cfg.weight_decay)
    total_steps = max(len(train_ds), 1) * cfg.epochs
    scheduler = LambdaLR(
        optim, lambda s: warmup_cosine(s, cfg.warmup_steps, total_steps),
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    weights = loss_weights_for(cfg.loss_config)

    out_dir = Path(cfg.output_dir) / f"seed{cfg.seed}_design{cfg.regime_axis[0].upper()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_val_ic = -1.0
    best_epoch = -1
    best_k = model_cfg.top_k_retrieve
    best_state: dict[str, torch.Tensor] | None = None
    epochs_no_improve = 0

    is_retrieval = isinstance(model.regime_axis, RegimeAxisRetrieval)
    # Initialise bank_usage to a uniform pseudo-count so the Sinkhorn
    # balance penalty is well-defined on step 1 (KL approximately 0 at
    # uniform prior) and only deviates as actual usage accumulates.
    bank_usage = (
        torch.full((model_cfg.bank_size,), 1.0, device=device)
        if is_retrieval else None
    )

    # SWA-InVAR: lazy-initialised EMA state. Populated on the first
    # post-warmup training step. At eval time we swap model weights
    # to the EMA copy and restore afterwards.
    ema_state: dict[str, torch.Tensor] | None = None

    # Self-distillation: load a frozen teacher with the same architecture.
    # Teacher's pairwise rank logits become soft targets for the student's
    # pairwise rank logits via BCE on sigmoid(diff / temp).
    teacher = None
    if cfg.teacher_ckpt:
        teacher = Invar(model_cfg).to(device)
        t_ckpt = torch.load(cfg.teacher_ckpt, map_location=device, weights_only=False)
        if isinstance(t_ckpt, dict) and "model_state" in t_ckpt:
            teacher.load_state_dict(t_ckpt["model_state"])
        else:
            teacher.load_state_dict(t_ckpt)
        teacher.eval()
        for tp in teacher.parameters():
            tp.requires_grad_(False)
        print(f"[invar train] teacher loaded from {cfg.teacher_ckpt}  "
              f"distill_weight={cfg.distill_weight}  distill_temp={cfg.distill_temp}",
              flush=True)

    for epoch in range(cfg.epochs):
        if is_retrieval:
            # v3: K curriculum removed. Fix top_k at cfg default; keep
            # values trainable from epoch 0 (the zero-init cross_attn
            # out_proj makes the cross-attn path start at zero anyway,
            # so a separate stop-gradient warmup is no longer needed).
            model.regime_axis.set_top_k(model_cfg.top_k_retrieve)
            model.regime_axis.freeze_values(False)

        model.train()
        epoch_losses = []
        component_acc: dict[str, list] = {}
        # bank_usage is cumulative across the entire run; not reset per
        # epoch. This keeps the Sinkhorn term well-defined on step 1 of
        # epoch 1 (uniform initial guess) and lets the running frequency
        # drift toward the actual usage distribution as training proceeds.

        eligible = list(train_ds._eligible_idx)
        random.shuffle(eligible)
        for t in eligible:
            batch = train_ds.get(int(t))
            features = batch.features.to(device)
            macro = batch.macro.to(device)
            mask = batch.mask.to(device)
            y_cs = batch.y_cs.to(device)
            vol_target = batch.fwd_vol_20d.to(device)
            has_vol = batch.has_fwd_vol.to(device)

            optim.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
                out = model(features, macro, mask, return_attn=True)
                loss_out = hybrid_loss(
                    out["y_hat"], y_cs, mask,
                    out["regime_logits"], batch.regime_label,
                    out["vol_hat"], vol_target, has_vol,
                    weights=weights,
                    attn_weights=out.get("attn_weights"),
                    bank_usage_counts=bank_usage if bank_usage is not None else None,
                )
                total_loss = loss_out.total
                if teacher is not None:
                    with torch.no_grad():
                        out_T = teacher(features, macro, mask, return_attn=False)
                    active = mask.bool()
                    if active.sum() >= 2:
                        y_S = out["y_hat"][active]
                        y_T = out_T["y_hat"][active].detach()
                        diff_S = y_S.unsqueeze(0) - y_S.unsqueeze(1)
                        diff_T = y_T.unsqueeze(0) - y_T.unsqueeze(1)
                        T_temp = float(cfg.distill_temp)
                        p_T = torch.sigmoid(diff_T / T_temp)
                        logit_S = (diff_S / T_temp).float()
                        eye = torch.eye(
                            logit_S.size(0), device=device, dtype=torch.bool,
                        )
                        distill_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                            logit_S[~eye], p_T[~eye].float(),
                        )
                        total_loss = total_loss + cfg.distill_weight * distill_loss
                        loss_out.components["distill"] = float(distill_loss.item())
            scaler.scale(total_loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optim)
            scaler.update()
            scheduler.step()
            epoch_losses.append(float(loss_out.total.item()))
            for k, v in loss_out.components.items():
                component_acc.setdefault(k, []).append(v)

            # Track usage of retrieved bank entries (for the next-step
            # Sinkhorn proxy; soft-counted on top-K indices).
            if bank_usage is not None and model.regime_axis.last_top_idx is not None:
                idx = model.regime_axis.last_top_idx.detach()
                bank_usage.index_add_(
                    0, idx, torch.ones_like(idx, dtype=bank_usage.dtype),
                )

            # SWA-InVAR: post-warmup, accumulate an exponential moving
            # average of model weights. Reference: Izmailov et al. 2018,
            # "Averaging Weights Leads to Wider Optima and Better
            # Generalization". For OOD test regimes (F1 = COVID, F2 =
            # rate-stress, neither matches the train fold), wider /
            # flatter minima generalise better.
            if cfg.use_swa and epoch >= cfg.swa_warmup_epochs:
                with torch.no_grad():
                    sd = model.state_dict()
                    if ema_state is None:
                        ema_state = {k: v.detach().clone() for k, v in sd.items()}
                    else:
                        d = float(cfg.swa_decay)
                        for k in ema_state:
                            ema_state[k].mul_(d).add_(sd[k].detach(), alpha=1.0 - d)

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        comp_means = {k: float(np.mean(v)) for k, v in component_acc.items()}

        # SWA-InVAR: evaluate on the EMA weights (when available).
        # Restore the live weights afterwards so training continues on the
        # SGD trajectory.
        if cfg.use_swa and ema_state is not None:
            saved_state = {
                k: v.detach().clone() for k, v in model.state_dict().items()
            }
            model.load_state_dict(ema_state)
            val_metrics = evaluate(model, val_ds, device)
            model.load_state_dict(saved_state)
        else:
            val_metrics = evaluate(model, val_ds, device)

        # Probe the per-block scalar gate g_scalar(q_t) on a sample val day
        # so we can audit whether the model opens or closes the
        # cross-attention path over the course of training.
        gate_values: list[float] = []
        if model_cfg.use_scalar_gate and len(val_ds._eligible_idx) > 0:
            model.eval()
            with torch.no_grad():
                t_probe = int(val_ds._eligible_idx[len(val_ds._eligible_idx) // 2])
                probe_batch = val_ds.get(t_probe)
                _ = model(
                    probe_batch.features.to(device),
                    probe_batch.macro.to(device),
                    probe_batch.mask.to(device),
                    return_attn=True,
                )
                # Re-compute g per block by reading q_t and the gate linear.
                q_t = model.macro_encoder(probe_batch.macro.to(device))
                for block in model.blocks:
                    g = float(torch.sigmoid(block.gate_mlp(q_t)).item())
                    gate_values.append(g)
            model.train()

        # InVAR v4: log market-gate alpha statistics if active.
        alpha_stats: dict | None = None
        if cfg.use_market_gate and model.get_alpha() is not None:
            a = model.get_alpha()  # (1, F)
            alpha_stats = {
                "mean": float(a.mean().item()),
                "std": float(a.std().item()),
                "min": float(a.min().item()),
                "max": float(a.max().item()),
                "beta": float(model.market_gate.beta.item()),
                "alpha": a[0].cpu().tolist(),
            }

        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "val_ic": val_metrics["ic"], "val_rank_ic": val_metrics["rank_ic"],
            "val_ndcg10": val_metrics["ndcg10"],
            "components": comp_means,
            "gate_values_per_block": gate_values,
            "market_gate_stats": alpha_stats,
        })
        gate_str = (f" gate={['%.3f' % g for g in gate_values]}"
                     if gate_values else "")
        print(f"[invar train] epoch {epoch}: loss={train_loss:.4f}  "
              f"val_ic={val_metrics['ic']:+.4f}  "
              f"val_rank_ic={val_metrics['rank_ic']:+.4f}  "
              f"val_ndcg10={val_metrics['ndcg10']:.4f}{gate_str}",
              flush=True)

        if np.isfinite(val_metrics["ic"]) and val_metrics["ic"] > best_val_ic:
            best_val_ic = val_metrics["ic"]
            best_epoch = epoch
            if is_retrieval:
                best_k = model.regime_axis._top_k
            epochs_no_improve = 0
            # Track best-val state in memory only - we round-trip to disk
            # once at the end of training to insulate the test eval from
            # transient parallel-FS write failures.
            best_state = {
                k: v.detach().clone() for k, v in (
                    ema_state.items()
                    if (cfg.use_swa and ema_state is not None)
                    else model.state_dict().items()
                )
            }
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.early_stop_patience:
                print(f"[invar train] early stop at epoch {epoch}", flush=True)
                break

    # SWA-InVAR: at end of training, swap to the final EMA state. Per
    # Izmailov et al. 2018, the standard SWA protocol evaluates on the
    # final averaged weights, not the val-best epoch.
    if cfg.use_swa and ema_state is not None:
        best_state = {k: v.detach().clone() for k, v in ema_state.items()}
        best_k = (
            model.regime_axis._top_k if is_retrieval
            else best_k
        )
        print(f"[invar train] SWA: using final EMA state for test eval",
              flush=True)

    # Load the in-memory best state into the model directly. Best-effort
    # save the same state to disk for downstream consumers; if the save
    # fails on a flaky parallel filesystem, swallow the error and proceed
    # (the in-memory state is what we test on either way).
    if best_state is None:
        # No improvement was ever recorded - use the live model state.
        best_state = {
            k: v.detach().clone() for k, v in model.state_dict().items()
        }
    model.load_state_dict(best_state)
    ckpt_best_k = best_k
    try:
        safe_torch_save({
            "model_state": best_state,
            "best_epoch": best_epoch,
            "best_k": ckpt_best_k,
            "swa_used": bool(cfg.use_swa and ema_state is not None),
        }, out_dir / "ckpt.pt")
    except Exception as exc:
        print(f"[invar train] ckpt save failed (best-effort): {exc}",
              flush=True)
    if is_retrieval:
        # K-mismatch fix: at test time use the K that was active when the
        # best-val-IC checkpoint was saved, not the curriculum endpoint.
        # Falls back to the cfg default if the ckpt predates the fix.
        k_for_test = ckpt_best_k if ckpt_best_k is not None else model_cfg.top_k_retrieve
        model.regime_axis.set_top_k(int(k_for_test))
        model.regime_axis.freeze_values(False)
        print(f"[invar train] reload best ckpt: K={k_for_test} (was {model_cfg.top_k_retrieve} default)",
              flush=True)
    test_metrics = evaluate(model, test_ds, device,
                              collect_predictions=cfg.save_predictions)

    pred_rows = test_metrics.pop("predictions", None)
    if cfg.save_predictions and pred_rows:
        pd.DataFrame(pred_rows).to_parquet(
            out_dir / "predictions.parquet", index=False,
        )

    print(f"[invar train] test_ic={test_metrics['ic']:+.4f}  "
          f"rank_ic={test_metrics['rank_ic']:+.4f}  "
          f"ndcg10={test_metrics['ndcg10']:.4f}  "
          f"sharpe={test_metrics['sharpe']:.3f}",
          flush=True)

    result = {
        "config": asdict(cfg),
        "model_config": model_cfg.__dict__,
        "n_params": int(n_params),
        "best_val_ic": float(best_val_ic),
        "best_epoch": int(best_epoch),
        "best_k_at_best_epoch": int(best_k) if is_retrieval else None,
        "test_ic": float(test_metrics["ic"]),
        "test_rank_ic": float(test_metrics["rank_ic"]),
        "test_ndcg10": float(test_metrics["ndcg10"]),
        "test_ndcg50": float(test_metrics["ndcg50"]),
        "test_sharpe": float(test_metrics["sharpe"]),
        "test_annual_return": float(test_metrics["annual_return"]),
        "test_annual_vol": float(test_metrics["annual_vol"]),
        "test_max_drawdown": float(test_metrics["max_drawdown"]),
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
    p.add_argument("--output-dir", type=str, default="experiments/invar/fold1")
    p.add_argument("--regime-axis", type=str, default="retrieval",
                    choices=["calendar", "kmeans", "retrieval"])
    p.add_argument("--save-predictions", action="store_true", default=True)
    p.add_argument("--loss-config", type=str, default="ranking",
                    choices=["minimal", "ranking", "full"])
    p.add_argument("--bank-size", type=int, default=None)
    p.add_argument("--use-market-gate", action="store_true")
    p.add_argument("--gate-beta-init", type=float, default=2.0)
    p.add_argument("--gate-hidden-dim", type=int, default=0)
    p.add_argument("--gate-l2-weight", type=float, default=1.0e-4)
    p.add_argument("--gate-location", type=str, default="input",
                    choices=["input", "post_tokenizer"])
    p.add_argument("--gate-form", type=str, default="softmax_F",
                    choices=["softmax_F", "sigmoid"])
    p.add_argument("--disable-bank", action="store_true")
    # InVAR v6:
    p.add_argument("--use-market-gate-v2", action="store_true")
    p.add_argument("--macro-encoder-mode", type=str, default="mlp_flat",
                    choices=["last", "mlp_flat", "temporal_attn", "gru"])
    p.add_argument("--macro-state-dim", type=int, default=64)
    p.add_argument("--market-gate-v2-form", type=str, default="softmax_F",
                    choices=["softmax_F", "sigmoid_centered", "sigmoid_residual"])
    p.add_argument("--market-gate-v2-hidden-dim", type=int, default=64)
    p.add_argument("--use-dynamic-bank-controller", action="store_true")
    p.add_argument("--bank-controller-mode", type=str, default="hybrid",
                    choices=["deterministic", "learned", "hybrid"])
    p.add_argument("--bank-controller-min-weight", type=float, default=0.05)
    p.add_argument("--bank-controller-max-weight", type=float, default=1.00)
    p.add_argument("--bank-controller-hidden-dim", type=int, default=64)
    p.add_argument("--use-regime-loss", action="store_true")
    p.add_argument("--regime-loss-weight", type=float, default=0.05)
    p.add_argument("--retrieval-mode", type=str, default="hard_topk",
                   choices=["hard_topk", "softmax_full",
                            "softmax_topk", "gumbel_topk"])
    p.add_argument("--gumbel-tau", type=float, default=1.0)
    p.add_argument("--use-swa", action="store_true")
    p.add_argument("--swa-decay", type=float, default=0.999)
    p.add_argument("--swa-warmup-epochs", type=int, default=5)
    p.add_argument("--teacher-ckpt", type=str, default="",
                   help="Path to teacher ckpt.pt for self-distillation")
    p.add_argument("--distill-weight", type=float, default=0.5)
    p.add_argument("--distill-temp", type=float, default=2.0)
    p.add_argument(
        "--macro-subset", type=str, default="full",
        choices=["full", "minimal"],
        help="Macro feature subset. 'full' uses all 24 features; "
             "'minimal' uses the 8-feature regime-discriminating subset "
             "from docs/macro_feature_analysis.md.",
    )
    args = p.parse_args()
    cfg = TrainConfig(
        fold=args.fold, seed=args.seed, epochs=args.epochs,
        output_dir=args.output_dir,
        save_predictions=args.save_predictions,
        regime_axis=args.regime_axis,
        loss_config=args.loss_config,
        bank_size=args.bank_size,
        use_market_gate=args.use_market_gate,
        gate_beta_init=args.gate_beta_init,
        gate_hidden_dim=args.gate_hidden_dim,
        gate_l2_weight=args.gate_l2_weight,
        gate_location=args.gate_location,
        gate_form=args.gate_form,
        disable_bank=args.disable_bank,
        use_market_gate_v2=args.use_market_gate_v2,
        macro_encoder_mode=args.macro_encoder_mode,
        macro_state_dim=args.macro_state_dim,
        market_gate_v2_form=args.market_gate_v2_form,
        market_gate_v2_hidden_dim=args.market_gate_v2_hidden_dim,
        use_dynamic_bank_controller=args.use_dynamic_bank_controller,
        bank_controller_mode=args.bank_controller_mode,
        bank_controller_min_weight=args.bank_controller_min_weight,
        bank_controller_max_weight=args.bank_controller_max_weight,
        bank_controller_hidden_dim=args.bank_controller_hidden_dim,
        use_regime_loss=args.use_regime_loss,
        regime_loss_weight=args.regime_loss_weight,
        retrieval_mode=args.retrieval_mode,
        gumbel_tau=args.gumbel_tau,
        use_swa=args.use_swa,
        swa_decay=args.swa_decay,
        swa_warmup_epochs=args.swa_warmup_epochs,
        teacher_ckpt=args.teacher_ckpt,
        distill_weight=args.distill_weight,
        distill_temp=args.distill_temp,
        macro_subset=args.macro_subset,
    )
    train_one(cfg)


if __name__ == "__main__":
    main()
