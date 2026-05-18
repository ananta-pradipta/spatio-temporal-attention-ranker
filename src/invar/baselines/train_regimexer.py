"""RegimeXer-iT trainer.

Dispatched from `train_baseline.py` when `--baseline regimexer` is set.

Uses the existing `src.invar.training.loss.hybrid_loss` (Huber + listwise
IC + pairwise margin), per the 2026-05-11 user override that dropped the
CCC component. Vol head is enabled for modes with `use_vol_head=True`
(no_gate, full, moe_k8); disabled (lambda_vol=0) for macro_tokens_only
and film. Alpha regularizer (`lambda_alpha * mean(alpha[mask])`) is
added on top of hybrid_loss.total.

Schedule and hyperparameters:
  optimizer    AdamW(lr=5e-4, weight_decay=1e-4) -- matches MAiT, not the
               existing-baselines default (lr=1e-4, wd=1e-5). This is the
               documented deviation from `train_baseline.py` conventions.
  schedule     cosine annealing over epochs, no warmup.
  epochs       up to cfg.epochs (default 30); early stop on val rank-IC,
               patience 5, restore-best.
  grad_clip    1.0
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.invar.baselines.regimexer import (
    RegimeXerIT,
    RegimeXerITConfig,
    count_parameters,
)
from src.invar.data.dataset import (
    InvarDataset,
    MACRO_FEATURE_DIM,
    PANEL_FEATURE_DIM,
)
from src.invar.evaluation.metrics import (
    daily_ic,
    daily_rank_ic,
    long_short_sharpe,
    ndcg_at_k,
)
from src.invar.training.loss import LossWeights, hybrid_loss
from src.invar.training.train import set_seeds


def _eval_regimexer(model: RegimeXerIT, dataset: InvarDataset,
                    device: torch.device,
                    collect_predictions: bool = False) -> dict:
    """Per-day evaluation for RegimeXer-iT, with alpha statistics."""
    model.eval()
    pred_rows: list[dict] = []
    alpha_means: list[float] = []
    alpha_stds: list[float] = []
    with torch.no_grad():
        for batch in dataset:
            features = batch.features.to(device)
            macro = batch.macro.to(device)
            mask = batch.mask.to(device)
            out = model(features, macro, mask, return_attn=False)
            yh = out["y_hat"].cpu().numpy()
            yt = batch.y_cs.numpy()
            alpha = out["alpha"].detach().cpu().numpy()
            alpha_means.append(float(alpha.mean()))
            alpha_stds.append(float(alpha.std()))
            date_str = batch.date.strftime("%Y-%m-%d")
            for i in range(features.shape[0]):
                pred_rows.append({
                    "date": date_str, "ticker": batch.tickers[i],
                    "y_hat": float(yh[i]), "y_true": float(yt[i]),
                    "sector_id": int(batch.sector_id[i]),
                    "size_decile": int(batch.size_decile[i]),
                    "age_bucket": int(batch.age_bucket[i]),
                })
    if not pred_rows:
        return dict(
            ic=float("nan"), rank_ic=float("nan"),
            ndcg10=float("nan"), ndcg50=float("nan"),
            sharpe=float("nan"), annual_return=float("nan"),
            annual_vol=float("nan"), max_drawdown=float("nan"),
            mean_alpha=float("nan"), std_alpha=float("nan"),
            n_days=0, predictions=[] if collect_predictions else None,
        )
    pred_df = pd.DataFrame(pred_rows)
    ic = daily_ic(pred_df)
    rk = daily_rank_ic(pred_df)
    nd10 = ndcg_at_k(pred_df, k=10)
    nd50 = ndcg_at_k(pred_df, k=50)
    sharpe = long_short_sharpe(pred_df)
    return dict(
        ic=ic["mean"], rank_ic=rk["mean"],
        ndcg10=nd10["mean"], ndcg50=nd50["mean"],
        sharpe=sharpe["sharpe"],
        annual_return=sharpe["annual_return"],
        annual_vol=sharpe["annual_vol"],
        max_drawdown=sharpe["max_drawdown"],
        mean_alpha=float(np.mean(alpha_means)),
        std_alpha=float(np.mean(alpha_stds)),
        n_days=ic["n_days"],
        predictions=pred_rows if collect_predictions else None,
    )


def train_regimexer(cfg) -> dict:
    """Train RegimeXer-iT end to end.

    cfg is a BaselineRunConfig plus `regimexer_mode` attribute attached by
    the dispatcher in train_baseline.py.
    """
    set_seeds(cfg.seed)
    # Spec style requirement: deterministic per (fold, seed).
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode = getattr(cfg, "regimexer_mode", "full")
    print(f"[regimexer] mode={mode}  fold={cfg.fold}  seed={cfg.seed}  "
          f"device={device}", flush=True)

    train_ds = InvarDataset(fold=cfg.fold, split="train")
    val_ds = InvarDataset(fold=cfg.fold, split="val")
    test_ds = InvarDataset(fold=cfg.fold, split="test")
    print(f"[regimexer] train days {len(train_ds)}  val days {len(val_ds)}  "
          f"test days {len(test_ds)}", flush=True)

    model_cfg = RegimeXerITConfig(
        n_panel=PANEL_FEATURE_DIM, n_macro=MACRO_FEATURE_DIM,
        lookback=60, mode=mode,
    )
    model = RegimeXerIT(model_cfg).to(device)
    n_params = count_parameters(model)

    optim = AdamW(model.parameters(), lr=5.0e-4, weight_decay=1.0e-4)
    scheduler = CosineAnnealingLR(optim, T_max=max(cfg.epochs, 1))

    # Loss weights:
    #   primary: hybrid_loss (Huber + listwise IC + pairwise margin) always on.
    #   vol_mse: lambda_vol when use_vol_head, else 0.
    #   regime_ce, entropy, sinkhorn: 0 (no regime classifier head, no bank).
    weights = LossWeights(
        regime_ce=0.0,
        vol_mse=model_cfg.lambda_vol if model.use_vol_head else 0.0,
        entropy=0.0,
        sinkhorn=0.0,
    )
    lambda_alpha = model_cfg.lambda_alpha

    out_dir = (
        Path(cfg.output_dir) / "regimexer" / mode
        / f"fold{cfg.fold}" / f"seed{cfg.seed}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best_val_rank_ic = -float("inf")
    best_epoch = -1
    epochs_no_improve = 0
    patience = 5
    best_state = None

    eligible_indices = list(train_ds._eligible_idx)

    for epoch in range(cfg.epochs):
        t0 = time.time()
        model.train()
        epoch_losses = []
        epoch_alpha_means = []
        random.shuffle(eligible_indices)
        for t in eligible_indices:
            batch = train_ds.get(int(t))
            features = batch.features.to(device)
            macro = batch.macro.to(device)
            mask = batch.mask.to(device)
            y_cs = batch.y_cs.to(device)
            vol_target = batch.fwd_vol_20d.to(device)
            has_vol = batch.has_fwd_vol.to(device)

            optim.zero_grad()
            out = model(features, macro, mask, return_attn=False)
            hl = hybrid_loss(
                out["y_hat"], y_cs, mask,
                out["regime_logits"], batch.regime_label,
                out["vol_hat"], vol_target, has_vol,
                weights=weights, attn_weights=None,
                bank_usage_counts=None,
            )
            alpha_reg = lambda_alpha * out["alpha"][mask].mean()
            total = hl.total + alpha_reg
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optim.step()
            epoch_losses.append(float(total.item()))
            epoch_alpha_means.append(float(out["alpha"].detach().mean().item()))

        scheduler.step()
        train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        train_alpha = float(np.mean(epoch_alpha_means)) if epoch_alpha_means else float("nan")

        val = _eval_regimexer(model, val_ds, device,
                               collect_predictions=False)
        history.append(dict(
            epoch=epoch, train_loss=train_loss,
            train_alpha_mean=train_alpha,
            val_ic=val["ic"], val_rank_ic=val["rank_ic"],
            val_ndcg10=val["ndcg10"], val_alpha_mean=val["mean_alpha"],
            val_alpha_std=val["std_alpha"],
        ))
        print(
            f"[regimexer {mode}] epoch {epoch}: loss={train_loss:.4f}  "
            f"val_ic={val['ic']:+.4f}  val_rank_ic={val['rank_ic']:+.4f}  "
            f"alpha_tr={train_alpha:.3f}  alpha_va={val['mean_alpha']:.3f}  "
            f"({time.time() - t0:.1f}s)",
            flush=True,
        )

        if (np.isfinite(val["rank_ic"])
                and val["rank_ic"] > best_val_rank_ic):
            best_val_rank_ic = val["rank_ic"]
            best_epoch = epoch
            epochs_no_improve = 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[regimexer {mode}] early stop at epoch {epoch} "
                      f"(best epoch {best_epoch} val_rank_ic={best_val_rank_ic:+.4f})",
                      flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(best_state, out_dir / "ckpt.pt")

    test_eval = _eval_regimexer(model, test_ds, device,
                                  collect_predictions=cfg.save_predictions)
    pred_rows = test_eval.pop("predictions", None)
    if cfg.save_predictions and pred_rows:
        pd.DataFrame(pred_rows).to_parquet(
            out_dir / "predictions.parquet", index=False,
        )
    print(
        f"[regimexer {mode}] FINAL test_ic={test_eval['ic']:+.4f}  "
        f"rank_ic={test_eval['rank_ic']:+.4f}  "
        f"sharpe={test_eval['sharpe']:.3f}  "
        f"mean_alpha_test={test_eval['mean_alpha']:.3f}",
        flush=True,
    )

    result = {
        "baseline": cfg.baseline_name,
        "regimexer_mode": mode,
        "config": asdict(cfg),
        "n_params": int(n_params),
        "best_val_rank_ic": float(best_val_rank_ic),
        "best_epoch": int(best_epoch),
        "test_ic": float(test_eval["ic"]),
        "test_rank_ic": float(test_eval["rank_ic"]),
        "test_ndcg10": float(test_eval["ndcg10"]),
        "test_ndcg50": float(test_eval["ndcg50"]),
        "test_sharpe": float(test_eval["sharpe"]),
        "test_annual_return": float(test_eval["annual_return"]),
        "test_annual_vol": float(test_eval["annual_vol"]),
        "test_max_drawdown": float(test_eval["max_drawdown"]),
        "mean_alpha_test": float(test_eval["mean_alpha"]),
        "std_alpha_test": float(test_eval["std_alpha"]),
        "history": history,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    return result
