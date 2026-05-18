"""Train an external baseline on the LATTICE panel under the InVAR protocol.

Produces ``results.json`` and ``predictions.parquet`` with the same schema
as ``src.invar.training.train`` so the headline aggregation can read
either an InVAR run or a baseline run uniformly.

Loss: same hybrid as InVAR with auxiliary heads disabled (regime CE and
vol MSE weight 0). Huber + listwise IC + pairwise margin remain active.
This matches spec line 165 ("use same hybrid loss where architecture
supports auxiliary heads; pure cross-sectional MSE where it does not").
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
from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from src.invar.data.dataset import InvarDataset, PANEL_FEATURE_DIM, MACRO_FEATURE_DIM
from src.invar.evaluation.metrics import (
    daily_ic, daily_rank_ic, ndcg_at_k, long_short_sharpe,
)
from src.invar.training.loss import LossWeights, hybrid_loss
from src.invar.training.train import (
    set_seeds, warmup_cosine, evaluate, TrainConfig,
)


@dataclass
class BaselineRunConfig:
    fold: int = 1
    seed: int = 42
    lr: float = 1.0e-4
    weight_decay: float = 1.0e-5
    warmup_steps: int = 500
    grad_clip: float = 1.0
    epochs: int = 10
    early_stop_patience: int = 3
    output_dir: str = "experiments/invar/baselines"
    save_predictions: bool = True
    baseline_name: str = "itransformer"


def build_baseline(name: str, n_features: int | None = None) -> torch.nn.Module:
    """Instantiate the named baseline. Adapter signatures must match
    InVAR's forward dict (y_hat, regime_logits, vol_hat).

    Args:
        name: baseline name.
        n_features: effective panel feature width from the dataset. Defaults
            to ``PANEL_FEATURE_DIM`` for backward compatibility with the
            canonical 26-col panel call sites; pass the dataset's
            ``feature_dim`` when running on a panel with a different
            column count (e.g. the augmented 37-feature panel).
    """
    F = n_features if n_features is not None else PANEL_FEATURE_DIM
    if name == "itransformer":
        from src.invar.baselines.itransformer import (
            ITransformer, ITransformerConfig,
        )
        cfg = ITransformerConfig(
            n_features=F, lookback=60,
        )
        return ITransformer(cfg)
    if name == "stockmixer":
        from src.invar.baselines.stockmixer import StockMixer, StockMixerConfig
        cfg = StockMixerConfig(n_features=F, lookback=60)
        return StockMixer(cfg)
    if name == "master":
        from src.invar.baselines.master import Master, MasterConfig
        cfg = MasterConfig(
            n_features=F, macro_dim=MACRO_FEATURE_DIM,
            lookback=60,
        )
        return Master(cfg)
    if name == "factorvae":
        from src.invar.baselines.factorvae_adapter import (
            FactorVAEAdapter, FactorVAEConfig,
        )
        cfg = FactorVAEConfig(
            n_features=F, lookback=60,
        )
        return FactorVAEAdapter(cfg)
    raise ValueError(f"unknown baseline: {name}")


def train_baseline(cfg: BaselineRunConfig) -> dict:
    # MAiT uses a different forward signature, loss, optimizer, and sampler,
    # so dispatch to a dedicated trainer instead of branching this loop.
    if cfg.baseline_name == "mait":
        from src.invar.baselines.train_mait import train_mait
        return train_mait(cfg)
    # RegimeXer-iT: same data path and hybrid_loss as the existing
    # baselines, but adds (a) an auxiliary vol head (lambda_vol=0.1
    # for modes with vol head), (b) an alpha-mean regularizer
    # (lambda_alpha=1e-3), and (c) per-epoch alpha logging. Mode-specific
    # behaviour (A1 through A5) is selected via cfg.regimexer_mode.
    if cfg.baseline_name == "regimexer":
        from src.invar.baselines.train_regimexer import train_regimexer
        return train_regimexer(cfg)
    set_seeds(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[baseline {cfg.baseline_name}] fold={cfg.fold} seed={cfg.seed} "
          f"device={device}", flush=True)

    train_ds = InvarDataset(fold=cfg.fold, split="train")
    val_ds = InvarDataset(fold=cfg.fold, split="val")
    test_ds = InvarDataset(fold=cfg.fold, split="test")
    print(f"[baseline {cfg.baseline_name}] train days {len(train_ds)}  "
          f"val days {len(val_ds)}  test days {len(test_ds)}", flush=True)

    model = build_baseline(cfg.baseline_name,
                            n_features=train_ds.feature_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[baseline {cfg.baseline_name}] params={n_params:,}", flush=True)

    optim = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = max(len(train_ds), 1) * cfg.epochs
    scheduler = LambdaLR(
        optim, lambda s: warmup_cosine(s, cfg.warmup_steps, total_steps),
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    # Auxiliary heads disabled for baselines lacking native regime / vol
    # supervision; entropy and Sinkhorn off by default (no retrieval bank).
    weights = LossWeights(regime_ce=0.0, vol_mse=0.0, entropy=0.0, sinkhorn=0.0)

    out_dir = Path(cfg.output_dir) / cfg.baseline_name / f"fold{cfg.fold}/seed{cfg.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    history = []
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
            features = batch.features.to(device)
            macro = batch.macro.to(device)
            mask = batch.mask.to(device)
            y_cs = batch.y_cs.to(device)
            vol_target = batch.fwd_vol_20d.to(device)
            has_vol = batch.has_fwd_vol.to(device)

            optim.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
                # FactorVAE needs the cross-sectional target inside the
                # forward pass to run the posterior + KL branch.
                if cfg.baseline_name == "factorvae":
                    out = model(features, macro, mask, return_attn=False,
                                  y_cs=y_cs)
                else:
                    out = model(features, macro, mask, return_attn=False)
                loss_out = hybrid_loss(
                    out["y_hat"], y_cs, mask,
                    out["regime_logits"], batch.regime_label,
                    out["vol_hat"], vol_target, has_vol,
                    weights=weights, attn_weights=None,
                    bank_usage_counts=None,
                )
                total_loss = loss_out.total
                # Add the FactorVAE ELBO term when active.
                if (cfg.baseline_name == "factorvae"
                        and getattr(model, "_last_vae_loss", None) is not None):
                    lambda_vae = getattr(model.cfg, "lambda_vae", 0.10)
                    total_loss = total_loss + lambda_vae * model._last_vae_loss
            scaler.scale(total_loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optim)
            scaler.update()
            scheduler.step()
            epoch_losses.append(float(loss_out.total.item()))

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        val_metrics = evaluate(model, val_ds, device)
        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "val_ic": val_metrics["ic"], "val_rank_ic": val_metrics["rank_ic"],
            "val_ndcg10": val_metrics["ndcg10"],
        })
        print(f"[baseline {cfg.baseline_name}] epoch {epoch}: "
              f"loss={train_loss:.4f} val_ic={val_metrics['ic']:+.4f}  "
              f"val_rank_ic={val_metrics['rank_ic']:+.4f}",
              flush=True)

        if np.isfinite(val_metrics["ic"]) and val_metrics["ic"] > best_val_ic:
            best_val_ic = val_metrics["ic"]
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), out_dir / "ckpt.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.early_stop_patience:
                print(f"[baseline {cfg.baseline_name}] early stop at epoch {epoch}", flush=True)
                break

    model.load_state_dict(torch.load(out_dir / "ckpt.pt", weights_only=False))
    test_metrics = evaluate(model, test_ds, device,
                              collect_predictions=cfg.save_predictions)
    pred_rows = test_metrics.pop("predictions", None)
    if cfg.save_predictions and pred_rows:
        pd.DataFrame(pred_rows).to_parquet(
            out_dir / "predictions.parquet", index=False,
        )
    print(f"[baseline {cfg.baseline_name}] test_ic={test_metrics['ic']:+.4f}  "
          f"rank_ic={test_metrics['rank_ic']:+.4f}  "
          f"sharpe={test_metrics['sharpe']:.3f}", flush=True)

    result = {
        "baseline": cfg.baseline_name,
        "config": asdict(cfg),
        "n_params": int(n_params),
        "best_val_ic": float(best_val_ic),
        "best_epoch": int(best_epoch),
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
    p.add_argument("--baseline", type=str, required=True)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--output-dir", type=str,
                    default="experiments/invar/baselines")
    p.add_argument("--save-predictions", action="store_true", default=True)
    p.add_argument(
        "--regimexer-mode", type=str, default="full",
        choices=["macro_tokens_only", "film", "no_gate", "full", "moe_k8"],
        help="RegimeXer-iT mode (A1=macro_tokens_only, A2=film, "
             "A3=no_gate, A4=full, A5=moe_k8). Ignored for non-regimexer baselines.",
    )
    args = p.parse_args()
    cfg = BaselineRunConfig(
        baseline_name=args.baseline, fold=args.fold, seed=args.seed,
        epochs=args.epochs, output_dir=args.output_dir,
        save_predictions=args.save_predictions,
    )
    # Attach mode without changing BaselineRunConfig schema (the dataclass
    # is shared with the existing baselines).
    cfg.regimexer_mode = args.regimexer_mode
    train_baseline(cfg)


if __name__ == "__main__":
    main()
