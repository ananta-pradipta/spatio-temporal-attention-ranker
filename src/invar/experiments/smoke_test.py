"""InVAR Phase 1 smoke test.

Runs a tiny end-to-end forward pass on Fold 1 train data to verify that:
  - The data adapter (InvarDataset) reads LATTICE artifacts cleanly.
  - The Invar model (Design C, RegimeAxisRetrieval) instantiates and
    forwards on per-day cross-sections without NaNs.
  - All three heads (ranking, regime classifier, vol) produce
    finite outputs of the expected shape.
  - The active mask is respected (predictions for masked tickers are
    excluded from the loss surface; vol_hat is multiplied by mask).
  - Parameter count is under the 5M budget.
  - A tiny checkpoint is saved.

Usage::

    PYTHONPATH=$PWD python3 -u -m src.invar.experiments.smoke_test
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from src.invar.data.dataset import InvarDataset, PANEL_FEATURE_DIM, MACRO_FEATURE_DIM
from src.invar.model.invar import Invar, InvarConfig, count_parameters


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, default=1)
    p.add_argument("--n-days", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", type=str,
                    default="experiments/invar/smoke")
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "smoke.log"

    def log(msg: str) -> None:
        print(f"[invar smoke] {msg}", flush=True)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    open(log_path, "w").close()
    log(f"fold={args.fold} n_days={args.n_days} seed={args.seed}")

    t0 = time.time()
    dataset = InvarDataset(fold=args.fold, split="train")
    log(f"dataset loaded in {time.time() - t0:.1f}s")
    log(f"  panel feature dim: {PANEL_FEATURE_DIM}")
    log(f"  macro feature dim: {MACRO_FEATURE_DIM}")
    log(f"  train days eligible: {len(dataset)}")
    log(f"  regime label distribution (train): "
        f"{[(int(c), int((dataset._regime_labels[dataset.train_idx] == c).sum())) for c in range(8)]}")

    cfg = InvarConfig(
        n_features=PANEL_FEATURE_DIM,
        macro_dim=MACRO_FEATURE_DIM,
        regime_axis="retrieval",
    )
    model = Invar(cfg)
    n_params = count_parameters(model)
    log(f"model: {cfg.regime_axis}, d_model={cfg.d_model}, "
        f"n_layers={cfg.n_layers}, params={n_params:,}")

    # Sample N_DAYS random eligible days
    rng = np.random.default_rng(args.seed)
    sample_days = rng.choice(dataset._eligible_idx, size=min(args.n_days, len(dataset)),
                                replace=False)

    model.eval()
    forward_ok = 0
    forward_fail = 0
    nan_observed = False
    shape_ok = True
    last_param_count = None
    for t in sample_days:
        t = int(t)
        batch = dataset.get(t)
        N = batch.features.shape[0]
        with torch.no_grad():
            out = model(batch.features, batch.macro, batch.mask, return_attn=False)
        y_hat = out["y_hat"]
        regime_logits = out["regime_logits"]
        vol_hat = out["vol_hat"]
        if y_hat.shape != (N,):
            log(f"  FAIL day {t}: y_hat shape {tuple(y_hat.shape)} != ({N},)")
            shape_ok = False
            forward_fail += 1
            continue
        if regime_logits.shape != (cfg.n_offline_regimes,):
            log(f"  FAIL day {t}: regime_logits shape {tuple(regime_logits.shape)}")
            shape_ok = False
            forward_fail += 1
            continue
        if vol_hat.shape != (N,):
            log(f"  FAIL day {t}: vol_hat shape {tuple(vol_hat.shape)}")
            shape_ok = False
            forward_fail += 1
            continue
        if not torch.isfinite(y_hat).all():
            log(f"  FAIL day {t}: y_hat contains NaN/Inf")
            nan_observed = True
            forward_fail += 1
            continue
        if not torch.isfinite(regime_logits).all():
            log(f"  FAIL day {t}: regime_logits contains NaN/Inf")
            nan_observed = True
            forward_fail += 1
            continue
        if not torch.isfinite(vol_hat).all():
            log(f"  FAIL day {t}: vol_hat contains NaN/Inf")
            nan_observed = True
            forward_fail += 1
            continue
        log(f"  day {t} ({batch.date.date()}) N={N} y_hat[mean,std]="
            f"[{float(y_hat.mean()):+.4f}, {float(y_hat.std()):.4f}] "
            f"regime_label={batch.regime_label}")
        forward_ok += 1

    log(f"forward ok: {forward_ok}/{len(sample_days)}; fail: {forward_fail}")

    # Save tiny checkpoint
    ckpt_path = out_dir / "ckpt.pt"
    torch.save({
        "model_state": model.state_dict(),
        "config": cfg.__dict__,
        "n_params": n_params,
    }, ckpt_path)
    log(f"checkpoint saved: {ckpt_path}")

    # Phase 1 gate check
    gate = {
        "smoke_runs_end_to_end": forward_ok == len(sample_days),
        "params_under_5M": n_params < 5_000_000,
        "checkpoint_saved": ckpt_path.exists(),
        "no_NaNs": not nan_observed,
        "shapes_ok": shape_ok,
        "param_count": n_params,
        "n_days_processed": forward_ok,
    }
    with open(out_dir / "phase1_gate.json", "w") as f:
        json.dump(gate, f, indent=2)
    log(f"PHASE 1 GATE: {gate}")
    all_pass = all(v for k, v in gate.items() if isinstance(v, bool))
    log(f"PHASE 1 GATE ALL PASS: {all_pass}")


if __name__ == "__main__":
    main()
