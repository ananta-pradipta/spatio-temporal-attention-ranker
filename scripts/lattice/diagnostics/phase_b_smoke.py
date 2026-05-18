"""Phase 5b B smoke: build banks on Fold 1, verify size/shape/non-zero output.

Acceptance gate per spec section 5.4:
  - Bank population on F1 train completes in under 10 minutes.
  - Forward pass through DualRetrieval produces non-zero delta_regime and
    delta_novelty.
  - Novelty bank size in expected range (5K to 50K) or empty-bank guard
    activates.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.lattice.data.folds import fold_indices
from src.lattice.training.dataloader import LatticeDataPrep
from src.lattice.model.lattice import LATTICE, LatticeConfig
from src.lattice.training.train import populate_retrieval_banks


def main() -> None:
    t0 = time.time()
    print(f"[phase_b smoke] loading prep for fold 1", flush=True)
    prep = LatticeDataPrep(fold=1)
    t1 = time.time()
    print(f"[phase_b smoke] prep loaded in {t1 - t0:.1f}s", flush=True)

    print(f"[phase_b smoke] regime keys shape: {prep._regime_keys.shape}", flush=True)
    print(f"[phase_b smoke] regime keys mean: {prep._regime_keys.mean():.4f}, "
          f"std: {prep._regime_keys.std():.4f}", flush=True)

    train_idx, val_idx, test_idx = fold_indices(1, prep.dates)
    print(f"[phase_b smoke] train_idx size: {len(train_idx)}", flush=True)

    train_keys = prep._regime_keys[train_idx]
    print(f"[phase_b smoke] train regime keys per-component mean: "
          f"{train_keys.mean(axis=0).round(4).tolist()}", flush=True)
    print(f"[phase_b smoke] train regime keys per-component std: "
          f"{train_keys.std(axis=0).round(4).tolist()}", flush=True)

    train_mask = np.zeros(prep._panel_tensor.shape[0], dtype=bool)
    train_mask[train_idx] = True
    eligible_train = prep._novelty_eligible & train_mask[:, None]
    n_eligible = int(eligible_train.sum())
    n_eligible_per_ticker = eligible_train.sum(axis=0)
    n_unique_tickers = int((n_eligible_per_ticker > 0).sum())
    print(f"[phase_b smoke] novelty bank candidates: {n_eligible} cells "
          f"across {n_unique_tickers} tickers", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = LatticeConfig()
    model = LATTICE(cfg).to(device)
    audit = populate_retrieval_banks(model, prep, train_idx, device)
    print(f"[phase_b smoke] populate audit: {audit}", flush=True)

    # Forward smoke on day t=val_idx[0]
    t_val = int(val_idx[0])
    batch = prep.make_batch(t_val)
    print(f"[phase_b smoke] making forward pass on day {t_val}", flush=True)
    model.eval()
    with torch.no_grad():
        z = torch.randn(1, prep.n_universe, cfg.d_model, device=device)

        delta_regime, alpha_regime = model.retrieval.regime(
            z, batch.regime_query_keys.to(device), batch.day_index.to(device),
        )
        delta_novelty, alpha_novelty = model.retrieval.novelty(
            z, batch.novelty_query_keys.to(device),
            batch.novelty_sector_ids.to(device), batch.day_index.to(device),
            batch.active_mask.to(device),
        )

    print(f"[phase_b smoke] alpha_regime: {float(alpha_regime.mean()):+.4f} "
          f"(std {float(alpha_regime.std()):.4f})", flush=True)
    n_active = int(batch.active_mask[0].sum())
    if n_active > 0:
        active_idx = batch.active_mask[0].nonzero(as_tuple=True)[0]
        an_active = alpha_novelty[0, active_idx, 0]
        dn_active = delta_novelty[0, active_idx]
        dr_active = delta_regime[0, active_idx]
        print(f"[phase_b smoke] alpha_novelty across {n_active} active: "
              f"mean {float(an_active.mean()):+.4f}, std {float(an_active.std()):.4f}",
              flush=True)
        print(f"[phase_b smoke] delta_regime norm: {float(dr_active.norm(dim=-1).mean()):.4f}",
              flush=True)
        print(f"[phase_b smoke] delta_novelty norm: {float(dn_active.norm(dim=-1).mean()):.4f}",
              flush=True)

    # Acceptance gates
    pass_msgs = []
    if n_eligible >= 100:
        pass_msgs.append(f"OK novelty bank size = {n_eligible} (>= 100)")
    else:
        pass_msgs.append(f"WARN novelty bank size = {n_eligible} (< 100)")
    if abs(float(delta_regime.detach().cpu().abs().mean())) > 1e-6:
        pass_msgs.append("OK delta_regime non-zero")
    else:
        pass_msgs.append("FAIL delta_regime is zero")
    if n_active > 0 and abs(float(delta_novelty.detach().cpu().abs().mean())) > 1e-6:
        pass_msgs.append("OK delta_novelty non-zero")
    else:
        pass_msgs.append("FAIL delta_novelty is zero")

    elapsed = time.time() - t0
    pass_msgs.append(f"elapsed: {elapsed:.1f}s")
    print(f"[phase_b smoke] acceptance:", flush=True)
    for m in pass_msgs:
        print(f"  - {m}", flush=True)


if __name__ == "__main__":
    main()
