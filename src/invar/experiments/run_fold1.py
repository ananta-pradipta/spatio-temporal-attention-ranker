"""InVAR Phase 2 Fold 1 sanity run (Design C, seed 42).

Sanity criteria (per spec PHASE 2 GATE):
  - Train loss decreases monotonically over the first 3 epochs.
  - Validation IC is positive by epoch 5.
  - Test IC on Fold 1 is at least +0.020.

Runs a single (fold=1, seed=42, regime_axis=retrieval) training, prints a
phase summary on completion. If sanity criteria fail, prints FAIL and
exits 1.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from src.invar.training.train import TrainConfig, train_one


def main() -> int:
    cfg = TrainConfig(
        fold=1, seed=42, epochs=10,
        output_dir="experiments/invar/fold1",
        regime_axis="retrieval",
        save_predictions=True,
    )
    result = train_one(cfg)

    history = result["history"]
    train_losses = [h["train_loss"] for h in history[:3]]
    losses_decreasing = (
        len(train_losses) >= 3
        and train_losses[1] < train_losses[0]
        and train_losses[2] < train_losses[1]
    )
    val_ic_by_epoch5 = next(
        (h["val_ic"] for h in history if h["epoch"] >= 4), float("nan")
    )
    val_positive_by_5 = val_ic_by_epoch5 > 0.0
    test_ic = result["test_ic"]
    sanity_floor = test_ic >= 0.020

    gate = {
        "train_loss_decreasing_first3": losses_decreasing,
        "val_ic_positive_by_epoch5": val_positive_by_5,
        "test_ic_above_floor_0.020": sanity_floor,
        "test_ic": test_ic,
        "test_rank_ic": result["test_rank_ic"],
        "test_ndcg10": result["test_ndcg10"],
        "test_sharpe": result["test_sharpe"],
        "best_val_ic": result["best_val_ic"],
        "best_epoch": result["best_epoch"],
    }
    out_dir = Path(cfg.output_dir) / f"seed{cfg.seed}_design{cfg.regime_axis[0].upper()}"
    with open(out_dir / "phase2_gate.json", "w") as f:
        json.dump(gate, f, indent=2)
    print(f"\n[invar phase 2 GATE]:", flush=True)
    for k, v in gate.items():
        print(f"  {k}: {v}", flush=True)
    bool_keys = ("train_loss_decreasing_first3", "val_ic_positive_by_epoch5",
                  "test_ic_above_floor_0.020")
    all_pass = all(gate[k] for k in bool_keys)
    print(f"  ALL PASS: {all_pass}", flush=True)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
