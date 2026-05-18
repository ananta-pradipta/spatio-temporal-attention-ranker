"""Phase 5b A: extract gate trajectories, router utilisation, predictions.

Loads each Phase 5 checkpoint under
``experiments/lattice/headline/fold{F}_seed{S}/ckpt.pt``, runs forward-only
inference on the validation and test splits with hooks on the relevant
modules, and writes three artifacts:

  - ``experiments/lattice/diagnostics/gate_trajectories/fold{F}_seed{S}.json``
  - ``experiments/lattice/diagnostics/router_utilisation/fold{F}_seed{S}.json``
  - ``experiments/lattice/diagnostics/predictions/fold{F}_seed{S}.parquet``

The runner is unified rather than split into separate ``extract_gate_trajectories``
and ``extract_router_utilisation`` scripts (per Phase 5b spec section 4.1, 4.2)
so each checkpoint is loaded once. The acceptance gate is the artifact set,
not the file count.

Usage::

    PYTHONPATH=$PWD python3 -u -m scripts.lattice.diagnostics.extract_phase_a \\
        --headline-dir experiments/lattice/headline \\
        --output-dir experiments/lattice/diagnostics

Phase 5 checkpoints have placeholder retrieval banks (regime keys all zero;
novelty numeric keys all zero, sector-embedding lookup is real). Banks are
restored from the state dict; ``_bank_populated`` is forced True after load
to replicate the test-time behavior of the trainer's ``model.train_one()``
call (``populate_retrieval_banks`` is called pre-training and the flag
persists through ``load_state_dict``).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from src.lattice.data.folds import fold_indices
from src.lattice.model.lattice import LATTICE, LatticeConfig
from src.lattice.training.dataloader import LatticeDataPrep, SECTOR_TO_ID
from src.lattice.training.train import populate_retrieval_banks


SECTOR_ID_TO_NAME = {v: k for k, v in SECTOR_TO_ID.items()}


def install_hooks(model: LATTICE) -> dict[str, list]:
    """Install forward hooks. Returns a dict of capture buffers cleared per day."""
    captures: dict[str, list] = {
        "blend_logits": [],
        "alpha_regime": [],
        "alpha_novelty": [],
        "router_logits": [],
    }

    def _blend_hook(_mod, _inp, out: Tensor) -> None:
        captures["blend_logits"].append(out.detach().cpu())

    def _regime_hook(_mod, _inp, out: tuple[Tensor, Tensor]) -> None:
        _, alpha = out
        captures["alpha_regime"].append(alpha.detach().cpu())

    def _novelty_hook(_mod, _inp, out: tuple[Tensor, Tensor]) -> None:
        _, alpha = out
        captures["alpha_novelty"].append(alpha.detach().cpu())

    def _router_hook(_mod, _inp, out: Tensor) -> None:
        captures["router_logits"].append(out.detach().cpu())

    handles = [
        model.graph.blend_mlp.register_forward_hook(_blend_hook),
        model.retrieval.regime.register_forward_hook(_regime_hook),
        model.retrieval.novelty.register_forward_hook(_novelty_hook),
        model.macro_router.router.register_forward_hook(_router_hook),
    ]
    return captures, handles


def reset_captures(captures: dict[str, list]) -> None:
    for k in captures:
        captures[k].clear()


def run_split(
    model: LATTICE,
    prep: LatticeDataPrep,
    indices: np.ndarray,
    device: torch.device,
    captures: dict[str, list],
) -> dict[str, Any]:
    """Run forward on each day in indices; collect per-day stats and predictions."""
    model.eval()

    per_day: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for t in indices:
            t = int(t)
            reset_captures(captures)
            batch = prep.make_batch(t)

            out, _ = model(
                batch.panel_features.to(device),
                batch.macro_state.to(device),
                batch.cohort_size_decile.to(device),
                batch.cohort_liquidity_decile.to(device),
                batch.cohort_sector_id.to(device),
                batch.cohort_age_bucket.to(device),
                batch.regime_query_keys.to(device),
                batch.novelty_query_keys.to(device),
                batch.novelty_sector_ids.to(device),
                batch.active_mask.to(device),
                batch.day_index.to(device),
                batch.corr_neighbor_idx.to(device),
                batch.corr_neighbor_mask.to(device),
            )

            yh = out[0].detach().cpu()
            yt = batch.y_target[0]
            mask = batch.active_mask[0].cpu()
            n_active = int(mask.sum().item())

            blend_logits = captures["blend_logits"][0] if captures["blend_logits"] else None
            alpha_blend = (
                torch.sigmoid(blend_logits).flatten().item() if blend_logits is not None else float("nan")
            )

            ar = captures["alpha_regime"][0] if captures["alpha_regime"] else None
            alpha_regime = float(ar.flatten().item()) if ar is not None else float("nan")

            an = captures["alpha_novelty"][0] if captures["alpha_novelty"] else None
            if an is not None:
                an_active = an.view(-1)[mask.view(-1).bool()]
                if an_active.numel() > 0:
                    alpha_novelty_mean = float(an_active.mean().item())
                    alpha_novelty_std = (
                        float(an_active.std().item()) if an_active.numel() > 1 else 0.0
                    )
                else:
                    alpha_novelty_mean = float("nan")
                    alpha_novelty_std = float("nan")
            else:
                alpha_novelty_mean = float("nan")
                alpha_novelty_std = float("nan")

            rl = captures["router_logits"][0] if captures["router_logits"] else None
            if rl is not None:
                router_probs = torch.softmax(rl, dim=-1).flatten().tolist()
            else:
                router_probs = []

            per_day.append({
                "day_index": t,
                "date": pd.Timestamp(prep.dates[t]).strftime("%Y-%m-%d"),
                "n_active": n_active,
                "alpha_blend": alpha_blend,
                "alpha_regime": alpha_regime,
                "alpha_novelty_mean": alpha_novelty_mean,
                "alpha_novelty_std": alpha_novelty_std,
                "router_probs": router_probs,
            })

            if n_active >= 5:
                date_str = pd.Timestamp(prep.dates[t]).strftime("%Y-%m-%d")
                size_arr = batch.cohort_size_decile[0].numpy()
                liq_arr = batch.cohort_liquidity_decile[0].numpy()
                sec_arr = batch.cohort_sector_id[0].numpy()
                age_arr = batch.cohort_age_bucket[0].numpy()
                yh_np = yh.numpy()
                yt_np = yt.numpy()
                mask_np = mask.numpy()
                tickers = batch.tickers
                for i in range(len(tickers)):
                    if not mask_np[i]:
                        continue
                    pred_rows.append({
                        "date": date_str,
                        "ticker": tickers[i],
                        "y_hat": float(yh_np[i]),
                        "y_true": float(yt_np[i]),
                        "sector_id": int(sec_arr[i]),
                        "sector_name": SECTOR_ID_TO_NAME.get(int(sec_arr[i]), "UNKNOWN"),
                        "size_decile": int(size_arr[i]),
                        "liquidity_decile": int(liq_arr[i]),
                        "age_bucket": int(age_arr[i]),
                    })

    return {"per_day": per_day, "predictions": pred_rows}


def aggregate_router(per_day: list[dict]) -> dict[str, Any]:
    """Compute per-fold-per-seed router utilisation summary."""
    probs = np.array([row["router_probs"] for row in per_day if row["router_probs"]])
    if probs.size == 0:
        return {
            "n_days": 0,
            "expert_mean_utilisation": [],
            "mean_routing_entropy": float("nan"),
            "max_per_day_routing_weight": float("nan"),
            "p95_per_day_routing_weight": float("nan"),
            "frac_days_argmax_top": float("nan"),
        }
    mean_util = probs.mean(axis=0)
    eps = 1e-12
    entropy_per_day = -np.sum(probs * np.log(probs + eps), axis=1)
    mean_entropy = float(entropy_per_day.mean())
    max_per_day = probs.max(axis=1)
    return {
        "n_days": int(probs.shape[0]),
        "expert_mean_utilisation": [float(x) for x in mean_util],
        "mean_routing_entropy": mean_entropy,
        "max_routing_entropy_possible": float(np.log(probs.shape[1])),
        "max_per_day_routing_weight_max": float(max_per_day.max()),
        "max_per_day_routing_weight_mean": float(max_per_day.mean()),
        "p95_per_day_routing_weight": float(np.percentile(max_per_day, 95)),
        "frac_days_argmax_top1": float(np.mean(max_per_day > 0.95)),
    }


def aggregate_gates(per_day: list[dict]) -> dict[str, Any]:
    """Compute per-fold-per-seed gate trajectory summary."""
    if not per_day:
        return {}
    arr_blend = np.array([r["alpha_blend"] for r in per_day])
    arr_regime = np.array([r["alpha_regime"] for r in per_day])
    arr_novelty_mean = np.array([r["alpha_novelty_mean"] for r in per_day])
    arr_novelty_std = np.array([r["alpha_novelty_std"] for r in per_day])

    def _stat(x: np.ndarray) -> dict[str, float]:
        x = x[np.isfinite(x)]
        if x.size == 0:
            return {"mean": float("nan"), "std": float("nan"),
                    "min": float("nan"), "max": float("nan")}
        return {"mean": float(x.mean()), "std": float(x.std()),
                "min": float(x.min()), "max": float(x.max())}

    return {
        "n_days": len(per_day),
        "alpha_blend": _stat(arr_blend),
        "alpha_regime": _stat(arr_regime),
        "alpha_novelty_per_day_mean": _stat(arr_novelty_mean),
        "alpha_novelty_per_day_std": _stat(arr_novelty_std),
    }


def process_one(
    fold: int,
    seed: int,
    headline_dir: Path,
    output_dir: Path,
    device: torch.device,
    prep_cache: dict[int, LatticeDataPrep],
) -> dict[str, Any] | None:
    """Process one (fold, seed) ckpt; write 3 artifacts; return summary."""
    ckpt_path = headline_dir / f"fold{fold}_seed{seed}" / "ckpt.pt"
    if not ckpt_path.exists():
        print(f"[phase_a] missing ckpt: {ckpt_path}; skipping", flush=True)
        return None

    if fold not in prep_cache:
        prep_cache[fold] = LatticeDataPrep(fold=fold)
    prep = prep_cache[fold]
    train_idx, val_idx, test_idx = fold_indices(fold, prep.dates)

    cfg = LatticeConfig()
    model = LATTICE(cfg).to(device)
    # Resize the retrieval-bank buffers to match the trainer's pre-training
    # shapes ([n_train, 14] regime, [n_entries, 24] novelty). load_state_dict
    # then overwrites these with the saved (placeholder zero-key) values.
    populate_retrieval_banks(model, prep, train_idx, device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.retrieval.regime._bank_populated = True
    model.retrieval.novelty._bank_populated = True

    captures, handles = install_hooks(model)

    print(f"[phase_a] fold={fold} seed={seed} val={len(val_idx)} test={len(test_idx)}", flush=True)
    val_out = run_split(model, prep, val_idx, device, captures)
    test_out = run_split(model, prep, test_idx, device, captures)

    for h in handles:
        h.remove()

    gate_summary = {
        "fold": fold,
        "seed": seed,
        "init_values": {
            "alpha_blend": 0.5,
            "alpha_regime": 0.27,
            "alpha_novelty": 0.27,
            "lambda_macro": 0.05,
        },
        "lambda_macro": float(torch.sigmoid(model.head.lambda_macro_bias.detach().cpu()).item()),
        "lambda_macro_bias": float(model.head.lambda_macro_bias.detach().cpu().item()),
        "val": {
            "summary": aggregate_gates(val_out["per_day"]),
            "per_day": val_out["per_day"],
        },
        "test": {
            "summary": aggregate_gates(test_out["per_day"]),
            "per_day": test_out["per_day"],
        },
    }
    router_summary = {
        "fold": fold,
        "seed": seed,
        "n_experts": model.macro_router.cfg.n_experts,
        "val": aggregate_router(val_out["per_day"]),
        "test": aggregate_router(test_out["per_day"]),
        "test_per_day_probs": [r["router_probs"] for r in test_out["per_day"]],
    }

    gate_path = output_dir / "gate_trajectories" / f"fold{fold}_seed{seed}.json"
    router_path = output_dir / "router_utilisation" / f"fold{fold}_seed{seed}.json"
    pred_path = output_dir / "predictions" / f"fold{fold}_seed{seed}.parquet"
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    router_path.parent.mkdir(parents=True, exist_ok=True)
    pred_path.parent.mkdir(parents=True, exist_ok=True)

    with open(gate_path, "w") as f:
        json.dump(gate_summary, f, indent=2)
    with open(router_path, "w") as f:
        json.dump(router_summary, f, indent=2)

    pred_rows = test_out["predictions"]
    if pred_rows:
        df_pred = pd.DataFrame(pred_rows)
        df_pred["fold"] = fold
        df_pred["seed"] = seed
        df_pred.to_parquet(pred_path, index=False)

    print(f"[phase_a] fold={fold} seed={seed} done; wrote {gate_path}, {router_path}, {pred_path}",
          flush=True)
    return {"gate": gate_summary, "router": router_summary, "n_pred_rows": len(pred_rows)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--headline-dir", type=str, default="experiments/lattice/headline")
    p.add_argument("--output-dir", type=str, default="experiments/lattice/diagnostics")
    p.add_argument("--folds", type=str, default="1,2,3")
    p.add_argument("--seeds", type=str, default="42,43,44,45,46")
    p.add_argument("--cpu", action="store_true",
                    help="Force CPU even if CUDA is available.")
    args = p.parse_args()

    device = torch.device("cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu")
    print(f"[phase_a] device={device}", flush=True)

    headline_dir = Path(args.headline_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    folds = [int(x) for x in args.folds.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    prep_cache: dict[int, LatticeDataPrep] = {}

    for fold in folds:
        for seed in seeds:
            try:
                process_one(fold, seed, headline_dir, output_dir, device, prep_cache)
            except Exception as exc:
                print(f"[phase_a] fold={fold} seed={seed} FAILED: {exc!r}", flush=True)
                raise


if __name__ == "__main__":
    main()
