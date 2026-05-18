"""LATTICE training loop.

Per spec section 7.2:
  AdamW lr=1e-4, weight_decay=1e-5, 500-step warmup, cosine to 10%,
  grad clip norm 1, 10 epochs max, 3-epoch early stop on val rank IC,
  mixed precision via torch.amp.autocast.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from src.lattice.data.folds import fold_indices
from src.lattice.model.lattice import LATTICE, LatticeConfig
from src.lattice.training.dataloader import LatticeDataPrep, LatticeDayBatch
from src.lattice.training.loss import (
    LossConfig, cohort_stratified_ranking_loss, top_decile_hinge_loss,
    soft_spearman,
)


@dataclass
class TrainConfig:
    fold: int = 1
    seed: int = 42
    lr: float = 1e-4
    weight_decay: float = 1e-5
    warmup_steps: int = 500
    grad_clip: float = 1.0
    epochs: int = 10
    early_stop_patience: int = 3
    balance_loss_weight: float = 0.01
    top_decile_hinge_weight: float = 0.05
    output_dir: str = "experiments/lattice/headline"
    save_predictions: bool = False


def set_seeds(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def warmup_cosine_lr(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))


def populate_retrieval_banks(
    model: LATTICE, prep: LatticeDataPrep, train_idx: np.ndarray,
    device: torch.device,
) -> dict:
    """Populate the regime + novelty banks from training-fold cells.

    Phase 5b: keys are constructed from real per-day / per-(day, ticker)
    features (regime: 14-d per spec section 5.1; novelty: 8-d per section
    5.2) using train-fold-only standardisation. Banks are pre-computed in
    ``LatticeDataPrep._build_episode_keys`` so this function only does the
    per-fold filtering and the value-projection forward pass.

    Returns a small audit dict logging bank sizes and eligibility counts.
    """
    audit: dict[str, int] = {}

    # Regime bank: one entry per training day with the prep's pre-computed
    # 14-d standardised key. Falls back to zeros if episode keys are not
    # built (e.g., legacy fixtures).
    if prep._regime_keys is not None:
        regime_keys = torch.from_numpy(prep._regime_keys[train_idx].copy()).to(device)
    else:
        regime_keys = torch.zeros(len(train_idx), 14, device=device)
    regime_day_indices = torch.from_numpy(train_idx.astype(np.int64)).to(device)
    model.retrieval.regime.populate_bank(regime_keys, regime_day_indices)
    audit["regime_bank_size"] = int(regime_keys.shape[0])

    # Novelty bank: every (training day, training ticker) cell where the
    # ticker is active AND a recent IPO with months_since_ipo <= 36.
    train_set = np.zeros(prep._panel_tensor.shape[0], dtype=bool)
    train_set[train_idx] = True
    if prep._novelty_eligible is not None:
        eligible = prep._novelty_eligible & train_set[:, None]
        day_idx_arr, ticker_idx_arr = np.where(eligible)
        sector_arr = np.array(
            [prep._sector_per_ticker[ti] if prep._sector_per_ticker[ti] >= 0 else 0
             for ti in ticker_idx_arr], dtype=np.int64,
        )
        keys_numeric_np = prep._novelty_keys[day_idx_arr, ticker_idx_arr]
    else:
        day_idx_arr = np.empty(0, dtype=np.int64)
        ticker_idx_arr = np.empty(0, dtype=np.int64)
        sector_arr = np.empty(0, dtype=np.int64)
        keys_numeric_np = np.empty((0, 8), dtype=np.float32)

    audit["novelty_eligible_train_cells"] = int(day_idx_arr.size)
    if day_idx_arr.size == 0:
        # Empty-bank guard: load a single zero-vector entry so the model's
        # forward pass treats the bank as not-populated and emits zero
        # delta_novelty. The model's existing check
        # (``self.bank_keys.shape[0] < 2``) handles the degenerate case.
        keys_numeric = torch.zeros(1, 8, device=device)
        sector_ids = torch.zeros(1, dtype=torch.long, device=device)
        day_indices = torch.zeros(1, dtype=torch.long, device=device)
    else:
        keys_numeric = torch.from_numpy(keys_numeric_np.copy()).float().to(device)
        sector_ids = torch.from_numpy(sector_arr).to(device)
        day_indices = torch.from_numpy(day_idx_arr.astype(np.int64)).to(device)

    model.retrieval.novelty.populate_bank(keys_numeric, sector_ids, day_indices)
    audit["novelty_bank_size"] = int(keys_numeric.shape[0])
    return audit


def evaluate(model: LATTICE, prep: LatticeDataPrep,
              eval_idx: np.ndarray, device: torch.device,
              loss_cfg: LossConfig,
              collect_predictions: bool = False) -> dict:
    """Compute mean rank IC and Pearson IC over eval days.

    If ``collect_predictions`` is True, also returns a list of prediction
    rows under the ``predictions`` key with schema
    ``(date, ticker, y_hat, y_true, sector_id, size_decile, liquidity_decile,
    age_bucket)``.
    """
    model.eval()
    rank_ics = []
    pearson_ics = []
    pred_rows: list = []
    with torch.no_grad():
        for t in eval_idx:
            t = int(t)
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
            yh = out[0].cpu()
            yt = batch.y_target[0]
            m = batch.active_mask[0]
            if m.sum() < 5:
                continue
            yh_a = yh[m]; yt_a = yt[m]
            if yh_a.std() < 1e-9 or yt_a.std() < 1e-9:
                continue
            try:
                from scipy.stats import spearmanr
                rho, _ = spearmanr(yh_a.numpy(), yt_a.numpy())
                rank_ics.append(float(rho))
            except Exception:
                rank_ics.append(float(soft_spearman(yh_a, yt_a, 0.001)))
            pearson_ics.append(float(np.corrcoef(yh_a.numpy(), yt_a.numpy())[0, 1]))
            if collect_predictions:
                date_str = pd.Timestamp(prep.dates[t]).strftime("%Y-%m-%d")
                m_np = m.numpy()
                yh_np = yh.numpy()
                yt_np = yt.numpy()
                size_arr = batch.cohort_size_decile[0].numpy()
                liq_arr = batch.cohort_liquidity_decile[0].numpy()
                sec_arr = batch.cohort_sector_id[0].numpy()
                age_arr = batch.cohort_age_bucket[0].numpy()
                tickers = batch.tickers
                for i in range(len(tickers)):
                    if not m_np[i]:
                        continue
                    pred_rows.append({
                        "date": date_str,
                        "ticker": tickers[i],
                        "y_hat": float(yh_np[i]),
                        "y_true": float(yt_np[i]),
                        "sector_id": int(sec_arr[i]),
                        "size_decile": int(size_arr[i]),
                        "liquidity_decile": int(liq_arr[i]),
                        "age_bucket": int(age_arr[i]),
                    })
    return {
        "rank_ic": float(np.mean(rank_ics)) if rank_ics else 0.0,
        "pearson_ic": float(np.mean(pearson_ics)) if pearson_ics else 0.0,
        "n_eval_days": len(rank_ics),
        "predictions": pred_rows,
    }


def train_one(cfg: TrainConfig) -> dict:
    set_seeds(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[lattice train] fold={cfg.fold} seed={cfg.seed} device={device}", flush=True)

    prep = LatticeDataPrep(fold=cfg.fold)
    train_idx, val_idx, test_idx = fold_indices(cfg.fold, prep.dates)
    print(f"[lattice train] train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}",
          flush=True)

    model_cfg = LatticeConfig()
    model = LATTICE(model_cfg).to(device)
    populate_retrieval_banks(model, prep, train_idx, device)

    optim = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = len(train_idx) * cfg.epochs
    scheduler = LambdaLR(optim, lambda step: warmup_cosine_lr(step, cfg.warmup_steps, total_steps))

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    loss_cfg = LossConfig(balance_loss_weight=cfg.balance_loss_weight,
                          top_decile_hinge_weight=cfg.top_decile_hinge_weight)

    history = []
    best_val = -1.0
    best_epoch = -1
    epochs_no_improve = 0
    out_dir = Path(cfg.output_dir) / f"fold{cfg.fold}_seed{cfg.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg.epochs):
        model.train()
        perm = np.random.permutation(train_idx)
        epoch_losses = []
        for t in perm:
            t = int(t)
            batch = prep.make_batch(t)
            optim.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
                out, balance_loss = model(
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
                yh = out[0]
                yt = batch.y_target[0].to(device)
                losses = cohort_stratified_ranking_loss(
                    yh, yt, batch.active_mask[0].to(device),
                    batch.cohort_size_decile[0].to(device),
                    batch.cohort_liquidity_decile[0].to(device),
                    batch.cohort_sector_id[0].to(device),
                    batch.cohort_age_bucket[0].to(device),
                    loss_cfg,
                )
                # Top-decile hinge (auxiliary)
                m_active = batch.active_mask[0].to(device)
                if m_active.sum() >= 10:
                    yt_active = yt[m_active]
                    z_target_active = (yt_active - yt_active.mean()) / yt_active.std().clamp(min=1e-6)
                    z_target_full = torch.zeros_like(yt)
                    z_target_full[m_active] = z_target_active
                    hinge = top_decile_hinge_loss(yh, z_target_full, m_active, loss_cfg)
                else:
                    hinge = torch.zeros((), device=device)
                loss = (losses["loss_main"]
                        + cfg.balance_loss_weight * balance_loss
                        + cfg.top_decile_hinge_weight * hinge)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optim); scaler.update(); scheduler.step()
            epoch_losses.append(float(loss.item()))

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        val_metrics = evaluate(model, prep, val_idx, device, loss_cfg)
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_rank_ic": val_metrics["rank_ic"],
                        "val_pearson_ic": val_metrics["pearson_ic"]})
        print(f"[lattice train] epoch {epoch}: train_loss={train_loss:.4f}  "
              f"val_rank_ic={val_metrics['rank_ic']:+.4f}  "
              f"val_pearson_ic={val_metrics['pearson_ic']:+.4f}",
              flush=True)

        if val_metrics["rank_ic"] > best_val:
            best_val = val_metrics["rank_ic"]
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), out_dir / "ckpt.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.early_stop_patience:
                print(f"[lattice train] early stop at epoch {epoch}", flush=True)
                break

    # Reload best and evaluate on test
    model.load_state_dict(torch.load(out_dir / "ckpt.pt"))
    # Banks were populated in this train_one() call before training; flag
    # persists through state_dict load (it is a Python attribute, not a buffer).
    test_metrics = evaluate(
        model, prep, test_idx, device, loss_cfg,
        collect_predictions=cfg.save_predictions,
    )
    val_metrics_final = evaluate(model, prep, val_idx, device, loss_cfg)
    if cfg.save_predictions and test_metrics["predictions"]:
        df_pred = pd.DataFrame(test_metrics["predictions"])
        df_pred["fold"] = cfg.fold
        df_pred["seed"] = cfg.seed
        df_pred.to_parquet(out_dir / "predictions.parquet", index=False)
    print(f"[lattice train] test rank_ic={test_metrics['rank_ic']:+.4f}  "
          f"pearson_ic={test_metrics['pearson_ic']:+.4f}",
          flush=True)

    result = {
        "config": asdict(cfg),
        "fold": cfg.fold,
        "seed": cfg.seed,
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "best_epoch": int(best_epoch),
        "best_val_rank_ic": float(best_val),
        "test_rank_ic": float(test_metrics["rank_ic"]),
        "test_pearson_ic": float(test_metrics["pearson_ic"]),
        "val_rank_ic_final": float(val_metrics_final["rank_ic"]),
        "val_pearson_ic_final": float(val_metrics_final["pearson_ic"]),
        "history": history,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    return result


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--output-dir", type=str, default="experiments/lattice/headline")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--save-predictions", action="store_true",
                    help="Write a predictions.parquet alongside results.json.")
    args = p.parse_args()
    cfg = TrainConfig(fold=args.fold, seed=args.seed,
                       output_dir=args.output_dir, epochs=args.epochs,
                       save_predictions=args.save_predictions)
    train_one(cfg)


if __name__ == "__main__":
    main()
