"""MAiT trainer. Separate from `train_baseline.py` because MAiT's forward
signature, loss, optimizer, and per-day sampler differ from the existing
baselines, and folding it in via if-branches would muddy that code path.

Dispatched from `train_baseline.py` when `--baseline mait` is set.

Schedule and hyperparameters from `docs/mait_design.md` Section 6:
  optimizer       AdamW, lr=5e-4, weight_decay=1e-4
  schedule        cosine annealing over epochs, no warmup
  epochs_max      30, early stop on val rank IC, patience 5, restore-best
  grad_clip       1.0
  stream_dropout  0.15 (in model)
  loss            mait_loss = ic_loss(y_hat) + 0.5 * ic_loss(s_panel) + 0.05 * g^2

Sampler from Section 4.2: WeightedRandomSampler-style weights by daily
VIX. Low (<18) and mid (18-28) buckets weight 1; high (>=28) weight 2.
Per-day weights are computed once from the train split and used for
np.random.choice across all epochs.

Deviation from existing baseline conventions, per spec:
  - existing baselines use lr=1e-4, wd=1e-5, warmup_steps=500, patience=3, epochs=10
  - MAiT uses lr=5e-4, wd=1e-4, no warmup, patience=5, epochs=30
  This is by design (see docs/mait_design.md Section 4.1).
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.invar.baselines.mait import MAiT, count_parameters, mait_loss
from src.invar.baselines.mait_data import (
    MaitBatchAdapter,
    N_MACRO_KEPT,
    N_PANEL_KEPT,
    REGIME_DIM,
)
from src.invar.data.dataset import InvarDataset
from src.invar.evaluation.metrics import (
    daily_ic,
    daily_rank_ic,
    long_short_sharpe,
    ndcg_at_k,
)
from src.invar.training.train import set_seeds


# Bucket thresholds from design Section 4.2. Hardcoded, do not tune.
_VIX_LOW_THRESHOLD = 18.0
_VIX_HIGH_THRESHOLD = 28.0
_HIGH_VIX_OVERSAMPLE = 2.0


def _build_regime_weights(dataset: InvarDataset) -> np.ndarray:
    """Per-eligible-day sampling weights based on raw VIX.

    Reads raw (pre-z-score) VIX values from the dataset's macro tensor.
    The InvarDataset z-scores macro in-place; we recover raw VIX as
    `_macro_raw_tensor[:, 0]` (VIX is the first column of MACRO_FEATURE_COLS).

    Returns a (n_eligible,) array of weights normalized to sum 1, suitable
    for np.random.choice(p=weights).
    """
    vix_idx = 0
    eligible = np.asarray(dataset._eligible_idx, dtype=np.int64)
    raw_vix = dataset._macro_raw_tensor[eligible, vix_idx]
    weights = np.where(
        raw_vix < _VIX_LOW_THRESHOLD, 1.0,
        np.where(raw_vix < _VIX_HIGH_THRESHOLD, 1.0, _HIGH_VIX_OVERSAMPLE),
    ).astype(np.float64)
    total = weights.sum()
    if total <= 0.0:
        raise RuntimeError("regime weights summed to zero; bad VIX values")
    return weights / total, raw_vix


def _eval_mait(model: MAiT, adapter: MaitBatchAdapter, device: torch.device,
               collect_predictions: bool = False) -> dict:
    """Per-day evaluation for MAiT.

    Runs forward with `train_mode=False` (no stream-dropout, deterministic
    gate), aggregates metrics (IC, rankIC, NDCG@10, NDCG@50, Sharpe, etc.)
    over the dataset, and records the per-day gate value to compute
    `mean_gate`.
    """
    model.eval()
    pred_rows: list[dict] = []
    gate_values: list[float] = []
    with torch.no_grad():
        for batch in adapter.iter_days():
            x_panel = batch.x_panel.to(device)
            x_macro_lookback = batch.x_macro_lookback.to(device)
            regime_input = batch.regime_input.to(device)
            y_hat, s_panel, s_macro, g = model(
                x_panel, x_macro_lookback, regime_input, train_mode=False,
            )
            gate_values.append(float(g.detach().item()))
            yh = y_hat.cpu().numpy()
            yt = batch.y_cs.numpy()
            date_str = batch.date.strftime("%Y-%m-%d")
            tickers = (
                adapter.dataset.tickers_universe
                if not hasattr(batch, "tickers") else None
            )
            # batch is MaitBatch which does not carry tickers; pull from
            # the underlying day index via the dataset.
            inv_day = adapter.dataset.get(batch.day_index)
            for i in range(len(yh)):
                pred_rows.append({
                    "date": date_str, "ticker": inv_day.tickers[i],
                    "y_hat": float(yh[i]), "y_true": float(yt[i]),
                    "sector_id": int(inv_day.sector_id[i]),
                    "size_decile": int(inv_day.size_decile[i]),
                    "age_bucket": int(inv_day.age_bucket[i]),
                })
    if not pred_rows:
        return dict(ic=float("nan"), rank_ic=float("nan"),
                    ndcg10=float("nan"), ndcg50=float("nan"),
                    sharpe=float("nan"), annual_return=float("nan"),
                    annual_vol=float("nan"), max_drawdown=float("nan"),
                    n_days=0, mean_gate=float("nan"),
                    predictions=[] if collect_predictions else None)
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
        n_days=ic["n_days"],
        mean_gate=float(np.mean(gate_values)) if gate_values else float("nan"),
        gate_std=float(np.std(gate_values)) if gate_values else float("nan"),
        predictions=pred_rows if collect_predictions else None,
    )


def train_mait(cfg) -> dict:
    """Train MAiT end to end. Returns the results dict (also written to
    `out_dir/results.json`).
    """
    set_seeds(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[mait] fold={cfg.fold} seed={cfg.seed} device={device}", flush=True)

    train_ds = InvarDataset(fold=cfg.fold, split="train")
    val_ds = InvarDataset(fold=cfg.fold, split="val")
    test_ds = InvarDataset(fold=cfg.fold, split="test")
    train_adapter = MaitBatchAdapter(train_ds)
    val_adapter = MaitBatchAdapter(val_ds, ensure_scaler_persisted=False)
    test_adapter = MaitBatchAdapter(test_ds, ensure_scaler_persisted=False)
    print(f"[mait] train days {len(train_ds)}  "
          f"val days {len(val_ds)}  test days {len(test_ds)}", flush=True)

    # Honor `cfg.epochs` exactly. Phase 4 (full run) passes --epochs 30
    # explicitly per the design spec. Phase 3 smoke passes --epochs 2.
    # Early stopping (patience 5) governs actual stop time.
    n_epochs = cfg.epochs

    model = MAiT(
        n_panel=train_adapter.n_panel_kept,
        n_macro=train_adapter.n_macro_kept,
        L_lookback=60,
        d_model=128, n_heads=4, d_ff=256, n_layers=3, dropout=0.1,
        stream_dropout_p=0.15, regime_dim=REGIME_DIM,
    ).to(device)
    n_params = count_parameters(model)
    print(f"[mait] params={n_params:,}", flush=True)

    # MAiT-specific optimizer and schedule (design Section 4.1).
    optim = AdamW(model.parameters(), lr=5.0e-4, weight_decay=1.0e-4)
    scheduler = CosineAnnealingLR(optim, T_max=max(n_epochs, 1))

    # Regime-balanced per-day sampler weights (design Section 4.2).
    weights_norm, raw_vix = _build_regime_weights(train_ds)
    n_low = int((raw_vix < _VIX_LOW_THRESHOLD).sum())
    n_high = int((raw_vix >= _VIX_HIGH_THRESHOLD).sum())
    n_mid = len(raw_vix) - n_low - n_high
    print(f"[mait] VIX buckets: low={n_low} mid={n_mid} high={n_high}; "
          f"high oversampled {_HIGH_VIX_OVERSAMPLE}x", flush=True)

    rng = np.random.default_rng(cfg.seed)
    eligible = np.asarray(train_ds._eligible_idx, dtype=np.int64)
    n_per_epoch = len(eligible)

    out_dir = Path(cfg.output_dir) / cfg.baseline_name / f"fold{cfg.fold}/seed{cfg.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best_val_rank_ic = -float("inf")
    best_epoch = -1
    epochs_no_improve = 0
    patience = 5
    best_state = None

    for epoch in range(n_epochs):
        t0 = time.time()
        model.train()
        epoch_losses = []
        epoch_gates = []
        day_indices = rng.choice(eligible, size=n_per_epoch,
                                  replace=True, p=weights_norm)
        for t in day_indices:
            batch = train_adapter.adapt(train_ds.get(int(t)))
            x_panel = batch.x_panel.to(device)
            x_macro_lookback = batch.x_macro_lookback.to(device)
            regime_input = batch.regime_input.to(device)
            y_cs = batch.y_cs.to(device)

            optim.zero_grad()
            y_hat, s_panel, s_macro, g = model(
                x_panel, x_macro_lookback, regime_input, train_mode=True,
            )
            loss = mait_loss(y_hat, s_panel, g, y_cs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optim.step()
            epoch_losses.append(float(loss.item()))
            epoch_gates.append(float(g.detach().item()))

        scheduler.step()
        train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        train_gate_mean = float(np.mean(epoch_gates)) if epoch_gates else float("nan")

        val_metrics = _eval_mait(model, val_adapter, device,
                                   collect_predictions=False)
        history.append(dict(
            epoch=epoch, train_loss=train_loss,
            train_gate_mean=train_gate_mean,
            val_ic=val_metrics["ic"], val_rank_ic=val_metrics["rank_ic"],
            val_ndcg10=val_metrics["ndcg10"], val_gate=val_metrics["mean_gate"],
        ))
        print(f"[mait] epoch {epoch}: loss={train_loss:.4f}  "
              f"val_ic={val_metrics['ic']:+.4f}  "
              f"val_rank_ic={val_metrics['rank_ic']:+.4f}  "
              f"train_g={train_gate_mean:.3f}  val_g={val_metrics['mean_gate']:.3f}  "
              f"({time.time() - t0:.1f}s)",
              flush=True)

        if (np.isfinite(val_metrics["rank_ic"])
                and val_metrics["rank_ic"] > best_val_rank_ic):
            best_val_rank_ic = val_metrics["rank_ic"]
            best_epoch = epoch
            epochs_no_improve = 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[mait] early stop at epoch {epoch} "
                      f"(best epoch {best_epoch} val_rank_ic={best_val_rank_ic:+.4f})",
                      flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(best_state, out_dir / "ckpt.pt")
        print(f"[mait] restored best-val checkpoint from epoch {best_epoch}",
              flush=True)

    # Final eval with the restored best model.
    train_eval = _eval_mait(model, train_adapter, device,
                              collect_predictions=False)
    val_eval = _eval_mait(model, val_adapter, device, collect_predictions=False)
    test_eval = _eval_mait(model, test_adapter, device,
                              collect_predictions=cfg.save_predictions)
    pred_rows = test_eval.pop("predictions", None)
    if cfg.save_predictions and pred_rows:
        pd.DataFrame(pred_rows).to_parquet(
            out_dir / "predictions.parquet", index=False,
        )
    print(f"[mait] FINAL test_ic={test_eval['ic']:+.4f}  "
          f"rank_ic={test_eval['rank_ic']:+.4f}  "
          f"sharpe={test_eval['sharpe']:.3f}  "
          f"mean_gate_train={train_eval['mean_gate']:.3f}  "
          f"mean_gate_val={val_eval['mean_gate']:.3f}  "
          f"mean_gate_test={test_eval['mean_gate']:.3f}",
          flush=True)

    result = {
        "baseline": cfg.baseline_name,
        "config": asdict(cfg),
        "n_params": int(n_params),
        "best_val_ic": float(val_eval["ic"]),
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
        "mean_gate_train": float(train_eval["mean_gate"]),
        "mean_gate_val": float(val_eval["mean_gate"]),
        "mean_gate_test": float(test_eval["mean_gate"]),
        "gate_std_test": float(test_eval["gate_std"]),
        "history": history,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    return result
