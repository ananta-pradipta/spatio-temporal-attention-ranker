"""MASTER training loop on the biotech panel.

Uses the vendored MASTER architecture (SJTU-DMTai, AAAI 2024) with our
conventions: walk-forward folds, cross-sectional MSE on z-scored 5d
forward log returns, 5-day embargo, per-day IC + rank-IC, 5-seed
evaluation.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.baselines.master_adapter import MASTERAdapter
from src.investigation.regime_memory.signature import (
    compute_extended_signatures, forward_fill_signatures,
)
from src.mtgn.data.risk_features import build_risk_features
from src.mtgn.training.losses import cs_mse_loss
from src.mtgn.training.panel_enriched import (
    EnrichedPanelConfig, build_enriched_panel, panel_to_tensors,
)
from src.mtgn.training.train import information_coefficient, rank_ic
from src.mtgn.training.train_baselines import walk_forward_fold


LOG_RET_INDEX = 0


@dataclass
class MASTERTrainConfig:
    start_date: str = "2015-01-01"
    end_date: str = "2022-12-31"
    horizon_days: int = 5
    max_tickers: int | None = 100
    fold: int = 1
    epochs: int = 15
    patience: int = 5
    seed: int = 42
    lr: float = 1e-4
    weight_decay: float = 1e-5
    warmup_steps: int = 500
    lr_schedule: str = "cosine"
    # MASTER-specific
    d_model: int = 128
    t_nhead: int = 4
    s_nhead: int = 4
    T_dropout_rate: float = 0.1
    S_dropout_rate: float = 0.1
    beta: float = 5.0
    temporal_window: int = 20


def train(cfg: MASTERTrainConfig) -> dict:
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    panel_cfg = EnrichedPanelConfig(
        start_date=cfg.start_date, end_date=cfg.end_date,
        horizon_days=cfg.horizon_days, max_tickers=cfg.max_tickers,
    )
    panel, tickers, dates = build_enriched_panel(panel_cfg)
    tensors = panel_to_tensors(panel, tickers, dates)
    x = torch.from_numpy(tensors["x"]).to(device)
    y = torch.from_numpy(tensors["y"]).to(device)
    mask_all = torch.from_numpy(tensors["mask"]).to(device)
    T, N, Fdim = x.shape
    print(f"panel: T={T} N={N} F={Fdim}")

    tr, va, te = walk_forward_fold(dates, cfg.fold, cfg.horizon_days)
    print(f"split [fold{cfg.fold}]: train {tr.stop - tr.start}d  val {va.stop - va.start}d  test {te.stop - te.start}d")

    mu = x[tr].reshape(-1, Fdim).mean(dim=0)
    sd = x[tr].reshape(-1, Fdim).std(dim=0).clamp(min=1e-6)
    x = (x - mu) / sd

    # Build causal 7-dim regime signature (market gate input for MASTER)
    log_returns_np = tensors["x"][:, :, LOG_RET_INDEX]
    mask_np = tensors["mask"]
    risk_df = build_risk_features(cfg.start_date, cfg.end_date)
    sigs = compute_extended_signatures(log_returns_np, mask_np, dates, risk_df)
    sigs = forward_fill_signatures(sigs)
    # z-score using train-fold stats
    sig_train = sigs[tr]
    finite = np.all(np.isfinite(sig_train), axis=1)
    sig_mu = sig_train[finite].mean(axis=0)
    sig_sd = sig_train[finite].std(axis=0).clip(min=1e-6)
    sigs_z = (sigs - sig_mu) / sig_sd
    sigs_z = np.where(np.isfinite(sigs_z), sigs_z, 0.0).astype(np.float32)
    sigs_z_t = torch.from_numpy(sigs_z).to(device)
    d_gate = sigs.shape[1]  # 7
    print(f"market gate input: {d_gate} dims (extended regime signature)")

    # MASTER expects input [N, W, d_feat + d_gate]; stock features first,
    # then market (gate) features broadcast across W.
    model = MASTERAdapter(
        d_feat=Fdim, d_model=cfg.d_model,
        t_nhead=cfg.t_nhead, s_nhead=cfg.s_nhead,
        gate_input_start_index=Fdim,
        gate_input_end_index=Fdim + d_gate,
        T_dropout_rate=cfg.T_dropout_rate,
        S_dropout_rate=cfg.S_dropout_rate,
        beta=cfg.beta,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    train_days = tr.stop - tr.start
    total_steps = max(1, cfg.epochs * train_days)

    def _lr_lambda(step):
        if cfg.warmup_steps > 0 and step < cfg.warmup_steps:
            return max(1e-3, (step + 1) / cfg.warmup_steps)
        progress = (step - cfg.warmup_steps) / max(1, total_steps - cfg.warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)

    W = cfg.temporal_window

    def run_epoch(sl, train_: bool):
        model.train(train_)
        losses, preds, ys, ms = [], [], [], []
        for t in range(sl.start + W, sl.stop):
            m = mask_all[t]
            if m.sum() < 3:
                continue
            active_idx = m.nonzero(as_tuple=True)[0]
            # Stock features [N_active, W, F]
            stock_feat = x[t - W + 1: t + 1, active_idx, :].transpose(0, 1)
            # Market signature broadcast across W → [N_active, W, d_gate]
            # MASTER's gate uses ONLY the last timestep but we provide all for compatibility
            A = active_idx.shape[0]
            market_broadcast = sigs_z_t[t - W + 1: t + 1]            # [W, d_gate]
            market_broadcast = market_broadcast.unsqueeze(0).expand(A, W, d_gate)
            # Concat along feature axis → [N_active, W, F + d_gate]
            inp = torch.cat([stock_feat, market_broadcast], dim=-1)
            y_hat = model(inp)                                        # [N_active]
            # Scatter to [N] for cs-mse aligned with the full mask
            y_full = torch.zeros(N, device=device, dtype=y_hat.dtype)
            y_full[active_idx] = y_hat
            l = cs_mse_loss(y_full, y[t], m)
            if train_:
                opt.zero_grad(); l.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); scheduler.step()
            losses.append(l.item())
            preds.append(y_full.detach().cpu().numpy()[None, :])
            ys.append(y[t].detach().cpu().numpy()[None, :])
            ms.append(m.detach().cpu().numpy()[None, :])
        yhat_arr = np.concatenate(preds) if preds else np.zeros((0, N))
        y_arr   = np.concatenate(ys)    if ys    else np.zeros((0, N))
        m_arr   = np.concatenate(ms)    if ms    else np.zeros((0, N), dtype=bool)
        return (float(np.mean(losses)) if losses else float("nan"),
                yhat_arr, y_arr, m_arr)

    history = []
    best_val_ic = -float("inf"); best_state = None; best_epoch = -1; stale = 0
    for epoch in range(cfg.epochs):
        t0 = time.time()
        train_loss, _, _, _ = run_epoch(tr, True)
        val_loss, vp, vy, vm = run_epoch(va, False)
        v_ic = information_coefficient(vp, vy, vm)
        v_r = rank_ic(vp, vy, vm)
        dt = time.time() - t0
        improved = v_ic > best_val_ic
        print(f"[{epoch:02d}] train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"val_ic={v_ic:+.4f}  val_rank_ic={v_r:+.4f}  ({dt:.1f}s){'  *best*' if improved else ''}")
        history.append(dict(epoch=epoch, train_loss=train_loss, val_loss=val_loss,
                            val_ic=v_ic, val_rank_ic=v_r, time_sec=round(dt, 2)))
        if improved:
            best_val_ic = v_ic
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch; stale = 0
        else:
            stale += 1
            if stale >= cfg.patience:
                print(f"early stop epoch {epoch}  best={best_val_ic:+.4f} @ {best_epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    _, tp, ty, tm = run_epoch(te, False)
    test_ic = information_coefficient(tp, ty, tm)
    test_r = rank_ic(tp, ty, tm)
    print(f"\nTEST ic={test_ic:+.4f}  rank_ic={test_r:+.4f}")

    return dict(
        panel_T=T, panel_N=N, panel_F=Fdim,
        test_ic=test_ic, test_rank_ic=test_r,
        best_val_ic=best_val_ic, best_epoch=best_epoch,
        history=history, config=asdict(cfg), fold=cfg.fold,
        _test_preds=tp.astype(np.float32),
        _test_y=ty.astype(np.float32),
        _test_mask=tm.astype(bool),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fold", type=int, required=True, choices=[1, 2, 3])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--max-tickers", type=int, default=None,
                   help="Cap on universe size. None = all active tickers under the v2 mask.")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    cfg = MASTERTrainConfig(fold=args.fold, seed=args.seed,
                            max_tickers=args.max_tickers)
    result = train(cfg)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tp = result.pop("_test_preds"); ty = result.pop("_test_y"); tm = result.pop("_test_mask")
    args.output.write_text(json.dumps(result, indent=2, default=str))
    np.savez_compressed(args.output.with_suffix(".npz"), preds=tp, y=ty, mask=tm)
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
