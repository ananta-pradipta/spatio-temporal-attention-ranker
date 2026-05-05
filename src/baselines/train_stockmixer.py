"""StockMixer training loop on the biotech panel.

Uses the vendored StockMixer architecture (SJTU-DMTai, AAAI 2024)
with our conventions: walk-forward folds, cross-sectional MSE loss
on z-scored 5d forward returns, 5-day embargo, per-day IC and
rank-IC, 5-seed evaluation.

Output: JSON result + .npz with per-day predictions (for fold-2
Diagnostic 4 consumption).
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

from src.baselines.stockmixer_adapter import StockMixerAdapter
from src.mtgn.training.losses import cs_mse_loss
from src.mtgn.training.panel_enriched import (
    EnrichedPanelConfig, build_enriched_panel, panel_to_tensors,
)
from src.mtgn.training.train import information_coefficient, rank_ic, temporal_split
from src.mtgn.training.train_baselines import walk_forward_fold


@dataclass
class StockMixerTrainConfig:
    start_date: str = "2015-01-01"
    end_date: str = "2022-12-31"
    horizon_days: int = 5
    max_tickers: int | None = 100
    fold: int = 3
    epochs: int = 15
    patience: int = 5
    seed: int = 42
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    lr: float = 1e-4
    weight_decay: float = 1e-5
    warmup_steps: int = 500
    lr_schedule: str = "cosine"
    temporal_window: int = 16           # StockMixer conv stride=2 requires even W; their default 16
    market_hidden: int = 20             # NoGraphMixer hidden dim


def build_slices(dates, cfg):
    if cfg.fold == 0:
        T = len(dates)
        return temporal_split(T, cfg.val_fraction, cfg.test_fraction), "single"
    return walk_forward_fold(dates, cfg.fold, cfg.horizon_days), f"fold{cfg.fold}"


def train(cfg: StockMixerTrainConfig) -> dict:
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
    print(f"panel: T={T} N={N} F={Fdim}  dates {dates[0].date()}..{dates[-1].date()}")

    slices, fold_label = build_slices(dates, cfg)
    tr, va, te = slices
    print(f"split [{fold_label}]: train {tr.stop - tr.start}d  val {va.stop - va.start}d  test {te.stop - te.start}d")

    mu = x[tr].reshape(-1, Fdim).mean(dim=0)
    sd = x[tr].reshape(-1, Fdim).std(dim=0).clamp(min=1e-6)
    x = (x - mu) / sd

    W = cfg.temporal_window
    model = StockMixerAdapter(
        stocks_pad=N, time_steps=W, channels=Fdim, market=cfg.market_hidden,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    train_days = tr.stop - tr.start
    total_steps = max(1, cfg.epochs * train_days)

    def _lr_lambda(step):
        if cfg.warmup_steps > 0 and step < cfg.warmup_steps:
            return max(1e-3, (step + 1) / cfg.warmup_steps)
        if cfg.lr_schedule == "cosine":
            progress = (step - cfg.warmup_steps) / max(1, total_steps - cfg.warmup_steps)
            progress = min(max(progress, 0.0), 1.0)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda) \
        if (cfg.warmup_steps > 0 or cfg.lr_schedule != "none") else None

    def run_epoch(sl, train_: bool):
        model.train(train_)
        losses, preds, ys, ms = [], [], [], []
        for t in range(sl.start + W, sl.stop):
            m = mask_all[t]
            if m.sum() < 3:
                continue
            # Gather full (N, W, F) window — pass the entire N=84 to StockMixer
            x_win = x[t - W + 1: t + 1].permute(1, 0, 2)  # [N, W, F]
            y_hat = model(x_win, m)                       # [N]

            l = cs_mse_loss(y_hat, y[t], m)
            if train_:
                opt.zero_grad(); l.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                if scheduler is not None:
                    scheduler.step()
            losses.append(l.item())
            preds.append(y_hat.detach().cpu().numpy()[None, :])
            ys.append(y[t].detach().cpu().numpy()[None, :])
            ms.append(m.detach().cpu().numpy()[None, :])

        yhat_arr = np.concatenate(preds) if preds else np.zeros((0, N))
        y_arr   = np.concatenate(ys)    if ys    else np.zeros((0, N))
        m_arr   = np.concatenate(ms)    if ms    else np.zeros((0, N), dtype=bool)
        return (float(np.mean(losses)) if losses else float("nan"),
                yhat_arr, y_arr, m_arr)

    history_log: list[dict] = []
    best_val_ic = -float("inf"); best_state = None; best_epoch = -1; stale = 0

    for epoch in range(cfg.epochs):
        t0 = time.time()
        train_loss, _, _, _ = run_epoch(tr, True)
        val_loss, vp, vy, vm = run_epoch(va, False)
        v_ic = information_coefficient(vp, vy, vm)
        v_r  = rank_ic(vp, vy, vm)
        dt = time.time() - t0
        improved = v_ic > best_val_ic
        print(f"[{epoch:02d}] train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"val_ic={v_ic:+.4f}  val_rank_ic={v_r:+.4f}  ({dt:.1f}s)"
              f"{'  *best*' if improved else ''}")
        history_log.append(dict(epoch=epoch, train_loss=train_loss, val_loss=val_loss,
                                val_ic=v_ic, val_rank_ic=v_r, time_sec=round(dt, 2)))
        if improved:
            best_val_ic = v_ic
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch; stale = 0
        else:
            stale += 1
            if stale >= cfg.patience:
                print(f"early stop at epoch {epoch}  best={best_val_ic:+.4f} @ {best_epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    _, tp, ty, tm = run_epoch(te, False)
    test_ic = information_coefficient(tp, ty, tm)
    test_r  = rank_ic(tp, ty, tm)
    print(f"\nTEST ic={test_ic:+.4f}  rank_ic={test_r:+.4f}")

    return dict(
        panel_T=T, panel_N=N, panel_F=Fdim,
        test_ic=test_ic, test_rank_ic=test_r,
        best_val_ic=best_val_ic, best_epoch=best_epoch,
        history=history_log, config=asdict(cfg),
        fold=cfg.fold, fold_label=fold_label,
        train_range=[str(dates[tr.start].date()), str(dates[tr.stop - 1].date())],
        val_range  =[str(dates[va.start].date()),  str(dates[va.stop - 1].date())],
        test_range =[str(dates[te.start].date()),  str(dates[te.stop - 1].date())],
        _test_preds=tp.astype(np.float32),
        _test_y=ty.astype(np.float32),
        _test_mask=tm.astype(bool),
        _test_dates=np.array([str(pd.Timestamp(dates[i]).date())
                               for i in range(te.start, te.stop)], dtype="U10"),
        _tickers=np.array(tickers, dtype=object),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fold", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-tickers", type=int, default=100)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--temporal-window", type=int, default=None)
    p.add_argument("--output", type=Path, default=Path("results/baselines_sota/stockmixer/run.json"))
    args = p.parse_args()

    cfg = StockMixerTrainConfig(fold=args.fold, seed=args.seed, max_tickers=args.max_tickers)
    if args.epochs is not None: cfg.epochs = args.epochs
    if args.lr is not None: cfg.lr = args.lr
    if args.temporal_window is not None: cfg.temporal_window = args.temporal_window

    result = train(cfg)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tp = result.pop("_test_preds"); ty = result.pop("_test_y"); tm = result.pop("_test_mask")
    td = result.pop("_test_dates"); tk = result.pop("_tickers")
    args.output.write_text(json.dumps(result, indent=2, default=str))
    np.savez_compressed(args.output.with_suffix(".npz"),
                        preds=tp, y=ty, mask=tm, test_dates=td, tickers=tk)
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
