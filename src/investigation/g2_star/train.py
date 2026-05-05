"""G²-STAR training loop. Folds 1 and 2 only (fold 3 reserved)."""
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

from src.investigation.g2_star.model import G2Config, GraphGatedSTAR
from src.investigation.regime_memory.signature import (
    compute_extended_signatures, compute_signatures, forward_fill_signatures,
)
from src.mtgn.data.risk_features import build_risk_features
from src.mtgn.graph.edges import EdgeBuildConfig, build_mechanistic_edges
from src.mtgn.model.utils.patch_construction import build_patches, precompute_top_neighbors
from src.mtgn.training.losses import cs_mse_loss
from src.mtgn.training.panel_enriched import (
    EnrichedPanelConfig, build_enriched_panel, panel_to_tensors,
)
from src.mtgn.training.train import information_coefficient, rank_ic
from src.mtgn.training.train_baselines import walk_forward_fold


LOG_RET_FEATURE_INDEX = 0


@dataclass
class G2TrainConfig:
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
    hidden_dim: int = 128
    num_neighbors: int = 8
    temporal_window: int = 20
    num_heads: int = 4
    num_layers: int = 2
    ff_dim: int = 256
    transformer_dropout: float = 0.1
    head_hidden: int = 64
    head_dropout: float = 0.2
    gate_hidden: int = 16
    gate_entropy_weight: float = 0.001
    use_xstock_mlp: bool = False
    xstock_hidden: int = 20
    gate_distill_weight: float = 0.0
    extended_signature: bool = False  # 7-dim (4 base + PC1 share + cs_skew + cs_kurt)
    gate_temperature: float = 1.0      # Fix A: τ < 1 widens α output
    per_ticker_gate: bool = False      # Iter 7: α per ticker instead of per day
    num_event_tokens: int = 0          # Iter 8: K event-cluster bank size; 0 = off
    event_temperature: float = 1.0     # softmax temperature for event-similarity weights


def train(cfg: G2TrainConfig) -> dict:
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

    tr, va, te = walk_forward_fold(dates, cfg.fold, cfg.horizon_days)
    print(f"split [fold{cfg.fold}]: train {tr.stop - tr.start}d  val {va.stop - va.start}d  test {te.stop - te.start}d")

    mu = x[tr].reshape(-1, Fdim).mean(dim=0)
    sd = x[tr].reshape(-1, Fdim).std(dim=0).clamp(min=1e-6)
    x = (x - mu) / sd

    # Mechanistic graph (single, static)
    edge_train_end = str(pd.Timestamp(dates[tr.stop - 1]).date())
    ei_np, ew_np = build_mechanistic_edges(
        tickers,
        EdgeBuildConfig(train_start=cfg.start_date, train_end=edge_train_end),
        require_nonempty=False,
    )
    if ei_np.shape[1] == 0:
        raise RuntimeError("edge builder returned 0 edges")
    top_neighbors = precompute_top_neighbors(ei_np, ew_np, N, cfg.num_neighbors)
    top_neighbors_t = torch.from_numpy(top_neighbors).long().to(device)
    print(f"edges: {ei_np.shape[1]}  top-{cfg.num_neighbors} neighbors per node")

    # Regime signatures
    log_returns_np = tensors["x"][:, :, LOG_RET_FEATURE_INDEX]
    mask_np = tensors["mask"]
    risk_df = build_risk_features(cfg.start_date, cfg.end_date)
    if cfg.extended_signature:
        print("computing extended 7-dim signatures (PC1 + skew + kurt)...")
        sigs_raw = compute_extended_signatures(log_returns_np, mask_np, dates, risk_df)
    else:
        sigs_raw = compute_signatures(log_returns_np, mask_np, dates, risk_df)
    sigs = forward_fill_signatures(sigs_raw)
    sig_train = sigs[tr]
    finite = np.all(np.isfinite(sig_train), axis=1)
    sig_train_f = sig_train[finite]
    sig_mu = sig_train_f.mean(axis=0)
    sig_sd = sig_train_f.std(axis=0).clip(min=1e-6)
    sigs_z = (sigs - sig_mu) / sig_sd
    sigs_z = np.where(np.isfinite(sigs_z), sigs_z, 0.0)
    sigs_z_t = torch.from_numpy(sigs_z.astype(np.float32)).to(device)

    # Iter 8: event-memory bank (precomputed per-day similarity weights)
    event_weights_t = None
    if cfg.num_event_tokens > 0:
        from src.investigation.event_memory_g2.event_memory import (
            build_event_fingerprints, build_event_memory,
            event_similarity_weights, select_event_days,
        )
        fingerprints = build_event_fingerprints(log_returns_np, mask_np, dates, risk_df)
        event_mask = select_event_days(fingerprints, tr, stress_quantile=0.75)
        bank = build_event_memory(fingerprints, event_mask, tr,
                                   K=cfg.num_event_tokens, kmeans_seed=0)
        # Precompute per-day similarity weights [T, K]
        all_weights = np.stack([
            event_similarity_weights(fingerprints[t], bank, temperature=cfg.event_temperature)
            for t in range(T)
        ], axis=0).astype(np.float32)
        event_weights_t = torch.from_numpy(all_weights).to(device)
        print(f"[event-mem] precomputed weights: shape {event_weights_t.shape}, "
              f"mean entropy {-(all_weights * np.log(all_weights + 1e-12)).sum(axis=1).mean():.3f}")

    mcfg = G2Config(
        feature_dim=Fdim, hidden_dim=cfg.hidden_dim,
        num_neighbors=cfg.num_neighbors, temporal_window=cfg.temporal_window,
        num_heads=cfg.num_heads, num_layers=cfg.num_layers, ff_dim=cfg.ff_dim,
        transformer_dropout=cfg.transformer_dropout,
        head_hidden=cfg.head_hidden, head_dropout=cfg.head_dropout,
        signature_dim=sigs.shape[1],
        gate_hidden=cfg.gate_hidden,
        gate_entropy_weight=cfg.gate_entropy_weight,
        use_xstock_mlp=cfg.use_xstock_mlp,
        xstock_hidden=cfg.xstock_hidden,
        num_stocks=N,
        gate_distill_weight=cfg.gate_distill_weight,
        gate_temperature=cfg.gate_temperature,
        per_ticker_gate=cfg.per_ticker_gate,
        num_event_tokens=cfg.num_event_tokens,
    )
    model = GraphGatedSTAR(mcfg).to(device)
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
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)

    W = cfg.temporal_window

    def run_epoch(sl, train_: bool):
        model.train(train_)
        losses, preds, ys, ms, alphas = [], [], [], [], []
        preds_g, preds_n = [], []
        for t in range(sl.start + W, sl.stop):
            m = mask_all[t]
            if m.sum() < 3:
                continue
            active_idx = m.nonzero(as_tuple=True)[0]
            x_win = x[t - W + 1: t + 1]
            mask_win = mask_all[t - W + 1: t + 1]
            patches, patch_mask = build_patches(x_win, mask_win, top_neighbors_t, active_idx)
            # No-graph path input: each active ticker's own (W, F) window
            self_window = x_win[:, active_idx, :].transpose(0, 1)  # [A, W, F]
            ev_w = event_weights_t[t] if event_weights_t is not None else None
            out = model.forward_day(patches, patch_mask, self_window, sigs_z_t[t], m,
                                    event_weights=ev_w)
            y_hat = out["y_hat"]
            l = cs_mse_loss(y_hat, y[t], m)
            # Iter 2: gate distillation. Per-day pathwise losses give a
            # direct gradient signal for α toward whichever path is better.
            # Detach the per-path losses so they only update the gate, not
            # the path encoders themselves.
            if train_ and cfg.gate_distill_weight > 0:
                l_g = cs_mse_loss(out["y_graph"], y[t], m).detach()
                l_n = cs_mse_loss(out["y_nograph"], y[t], m).detach()
                a = out["alpha"]
                l = l + cfg.gate_distill_weight * (a * l_g + (1.0 - a) * l_n)
            if train_ and cfg.gate_entropy_weight > 0:
                # Maximize gate entropy → subtract -entropy = subtract a positive bonus
                ent = model.gate_entropy(out["alpha"])
                l = l - cfg.gate_entropy_weight * ent
            if train_:
                opt.zero_grad(); l.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); scheduler.step()
            losses.append(l.item())
            preds.append(y_hat.detach().cpu().numpy()[None, :])
            preds_g.append(out["y_graph"].detach().cpu().numpy()[None, :])
            preds_n.append(out["y_nograph"].detach().cpu().numpy()[None, :])
            ys.append(y[t].detach().cpu().numpy()[None, :])
            ms.append(m.detach().cpu().numpy()[None, :])
            alphas.append(float(out["alpha"].detach().cpu()))
        yhat_arr = np.concatenate(preds) if preds else np.zeros((0, N))
        y_g_arr = np.concatenate(preds_g) if preds_g else np.zeros((0, N))
        y_n_arr = np.concatenate(preds_n) if preds_n else np.zeros((0, N))
        y_arr   = np.concatenate(ys)    if ys    else np.zeros((0, N))
        m_arr   = np.concatenate(ms)    if ms    else np.zeros((0, N), dtype=bool)
        return (float(np.mean(losses)) if losses else float("nan"),
                yhat_arr, y_arr, m_arr, alphas, y_g_arr, y_n_arr)

    history_log = []
    best_val_ic = -float("inf"); best_state = None; best_epoch = -1; stale = 0
    for epoch in range(cfg.epochs):
        t0 = time.time()
        train_loss, _, _, _, train_alphas, _, _ = run_epoch(tr, True)
        val_loss, vp, vy, vm, val_alphas, _, _ = run_epoch(va, False)
        v_ic = information_coefficient(vp, vy, vm)
        v_r  = rank_ic(vp, vy, vm)
        dt = time.time() - t0
        improved = v_ic > best_val_ic
        train_a_mean = float(np.mean(train_alphas)) if train_alphas else float("nan")
        val_a_mean = float(np.mean(val_alphas)) if val_alphas else float("nan")
        print(f"[{epoch:02d}] train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"val_ic={v_ic:+.4f}  val_rank_ic={v_r:+.4f}  α_train={train_a_mean:.3f}  α_val={val_a_mean:.3f}  ({dt:.1f}s)"
              f"{'  *best*' if improved else ''}")
        history_log.append(dict(epoch=epoch, train_loss=train_loss, val_loss=val_loss,
                                val_ic=v_ic, val_rank_ic=v_r,
                                alpha_train=train_a_mean, alpha_val=val_a_mean,
                                time_sec=round(dt, 2)))
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
    _, tp, ty, tm, test_alphas, tp_g, tp_n = run_epoch(te, False)
    test_ic = information_coefficient(tp, ty, tm)
    test_r  = rank_ic(tp, ty, tm)
    test_alpha_mean = float(np.mean(test_alphas)) if test_alphas else float("nan")
    test_alpha_std = float(np.std(test_alphas)) if test_alphas else float("nan")
    print(f"\nTEST ic={test_ic:+.4f}  rank_ic={test_r:+.4f}  α_test mean={test_alpha_mean:.3f} std={test_alpha_std:.3f}")

    return dict(
        panel_T=T, panel_N=N, panel_F=Fdim,
        test_ic=test_ic, test_rank_ic=test_r,
        best_val_ic=best_val_ic, best_epoch=best_epoch,
        test_alpha_mean=test_alpha_mean, test_alpha_std=test_alpha_std,
        history=history_log, config=asdict(cfg), fold=cfg.fold,
        train_range=[str(dates[tr.start].date()), str(dates[tr.stop - 1].date())],
        val_range  =[str(dates[va.start].date()),  str(dates[va.stop - 1].date())],
        test_range =[str(dates[te.start].date()),  str(dates[te.stop - 1].date())],
        _test_preds=tp.astype(np.float32),
        _test_preds_graph=tp_g.astype(np.float32),
        _test_preds_nograph=tp_n.astype(np.float32),
        _test_y=ty.astype(np.float32),
        _test_mask=tm.astype(bool),
        _test_alphas=np.array(test_alphas, dtype=np.float32),
        _test_dates=np.array([str(pd.Timestamp(dates[i]).date())
                               for i in range(te.start, te.stop)], dtype="U10"),
        _tickers=np.array(tickers, dtype=object),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fold", type=int, required=True, choices=[1, 2, 3],
                   help="fold 1 or 2 for iteration; fold 3 ONLY with user authorization per Section 7.3")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--gate-entropy", type=float, default=None,
                   help="gate entropy bonus weight (default 0.001; iter 2 sets to 0)")
    p.add_argument("--gate-distill", type=float, default=None,
                   help="gate distillation aux loss weight (iter 2: 0.1)")
    p.add_argument("--xstock-mlp", action="store_true",
                   help="add StockMixer-style cross-stock MLP to no-graph path (iter 2)")
    p.add_argument("--xstock-hidden", type=int, default=None,
                   help="cross-stock MLP hidden dim (iter 3 default 8 for less overfit)")
    p.add_argument("--extended-sig", action="store_true",
                   help="use 7-dim enriched signature (iter 4: adds PC1 share, skew, kurt)")
    p.add_argument("--gate-temp", type=float, default=None,
                   help="sigmoid temperature τ (Fix A, iter 5: τ=0.2 widens α range)")
    p.add_argument("--per-ticker-gate", action="store_true",
                   help="iter 7: per-ticker gate α(t, i) instead of per-day α(t)")
    p.add_argument("--event-tokens", type=int, default=0,
                   help="iter 8: K event-cluster bank size (8 recommended); 0 = off")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    # Fold 3 authorized by user per Section 7.3 (2026-04-16 03:02 UTC)

    cfg = G2TrainConfig(fold=args.fold, seed=args.seed)
    if args.gate_entropy is not None:
        cfg.gate_entropy_weight = args.gate_entropy
    if args.gate_distill is not None:
        cfg.gate_distill_weight = args.gate_distill
    if args.xstock_mlp:
        cfg.use_xstock_mlp = True
    if args.xstock_hidden is not None:
        cfg.xstock_hidden = args.xstock_hidden
    if args.extended_sig:
        cfg.extended_signature = True
    if args.gate_temp is not None:
        cfg.gate_temperature = args.gate_temp
    if args.per_ticker_gate:
        cfg.per_ticker_gate = True
    if args.event_tokens > 0:
        cfg.num_event_tokens = args.event_tokens

    result = train(cfg)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tp = result.pop("_test_preds"); ty = result.pop("_test_y"); tm = result.pop("_test_mask")
    td = result.pop("_test_dates"); tk = result.pop("_tickers")
    ta = result.pop("_test_alphas")
    tp_g = result.pop("_test_preds_graph"); tp_n = result.pop("_test_preds_nograph")
    args.output.write_text(json.dumps(result, indent=2, default=str))
    np.savez_compressed(args.output.with_suffix(".npz"),
                        preds=tp, preds_graph=tp_g, preds_nograph=tp_n,
                        y=ty, mask=tm, test_dates=td, tickers=tk,
                        test_alphas=ta)
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
