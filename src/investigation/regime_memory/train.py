"""Training loop for REM-STAR iteration 1.

Fold 1 and 2 only (fold 3 is reserved). 5-seed evaluation mandatory.
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

from src.investigation.regime_memory.catalog import (
    RegimeCatalog, assign_days_to_cluster, build_catalog,
)
from src.investigation.regime_memory.corr_attention import (
    build_attn_bias, precompute_rolling_corr,
)
from src.investigation.regime_memory.model import REMConfig, REMStar
from src.investigation.regime_memory.set_model import SetSTAR, SetSTARConfig
from src.investigation.regime_memory.signature import (
    compute_signatures, forward_fill_signatures,
)
from src.mtgn.data.risk_features import build_risk_features
from src.mtgn.model.utils.patch_construction import build_patches
from src.mtgn.training.losses import (
    cs_group_relative_robust_loss, cs_mse_loss, cs_robust_loss,
)
from src.mtgn.training.panel_enriched import (
    EnrichedPanelConfig, build_enriched_panel, panel_to_tensors,
)
from src.mtgn.training.train import information_coefficient, rank_ic
from src.mtgn.training.train_baselines import walk_forward_fold


LOG_RET_FEATURE_INDEX = 0


@dataclass
class REMTrainConfig:
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
    # REM specifics
    K: int = 6
    use_regime_token: bool = True
    dispersion_window: int = 20
    corr_window: int = 60
    kmeans_seed: int = 0
    # iter 3A: memory-token + single-graph switches
    num_memory_tokens: int = 0      # 0 = off; M > 0 = M learnable tokens per cluster/prototype
    single_graph: bool = False       # use pure STAR's single global graph
    # iter 3B/C: learnable prototypes
    num_prototypes: int = 0          # 0 = off (k-means). L > 0 = learnable prototypes
    proto_temperature: float = 1.0
    sparsity_weight: float = 0.0     # L1 on average soft-assignment, encourages sparse effective K
    # iter 9: RL prototype selection (REINFORCE)
    use_rl_proto: bool = False
    rl_baseline_decay: float = 0.95
    rl_loss_weight: float = 1.0
    # iter 10: robust loss (Huber + optional inverse-vol weighting)
    huber_delta: float = 0.0       # 0 = off (use MSE); > 0 enables Huber
    use_vol_weighting: bool = False  # inverse-volatility per-ticker weighting
    # Proposal A: correlation-aware attention bias (0 = off)
    corr_bias_alpha: float = 0.0
    # Proposal A (gated): make alpha depend on the regime signature's
    # mean-pairwise-correlation dimension so the bias is active only
    # during high-correlation (drawdown) regimes. gate_tau<=0 disables
    # the gate (constant-alpha behavior preserved).
    corr_gate_threshold: float = 0.0
    corr_gate_tau: float = 0.0
    # Proposal A learnable gate (alternative to hard-threshold gate)
    corr_bias_gate_hidden: int = 0
    # Proposal C: DANN-style regime-adversarial loss
    dann_lambda_max: float = 0.0
    dann_hidden: int = 64
    dann_loss_weight: float = 1.0
    # Proposal D2: per-ticker vol-normalized target. Rolls 60-day std of
    # daily log returns per ticker, divides the 5-day-forward target by
    # that to put every ticker's target on a comparable scale regardless
    # of its idiosyncratic volatility. Applied BEFORE cross-sectional
    # z-scoring. Critical for mixed-universe (some mature, some IPO)
    # settings where per-ticker volatilities differ sharply.
    target_vol_norm: bool = False
    target_vol_window: int = 60
    # Proposal D1: Set Transformer encoder (universe-agnostic, drops the
    # fixed graph of neighbors in favor of set-attention across all
    # active tickers). Intended for open-universe settings with IPOs.
    use_set_model: bool = False
    set_temporal_encoder: str = "gru"
    # Proposal B: group-relative target z-scoring using a train-window
    # correlation neighborhood (fixed per ticker) rather than the universe-
    # wide active set. Decontaminates the target from the fold-2 PC1-
    # dominant common factor. loss_group_size > 8 recommended for stable
    # group-level mean/std estimates.
    use_group_relative_loss: bool = False
    loss_group_size: int = 24


def train(cfg: REMTrainConfig) -> dict:
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

    # Feature standardization (Audit 3: train-fold only)
    mu = x[tr].reshape(-1, Fdim).mean(dim=0)
    sd = x[tr].reshape(-1, Fdim).std(dim=0).clamp(min=1e-6)
    x = (x - mu) / sd

    # Regime signatures (Audit REM-1: causal)
    log_returns_np = tensors["x"][:, :, LOG_RET_FEATURE_INDEX]  # raw log_return (pre z-score)
    mask_np = tensors["mask"]
    risk_df = build_risk_features(cfg.start_date, cfg.end_date)
    sigs_raw = compute_signatures(log_returns_np, mask_np, dates, risk_df,
                                  dispersion_window=cfg.dispersion_window,
                                  corr_window=cfg.corr_window)
    sigs = forward_fill_signatures(sigs_raw)
    print(f"signatures: shape {sigs.shape}  finite rows {np.all(np.isfinite(sigs), axis=1).sum()}/{T}")

    # Build catalog on train slice only (REM-Audits 2, 3, 4)
    edge_train_end = str(pd.Timestamp(dates[tr.stop - 1]).date())
    catalog: RegimeCatalog = build_catalog(
        signatures=sigs,
        tickers=tickers, dates=dates,
        log_returns=log_returns_np, mask=mask_np,
        train_slice=tr,
        cfg_start_date=cfg.start_date, cfg_train_end=edge_train_end,
        K=cfg.K, num_neighbors=cfg.num_neighbors, kmeans_seed=cfg.kmeans_seed,
        single_graph=cfg.single_graph,
    )

    # Assign every panel day (train + val + test) to its cluster
    day_cluster = assign_days_to_cluster(sigs, catalog)  # [T]
    print(f"cluster assignment: train histogram {np.bincount(day_cluster[tr], minlength=cfg.K)}")
    print(f"cluster assignment: val histogram   {np.bincount(day_cluster[va], minlength=cfg.K)}")
    print(f"cluster assignment: test histogram  {np.bincount(day_cluster[te], minlength=cfg.K)}")

    # Precompute per-cluster top_neighbors tensors on device
    topk_by_c = {c: torch.from_numpy(catalog.top_neighbors[c]).long().to(device)
                 for c in range(cfg.K)}

    # Proposal A: precompute rolling pairwise correlation [T, N, N]
    rolling_corr: torch.Tensor | None = None
    if cfg.corr_bias_alpha > 0:
        lr_t = torch.from_numpy(log_returns_np).float().to(device)
        mask_t = torch.from_numpy(mask_np).bool().to(device)
        rolling_corr = precompute_rolling_corr(lr_t, mask_t, window=cfg.temporal_window)
        print(f"rolling_corr: shape {tuple(rolling_corr.shape)}  "
              f"finite fraction {torch.isfinite(rolling_corr).float().mean().item():.4f}  "
              f"alpha={cfg.corr_bias_alpha}")

    # Proposal D2: per-ticker rolling vol for target normalization.
    # Strictly causal: sigma_i(t) uses log returns from t-W to t-1.
    if cfg.target_vol_norm:
        W_vol = cfg.target_vol_window
        lr_np = log_returns_np  # [T, N] raw log_return_1d
        mask_np_bool = mask_np.astype(bool)
        sigma = np.zeros_like(lr_np)
        for t in range(W_vol, T):
            win_lr = lr_np[t - W_vol:t]
            win_m = mask_np_bool[t - W_vol:t].astype(np.float64)
            count = np.maximum(win_m.sum(axis=0), 1.0)
            mu_t = (win_lr * win_m).sum(axis=0) / count
            centered = (win_lr - mu_t[None, :]) * win_m
            var_t = (centered ** 2).sum(axis=0) / np.maximum(count - 1.0, 1.0)
            sigma[t] = np.sqrt(np.clip(var_t, 1e-10, None))
        # 5-day forward return scale ≈ sqrt(5) × daily std. Use this to put
        # the target on unit-variance scale per ticker.
        sigma_5d = sigma * np.sqrt(cfg.horizon_days)
        # Replace any pre-warmup entries with median sigma to avoid NaNs.
        sigma_5d_median = float(np.median(sigma_5d[sigma_5d > 0])) if (sigma_5d > 0).any() else 1.0
        sigma_5d = np.where(sigma_5d > 0, sigma_5d, sigma_5d_median)
        target_sigma_t = torch.from_numpy(sigma_5d.astype(np.float32)).to(device)
        print(f"target_vol_norm: sigma_5d mean={sigma_5d.mean():.4f} median={sigma_5d_median:.4f}")
    else:
        target_sigma_t = None

    # Proposal B: larger top-K correlation peer graph from train window
    # (fixed per ticker, used only for group-relative loss z-scoring).
    loss_group_idx: torch.Tensor | None = None
    if cfg.use_group_relative_loss:
        k_loss = cfg.loss_group_size
        N_tot = log_returns_np.shape[1]
        tr_lr = log_returns_np[tr]
        tr_mask = mask_np[tr]
        # Pairwise-overlap correlation so tickers with trading gaps still
        # get peers. Requires min_overlap common active days to be valid.
        min_overlap = 60
        peer = np.full((N_tot, k_loss), -1, dtype=np.int64)
        lr_c = np.where(tr_mask, tr_lr, 0.0).astype(np.float64)
        m_f = tr_mask.astype(np.float64)
        count = m_f.T @ m_f                          # [N, N] pairwise overlap
        sum_x = m_f.T @ lr_c                         # [N, N] sum of j over i-active days
        sum_x2 = m_f.T @ (lr_c ** 2)                 # [N, N]
        # numerators: sum(x_i * x_j * m_i * m_j)
        cross = (lr_c * m_f).T @ (lr_c * m_f)        # [N, N]
        # means restricted to pairwise-overlap
        denom = np.clip(count, 1.0, None)
        mu_i = sum_x / denom                         # mu_i[i, j] = mean of j over i-active days (but we need joint)
        # Use joint means (both active)
        # Using product of aligned series: cross/denom - mu_both_i * mu_both_j
        # Compute mu_both_i[i,j] = sum(x_i m_i m_j) / count[i,j], analogously mu_both_j
        sum_xi = (lr_c * m_f).T @ m_f                 # sum_{i rows t} x_i * m_j  -> [N, N]
        sum_xj = sum_xi.T
        mu_both_i = sum_xi / denom
        mu_both_j = sum_xj / denom
        e_xy = cross / denom
        # variance on the pairwise-overlap
        sum_xi_sq = (lr_c ** 2 * m_f).T @ m_f
        sum_xj_sq = sum_xi_sq.T
        var_i = sum_xi_sq / denom - mu_both_i ** 2
        var_j = sum_xj_sq / denom - mu_both_j ** 2
        cov = e_xy - mu_both_i * mu_both_j
        sd = np.sqrt(np.clip(var_i, 1e-12, None) * np.clip(var_j, 1e-12, None))
        C = np.where(count >= min_overlap, cov / np.clip(sd, 1e-12, None), 0.0)
        np.fill_diagonal(C, 0.0)
        for i in range(N_tot):
            row = np.abs(C[i])
            topk_i = np.argpartition(-row, min(k_loss, row.size - 1))[:k_loss]
            topk_i = topk_i[np.argsort(-row[topk_i])]
            # Keep only peers with non-trivial correlation
            topk_i = topk_i[row[topk_i] > 0.01]
            peer[i, :len(topk_i)] = topk_i
        loss_group_idx = torch.from_numpy(peer).long().to(device)
        filled = (loss_group_idx >= 0).sum(dim=1)
        print(f"loss_group_idx: shape {tuple(loss_group_idx.shape)}  "
              f"coverage (>= 5 peers) {(filled >= 5).sum().item()}/{N_tot}  "
              f"mean peers {filled.float().mean().item():.1f}")

    # Z-scored signatures for model input
    sigs_z = (sigs - catalog.mu) / np.maximum(catalog.sd, 1e-6)
    sigs_z = np.where(np.isfinite(sigs_z), sigs_z, 0.0)
    sigs_z_t = torch.from_numpy(sigs_z.astype(np.float32)).to(device)

    if cfg.use_set_model:
        scfg = SetSTARConfig(
            feature_dim=Fdim, hidden_dim=cfg.hidden_dim,
            temporal_window=cfg.temporal_window,
            num_heads=cfg.num_heads, num_layers=cfg.num_layers, ff_dim=cfg.ff_dim,
            transformer_dropout=cfg.transformer_dropout,
            signature_dim=sigs.shape[1],
            head_hidden=cfg.head_hidden, head_dropout=cfg.head_dropout,
            use_regime_token=cfg.use_regime_token,
            num_prototypes=cfg.num_prototypes,
            proto_temperature=cfg.proto_temperature,
            temporal_encoder=cfg.set_temporal_encoder,
        )
        model = SetSTAR(scfg).to(device)
        print(f"model: SetSTAR (universe-agnostic, temporal_encoder={cfg.set_temporal_encoder})")
    else:
        mcfg = REMConfig(
            feature_dim=Fdim, hidden_dim=cfg.hidden_dim,
            num_neighbors=cfg.num_neighbors, temporal_window=cfg.temporal_window,
            num_heads=cfg.num_heads, num_layers=cfg.num_layers, ff_dim=cfg.ff_dim,
            transformer_dropout=cfg.transformer_dropout,
            signature_dim=sigs.shape[1],
            head_hidden=cfg.head_hidden, head_dropout=cfg.head_dropout,
            use_risk_head=False, use_regime_token=cfg.use_regime_token,
            num_memory_tokens=cfg.num_memory_tokens, num_clusters=cfg.K,
            num_prototypes=cfg.num_prototypes,
            proto_temperature=cfg.proto_temperature,
            use_rl_proto=cfg.use_rl_proto,
            corr_bias_alpha=cfg.corr_bias_alpha,
            corr_bias_gate_hidden=cfg.corr_bias_gate_hidden,
            dann_lambda_max=cfg.dann_lambda_max,
            dann_hidden=cfg.dann_hidden,
            dann_classes=cfg.K,
        )
        model = REMStar(mcfg).to(device)
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

    # iter 9: REINFORCE baseline (exponential moving average of recent rewards)
    rl_baseline = 0.0

    def run_epoch(sl, train_: bool):
        nonlocal rl_baseline
        model.train(train_)
        losses, preds, ys, ms = [], [], [], []
        for t in range(sl.start + W, sl.stop):
            m = mask_all[t]
            if m.sum() < 3:
                continue
            active_idx = m.nonzero(as_tuple=True)[0]
            c = int(day_cluster[t])
            top_k_t = topk_by_c[c]
            x_win = x[t - W + 1: t + 1]
            mask_win = mask_all[t - W + 1: t + 1]
            patches, patch_mask = build_patches(x_win, mask_win, top_k_t, active_idx)

            attn_bias_patch = None
            if rolling_corr is not None:
                nbr_idx = top_k_t[active_idx]  # [A, N_nbr]
                # Regime-conditional gate: alpha_t = base_alpha *
                # sigmoid((z_corr_t - threshold) / tau). When tau<=0 the
                # gate is disabled (constant alpha).
                if cfg.corr_gate_tau > 0:
                    z_corr = float(sigs_z_t[t, 2].item())
                    gate = 1.0 / (1.0 + math.exp(-(z_corr - cfg.corr_gate_threshold) / cfg.corr_gate_tau))
                    alpha_t = cfg.corr_bias_alpha * gate
                else:
                    alpha_t = cfg.corr_bias_alpha
                attn_bias_patch = build_attn_bias(
                    rolling_corr[t], active_idx, nbr_idx,
                    W=W, alpha=alpha_t,
                    memory_prefix_len=0, suffix_len=0,
                )  # [A, (N+1)*W, (N+1)*W]
            # DANN lambda schedule: linear ramp from 0 to dann_lambda_max
            # over training. Disabled during eval (dann_lambda=0).
            if train_ and cfg.dann_lambda_max > 0:
                prog = min(1.0, (epoch + 1) / max(1, cfg.epochs))
                dann_lambda = cfg.dann_lambda_max * prog
            else:
                dann_lambda = 0.0
            if cfg.use_set_model:
                # SetSTAR takes per-ticker temporal histories only (no neighbors).
                # patches[:, 0, :, :] is the self-row of each patch.
                x_hist = patches[:, 0, :, :]                  # [A, W, F]
                x_hist_mask = patch_mask[:, 0, :]             # [A, W]
                out = model.forward_day(
                    x_hist=x_hist, x_mask=x_hist_mask,
                    regime_sig=sigs_z_t[t], active_mask=m,
                )
            else:
                out = model.forward_day(
                    patches, patch_mask, sigs_z_t[t], m,
                    cluster_id=int(day_cluster[t]),
                    attn_bias_patch=attn_bias_patch,
                    dann_lambda=dann_lambda,
                )
            y_hat = out["y_hat"]
            # Proposal D2: per-ticker vol-normalize the target BEFORE the
            # cross-sectional z-score inside the loss. Keeps the loss code
            # unchanged; just feeds a normalized target tensor.
            if target_sigma_t is not None:
                y_for_loss = y[t] / target_sigma_t[t].clamp(min=1e-6)
            else:
                y_for_loss = y[t]
            # iter 10: optional robust loss (Huber + inverse-vol weighting)
            if cfg.use_group_relative_loss:
                vol_t = x[t, :, 6].abs() if cfg.use_vol_weighting else None
                delta = cfg.huber_delta if cfg.huber_delta > 0 else float("inf")
                l = cs_group_relative_robust_loss(
                    y_hat, y_for_loss, m, neighbor_idx=loss_group_idx,
                    delta=delta, vol=vol_t,
                )
            elif cfg.huber_delta > 0 or cfg.use_vol_weighting:
                # realized_vol_60d is feature index 6 in panel_enriched FEATURE_COLS;
                # already z-scored by our panel standardization, so convert back to
                # a positive magnitude by taking absolute value.
                vol_t = x[t, :, 6].abs() if cfg.use_vol_weighting else None
                delta = cfg.huber_delta if cfg.huber_delta > 0 else float("inf")
                l = cs_robust_loss(y_hat, y_for_loss, m, delta=delta, vol=vol_t)
            else:
                l = cs_mse_loss(y_hat, y_for_loss, m)
            l_pred_val = float(l.detach().cpu())
            # Proposal C: adversarial regime-classifier loss (GRL flips grads)
            if train_ and dann_lambda > 0 and out.get("regime_logits") is not None:
                regime_logits = out["regime_logits"]  # [A, K]
                target = torch.full((regime_logits.shape[0],), int(day_cluster[t]),
                                    dtype=torch.long, device=regime_logits.device)
                l_dann = torch.nn.functional.cross_entropy(regime_logits, target)
                l = l + cfg.dann_loss_weight * l_dann
            # L1 on soft-assignment to encourage effective-K < L
            if train_ and cfg.sparsity_weight > 0 and out.get("proto_weights") is not None:
                l = l + cfg.sparsity_weight * out["proto_weights"].abs().sum()
            # iter 9: REINFORCE policy gradient on prototype selection
            if train_ and cfg.use_rl_proto and out.get("rl_log_prob") is not None:
                reward = -l_pred_val
                advantage = reward - rl_baseline
                rl_baseline = cfg.rl_baseline_decay * rl_baseline + (1.0 - cfg.rl_baseline_decay) * reward
                l = l - cfg.rl_loss_weight * advantage * out["rl_log_prob"]
            if train_:
                opt.zero_grad(); l.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); scheduler.step()
            losses.append(l.item())
            preds.append(y_hat.detach().cpu().numpy()[None, :])
            ys.append(y[t].detach().cpu().numpy()[None, :])
            ms.append(m.detach().cpu().numpy()[None, :])
        yhat_arr = np.concatenate(preds) if preds else np.zeros((0, N))
        y_arr   = np.concatenate(ys)    if ys    else np.zeros((0, N))
        m_arr   = np.concatenate(ms)    if ms    else np.zeros((0, N), dtype=bool)
        return (float(np.mean(losses)) if losses else float("nan"),
                yhat_arr, y_arr, m_arr)

    history_log = []
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
        K=cfg.K, cluster_sizes_train=dict(catalog.n_per_cluster),
        history=history_log, config=asdict(cfg),
        fold=cfg.fold,
        train_range=[str(dates[tr.start].date()), str(dates[tr.stop - 1].date())],
        val_range  =[str(dates[va.start].date()),  str(dates[va.stop - 1].date())],
        test_range =[str(dates[te.start].date()),  str(dates[te.stop - 1].date())],
        _test_preds=tp.astype(np.float32),
        _test_y=ty.astype(np.float32),
        _test_mask=tm.astype(bool),
        _test_dates=np.array([str(pd.Timestamp(dates[i]).date())
                               for i in range(te.start, te.stop)], dtype="U10"),
        _tickers=np.array(tickers, dtype=object),
        _test_cluster=day_cluster[te.start:te.stop].astype(np.int64),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fold", type=int, required=True, choices=[1, 2, 3],
                   help="fold 3 only with user authorization per Section 7.3")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--K", type=int, default=6)
    p.add_argument("--no-regime-token", action="store_true")
    p.add_argument("--memory-tokens", type=int, default=0,
                   help="M learnable memory tokens per cluster/prototype (iter 3A/C); 0 disables")
    p.add_argument("--single-graph", action="store_true",
                   help="Use pure STAR's single global mechanistic graph")
    p.add_argument("--num-prototypes", type=int, default=0,
                   help="L learnable prototypes (iter 3B/C); 0 = k-means hard clustering")
    p.add_argument("--sparsity-weight", type=float, default=0.0,
                   help="L1 coefficient on soft assignments to encourage effective-K < L")
    p.add_argument("--rl-proto", action="store_true",
                   help="iter 9: REINFORCE-trained prototype selection (hard sample, policy gradient)")
    p.add_argument("--huber-delta", type=float, default=0.0,
                   help="iter 10: Huber loss delta (0 = MSE off; 1.0 is typical Huber)")
    p.add_argument("--vol-weight", action="store_true",
                   help="iter 10: inverse-volatility per-ticker loss weighting")
    p.add_argument("--corr-bias-alpha", type=float, default=0.0,
                   help="Proposal A: correlation-aware attention bias strength (0 = off)")
    p.add_argument("--max-tickers", type=int, default=None,
                   help="Cap on universe size (None = use all available tickers)")
    p.add_argument("--target-vol-norm", action="store_true",
                   help="Proposal D2: per-ticker vol-normalize the target before CS z-score")
    p.add_argument("--target-vol-window", type=int, default=60,
                   help="Proposal D2: rolling-vol window size (default 60 days)")
    p.add_argument("--use-set-model", action="store_true",
                   help="Proposal D1: use SetSTAR (universe-agnostic set transformer)")
    p.add_argument("--set-temporal-encoder", type=str, default="gru",
                   choices=["gru", "transformer"],
                   help="Proposal D1: per-ticker temporal encoder")
    p.add_argument("--hidden-dim", type=int, default=None,
                   help="Override hidden dim (default 128)")
    p.add_argument("--epochs", type=int, default=None,
                   help="Override max epochs (default 15)")
    p.add_argument("--patience", type=int, default=None,
                   help="Override early-stopping patience (default 5)")
    p.add_argument("--corr-gate-threshold", type=float, default=0.0,
                   help="Proposal A (gated): z-corr threshold for sigmoid gate on alpha")
    p.add_argument("--corr-gate-tau", type=float, default=0.0,
                   help="Proposal A (gated): sigmoid temperature (>0 enables gate)")
    p.add_argument("--group-relative-loss", action="store_true",
                   help="Proposal B: group-relative (correlation-neighborhood) target z-scoring")
    p.add_argument("--loss-group-size", type=int, default=24,
                   help="Proposal B: top-K correlation peers per ticker for loss z-score group")
    p.add_argument("--corr-bias-gate-hidden", type=int, default=0,
                   help="Proposal A (learnable gate): MLP hidden size (0 disables gate)")
    p.add_argument("--dann-lambda-max", type=float, default=0.0,
                   help="Proposal C: max GRL lambda (0 disables DANN)")
    p.add_argument("--dann-hidden", type=int, default=64,
                   help="Proposal C: regime-discriminator MLP hidden size")
    p.add_argument("--dann-loss-weight", type=float, default=1.0,
                   help="Proposal C: adversarial loss weight")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    # Fold 3 authorized per Section 7.3 (REM iter 3B fold-3 eval, 2026-04-16 13:02 UTC)

    cfg = REMTrainConfig(fold=args.fold, seed=args.seed, K=args.K)
    if args.no_regime_token:
        cfg.use_regime_token = False
    cfg.num_memory_tokens = args.memory_tokens
    cfg.single_graph = args.single_graph
    cfg.num_prototypes = args.num_prototypes
    cfg.sparsity_weight = args.sparsity_weight
    if args.rl_proto:
        cfg.use_rl_proto = True
    if args.huber_delta > 0:
        cfg.huber_delta = args.huber_delta
    if args.vol_weight:
        cfg.use_vol_weighting = True
    if args.corr_bias_alpha > 0:
        cfg.corr_bias_alpha = args.corr_bias_alpha
    if args.max_tickers is not None:
        cfg.max_tickers = args.max_tickers
    if args.target_vol_norm:
        cfg.target_vol_norm = True
        cfg.target_vol_window = args.target_vol_window
    if args.use_set_model:
        cfg.use_set_model = True
        cfg.set_temporal_encoder = args.set_temporal_encoder
    if args.hidden_dim is not None:
        cfg.hidden_dim = args.hidden_dim
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.patience is not None:
        cfg.patience = args.patience
    if args.corr_gate_tau > 0:
        cfg.corr_gate_tau = args.corr_gate_tau
        cfg.corr_gate_threshold = args.corr_gate_threshold
    if args.group_relative_loss:
        cfg.use_group_relative_loss = True
        cfg.loss_group_size = args.loss_group_size
    if args.corr_bias_gate_hidden > 0:
        cfg.corr_bias_gate_hidden = args.corr_bias_gate_hidden
    if args.dann_lambda_max > 0:
        cfg.dann_lambda_max = args.dann_lambda_max
        cfg.dann_hidden = args.dann_hidden
        cfg.dann_loss_weight = args.dann_loss_weight

    result = train(cfg)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tp = result.pop("_test_preds"); ty = result.pop("_test_y"); tm = result.pop("_test_mask")
    td = result.pop("_test_dates"); tk = result.pop("_tickers")
    tc = result.pop("_test_cluster")
    args.output.write_text(json.dumps(result, indent=2, default=str))
    np.savez_compressed(args.output.with_suffix(".npz"),
                        preds=tp, y=ty, mask=tm, test_dates=td, tickers=tk,
                        test_cluster=tc)
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
