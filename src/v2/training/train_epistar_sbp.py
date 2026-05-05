"""Training loop for epiSTAR-SBP v1.

Survivorship-Bias Patch on the epiSTAR-full base. Mirrors train_epistar.py
(dynamic-graph branch) with four additions, all applied at the
loss/key/retrieval level rather than the data-pipeline level:

    1. The 14-dim regime key is extended with a 4-dim cohort sub-key
       (fraction of active universe in age buckets 0-21d, 22-126d,
       127-252d, plus mean log(1 + age) capped at log(2520)). The bank
       keys are now 18-dim.
    2. Dual-pool retrieval (M1=5 raw similar + M2=3 cohort-near via L1
       distance < tau_cohort on the 4-dim cohort sub-key) is performed
       inside the model.
    3. Per-(day, ticker) IRF reweighting of the cross-sectional MSE loss
       (square-root tempered on training-fold cohort frequency) plus a
       V-REx penalty on the variance of per-cohort mean rank losses
       across cohort environments. lambda_vrex ramps linearly from 0 to
       its target over the first warmup_epochs epochs.
    4. Per-ticker confidence gate alpha_i with a Beta(2, 2) prior
       penalty added to the loss.

Output JSON includes per-cohort IC slices for diagnostic, the alpha
distribution, and the mu_t mean for the test fold.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from src.mtgn.model.utils.patch_construction import build_patches
from src.mtgn.training.panel_enriched import (
    EnrichedPanelConfig, build_enriched_panel, panel_to_tensors,
)
from src.v2.data.age_features import AgeFeatureConfig, build_age_feature_tensor
from src.v2.data.cohort_features import (
    COHORT_KEY_COLS, build_cohort_subkey, cohort_bucket_per_cell,
)
from src.v2.data.episode_keys import (
    EpisodeKeyConfig, EPISODE_KEY_COLS, build_episode_keys,
)
from src.v2.graph.dynamic_edges import (
    DynamicGraphConfig, build_dynamic_neighbors,
)
from src.v2.model.episode_memory import EpisodeMemoryConfig
from src.v2.model.epistar_sbp import EpiSTARSBP, EpiSTARSBPConfig
from src.v2.model.star_backbone import STARBackboneConfig
from src.v2.training.folds import fold_indices
from src.v2.training.sbp_losses import (
    alpha_beta_prior, cohort_irf_weights, compute_irf_freq, cs_mse_loss,
    vrex_penalty,
)


@dataclass
class TrainConfig:
    """Top-level hyperparameters."""

    fold: int = 1
    seed: int = 42
    panel_start: str = "2015-01-09"
    panel_end: str = "2022-12-31"
    horizon_days: int = 5
    universe_csv: str = "data/raw/biotech_universe_v1.csv"
    epochs: int = 12
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_steps: int = 500
    grad_clip: float = 1.0
    early_stop_patience: int = 4
    correlation_window: int = 60
    output_dir: str = "results/epistar_sbp"
    # SBP-specific
    lambda_irf: float = 1.0
    lambda_vrex: float = 5.0
    lambda_alpha_prior: float = 0.01
    irf_temper: float = 0.5
    vrex_warmup_epochs: int = 10
    cohort_num_buckets: int = 4


def set_seeds(seed: int) -> None:
    """Set seeds for Python, NumPy, and PyTorch."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def per_day_ic(y_hat, y, mask, rank=False):
    """Daily-then-mean IC. Returns (mean_ic, per_day_ic_array)."""
    t_total = y_hat.shape[0]
    ics = np.full(t_total, np.nan, dtype=np.float64)
    for t in range(t_total):
        m = mask[t]
        if m.sum() < 5:
            continue
        a = y_hat[t, m]; b = y[t, m]
        if rank:
            a = pd.Series(a).rank().to_numpy()
            b = pd.Series(b).rank().to_numpy()
        sa = a.std(); sb = b.std()
        if sa < 1e-9 or sb < 1e-9:
            continue
        ics[t] = float(np.corrcoef(a, b)[0, 1])
    if np.all(np.isnan(ics)):
        return 0.0, ics
    return float(np.nanmean(ics)), ics


def ndcg_at_k(y_hat, y, mask, k):
    """Daily-mean NDCG at k computed on the active cross-section."""
    out = []
    for t in range(y_hat.shape[0]):
        m = mask[t]
        if m.sum() < k + 1:
            continue
        scores = y_hat[t, m]; rels = y[t, m]
        rels_pos = rels - rels.min() + 1e-9
        order = np.argsort(-scores)[:k]
        gains = rels_pos[order]
        discounts = 1.0 / np.log2(np.arange(2, k + 2))
        dcg = float((gains * discounts).sum())
        ideal = np.sort(rels_pos)[::-1][:k]
        idcg = float((ideal * discounts).sum())
        if idcg < 1e-9:
            continue
        out.append(dcg / idcg)
    return float(np.mean(out)) if out else 0.0


def cohort_sliced_ic(y_hat, y, mask, cohort_buckets, num_buckets=4):
    """Per-cohort daily-then-mean IC. Computes IC on cells in the same
    bucket for each day, then averages across days."""
    t_total = y_hat.shape[0]
    out: dict[int, float] = {}
    for k in range(num_buckets):
        ics = []
        for t in range(t_total):
            cell = mask[t] & (cohort_buckets[t] == k)
            if cell.sum() < 5:
                continue
            a = y_hat[t, cell]; b = y[t, cell]
            sa = a.std(); sb = b.std()
            if sa < 1e-9 or sb < 1e-9:
                continue
            ics.append(float(np.corrcoef(a, b)[0, 1]))
        out[k] = float(np.mean(ics)) if ics else 0.0
    return out


def standardize_features(x, mask, train_idx):
    """Per-feature z-score using train-fold statistics; masked cells = 0."""
    flat_train_mask = mask[train_idx]; x_train = x[train_idx]
    out = np.zeros_like(x)
    for f in range(x.shape[2]):
        vals = x_train[..., f][flat_train_mask]
        if vals.size < 2:
            mu, sd = 0.0, 1.0
        else:
            mu = float(np.mean(vals)); sd = float(np.std(vals))
            if sd < 1e-6:
                sd = 1.0
        out[..., f] = (x[..., f] - mu) / sd
    out = out * mask[..., None]
    return out


def warmup_cosine_lr(step, warmup, total):
    """Linear warmup followed by cosine decay to 0.1x."""
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))


def build_episode_values(keys, panel_x, panel_mask):
    """Per-day episode value: 18-dim cohort-augmented key + 9 summary stats."""
    t_total = keys.shape[0]
    feature_idx = [0, 1, 5, 6]
    n_summary = 2 * len(feature_idx) + 1
    out = np.zeros((t_total, n_summary), dtype=np.float32)
    for t in range(t_total):
        m = panel_mask[t]
        if m.sum() < 5:
            continue
        for j, fi in enumerate(feature_idx):
            v = panel_x[t, m, fi]
            out[t, 2 * j] = float(np.mean(v))
            out[t, 2 * j + 1] = float(np.std(v))
        out[t, -1] = float(m.sum()) / 250.0
    return np.concatenate([keys, out], axis=1)


def main(cfg_path: str, fold: int, seed: int, smoke: bool = False) -> None:
    """Train epiSTAR-SBP on one (fold, seed) pair."""
    with open(cfg_path) as f:
        raw = yaml.safe_load(f)

    train_cfg = TrainConfig(**{**raw.get("train", {}), "fold": fold, "seed": seed})
    backbone_cfg = STARBackboneConfig(**raw.get("backbone", {}))
    memory_cfg = EpisodeMemoryConfig(**raw.get("memory", {}))
    sbp_cfg = EpiSTARSBPConfig(
        backbone=backbone_cfg, memory=memory_cfg,
        cohort_dim=raw.get("model", {}).get("cohort_dim", 4),
        dual_pool_m1=raw.get("model", {}).get("dual_pool_m1", 5),
        dual_pool_m2=raw.get("model", {}).get("dual_pool_m2", 3),
        tau_cohort=raw.get("model", {}).get("tau_cohort", 0.4),
        episode_value_dim=raw.get("model", {}).get("episode_value_dim", 32),
        cross_attn_heads=raw.get("model", {}).get("cross_attn_heads", 4),
        gate_hidden_dim=raw.get("model", {}).get("gate_hidden_dim", 64),
        head_hidden_dim=raw.get("model", {}).get("head_hidden_dim", 64),
        head_dropout=raw.get("model", {}).get("head_dropout", 0.1),
        use_per_ticker_gate=raw.get("ablation", {}).get("use_per_ticker_gate", True),
        use_two_head_xattn=raw.get("ablation", {}).get("use_two_head_xattn", True),
        use_dual_pool=raw.get("ablation", {}).get("use_dual_pool", True),
        disable_retrieval=raw.get("ablation", {}).get("disable_retrieval", False),
    )
    if smoke:
        train_cfg.epochs = 2

    set_seeds(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[epiSTAR-SBP] fold={fold} seed={seed} device={device}")

    # Panel.
    panel_cfg = EnrichedPanelConfig(
        start_date=pd.Timestamp(train_cfg.panel_start),
        end_date=pd.Timestamp(train_cfg.panel_end),
        horizon_days=train_cfg.horizon_days,
        universe_csv=Path(train_cfg.universe_csv),
    )
    panel, tickers, dates = build_enriched_panel(panel_cfg)
    tens = panel_to_tensors(panel, tickers, dates)
    x_raw = tens["x"]; y = tens["y"]; mask = tens["mask"]
    print(f"[epiSTAR-SBP] panel: T={x_raw.shape[0]} N={x_raw.shape[1]} F={x_raw.shape[2]}")
    if x_raw.shape[1] < 50:
        raise RuntimeError(f"Panel has only {x_raw.shape[1]} active tickers; aborting.")

    # Folds.
    train_idx, val_idx, test_idx = fold_indices(fold, dates)
    print(f"[epiSTAR-SBP] fold {fold}: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    # Standardize features.
    x = standardize_features(x_raw, mask, train_idx).astype(np.float32)

    # Age features (used for cohort buckets and per-ticker gate input).
    age_feat, _ = build_age_feature_tensor(mask, AgeFeatureConfig())
    age_days = age_feat[..., 0].astype(np.int64)
    log1p_age = age_feat[..., 1].astype(np.float32)
    hist_valid_60d = age_feat[..., 7].astype(np.float32)
    cohort_bkt = cohort_bucket_per_cell(age_days)
    print(f"[epiSTAR-SBP] cohort buckets per (day, ticker) computed: "
          f"shape={cohort_bkt.shape}")

    # Base 14-dim regime key + 4-dim cohort sub-key -> 18-dim cohort key.
    base_keys, _ = build_episode_keys(
        dates=dates, log_returns=x_raw[..., 0], mask=mask, cfg=EpisodeKeyConfig(),
    )
    cohort_subkey = build_cohort_subkey(age_days, mask)
    keys = np.concatenate([base_keys, cohort_subkey], axis=1).astype(np.float32)
    key_cols = list(EPISODE_KEY_COLS) + list(COHORT_KEY_COLS)
    print(f"[epiSTAR-SBP] cohort-augmented episode keys: {keys.shape[1]} dims")

    values = build_episode_values(keys, x_raw, mask)
    print(f"[epiSTAR-SBP] episode values: {values.shape[1]} dims")

    # Dynamic correlation graph (same as epiSTAR-full).
    dyn_cfg = DynamicGraphConfig(
        window_days=train_cfg.correlation_window,
        top_k=backbone_cfg.num_neighbors,
    )
    dyn_neighbors = build_dynamic_neighbors(
        returns=x_raw[..., 0], mask=mask, cfg=dyn_cfg, static_score=None,
    )
    dyn_neighbors_t = torch.from_numpy(dyn_neighbors).to(device)
    print(f"[epiSTAR-SBP] dynamic graph: window={dyn_cfg.window_days}d "
          f"top_K={dyn_cfg.top_k}")

    # Mu_t input fraction-of-universe-in-first-126-days. We reuse the
    # cohort sub-key components: frac_age_0_21 + frac_age_22_126.
    frac_first_126d = (cohort_subkey[:, 0] + cohort_subkey[:, 1]).astype(np.float32)

    # IRF training-fold cohort frequency.
    train_cohort = cohort_bkt[train_idx][mask[train_idx]]
    train_cohort_t = torch.from_numpy(train_cohort.astype(np.int64)).to(device)
    irf_freq = compute_irf_freq(train_cohort_t, num_buckets=train_cfg.cohort_num_buckets)
    print(f"[epiSTAR-SBP] train-fold cohort frequencies: "
          f"{irf_freq.detach().cpu().numpy().tolist()}")

    # Model.
    sbp_cfg.episode_value_dim = values.shape[1]
    model = EpiSTARSBP(sbp_cfg, episode_key_dim=keys.shape[1]).to(device)
    model.memory.populate(
        keys=keys, values=values,
        day_indices=np.arange(len(dates)), train_day_indices=train_idx,
    )
    model.memory.to(device)

    allowed_train = torch.from_numpy(train_idx).long().to(device)
    optim = AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay,
    )
    total_steps = train_cfg.epochs * max(1, len(train_idx))
    scheduler = LambdaLR(
        optim, lr_lambda=lambda s: warmup_cosine_lr(s, train_cfg.warmup_steps, total_steps)
    )
    use_amp = device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    w = backbone_cfg.temporal_window
    best_val_ic = -1e9; best_state = None; patience = 0
    history: list[dict] = []

    def forward_one_day(t_idx: int) -> dict:
        if t_idx < max(w, train_cfg.correlation_window):
            return {}
        active_mask_t = torch.from_numpy(mask[t_idx]).to(device)
        if active_mask_t.sum() < 5:
            return {}
        active_idx = active_mask_t.nonzero(as_tuple=False).squeeze(-1)

        x_window = torch.from_numpy(x[t_idx - w + 1 : t_idx + 1]).to(device)
        mask_window = torch.from_numpy(mask[t_idx - w + 1 : t_idx + 1]).to(device)
        top_neighbors_day = dyn_neighbors_t[t_idx]
        patches, patch_mask = build_patches(
            x_window=x_window, mask_window=mask_window,
            top_neighbors=top_neighbors_day, active_idx=active_idx,
        )
        query_raw_key = torch.from_numpy(keys[t_idx]).float().to(device)
        # Two regime scalars used by the gate: VIX z and avg pairwise corr.
        regime_scalars = model.memory.standardize_query(query_raw_key)[[0, 9]].clone()
        if torch.isnan(regime_scalars).any():
            regime_scalars = torch.zeros(2, device=device)

        # mu_t input: VIX z, avg pairwise corr, fraction in first 126d post-IPO.
        mu_input = torch.tensor(
            [regime_scalars[0].item(), regime_scalars[1].item(), float(frac_first_126d[t_idx])],
            dtype=torch.float32, device=device,
        )

        # Per-active-ticker age features for the per-ticker gate.
        active_idx_np = active_idx.detach().cpu().numpy()
        ticker_age_feats = torch.from_numpy(
            np.stack([log1p_age[t_idx, active_idx_np],
                      hist_valid_60d[t_idx, active_idx_np]], axis=1)
        ).float().to(device)

        with autocast(enabled=use_amp, dtype=torch.float16):
            out = model.forward_day(
                patches=patches, patch_mask=patch_mask, active_mask=active_mask_t,
                query_raw_key=query_raw_key, query_day_idx=int(t_idx),
                allowed_day_indices=allowed_train,
                gate_regime_scalars=regime_scalars, mu_input=mu_input,
                ticker_age_features=ticker_age_feats,
            )
        out["y_hat"] = out["y_hat"].float()
        out["alpha_per_ticker"] = out["alpha_per_ticker"].float()
        out["active_mask"] = active_mask_t
        out["t_idx"] = t_idx
        out["active_idx"] = active_idx
        return out

    def composite_loss(out: dict, lambda_vrex_eff: float) -> tuple[torch.Tensor, dict]:
        """Composite loss with optional IRF + V-REx + alpha-prior."""
        t_idx = out["t_idx"]
        y_true_t = torch.from_numpy(y[t_idx]).to(device)
        mask_t = out["active_mask"]
        active_idx = out["active_idx"]
        a = active_idx.shape[0]

        # Per-(day, ticker) cohort buckets for active cells.
        cohort_buckets_t = torch.from_numpy(
            cohort_bkt[t_idx, active_idx.detach().cpu().numpy()].astype(np.int64)
        ).to(device)

        # L_rank: unweighted cross-sectional MSE.
        l_rank = cs_mse_loss(out["y_hat"], y_true_t, mask_t)

        # L_irf: same loss with per-(day, ticker) IRF reweighting.
        if train_cfg.lambda_irf > 0:
            irf_w_active = cohort_irf_weights(
                cohort_buckets_t, irf_freq, temper=train_cfg.irf_temper
            )
            sample_w_full = torch.zeros_like(out["y_hat"])
            sample_w_full[active_idx] = irf_w_active
            l_irf = cs_mse_loss(
                out["y_hat"], y_true_t, mask_t, sample_weights=sample_w_full,
            )
        else:
            l_irf = torch.zeros((), device=device)

        # V-REx penalty: per-cohort mean rank-loss (using square-error in
        # zscore space) on the same active cross-section.
        env_losses: list[torch.Tensor] = []
        if lambda_vrex_eff > 0:
            yh = out["y_hat"][active_idx]
            yt = y_true_t[active_idx]
            if yt.numel() >= 2:
                mu_y = yt.mean(); sd_y = yt.std().clamp(min=1e-6)
                yt_zs = (yt - mu_y) / sd_y
                sq = (yh - yt_zs) ** 2
                for k in range(train_cfg.cohort_num_buckets):
                    in_k = cohort_buckets_t == k
                    if in_k.sum() >= 2:
                        env_losses.append(sq[in_k].mean())
        l_vrex = vrex_penalty(env_losses) if env_losses else torch.zeros((), device=device)

        # Beta(2, 2) prior on per-ticker alpha.
        l_alpha = alpha_beta_prior(out["alpha_per_ticker"]) if a > 0 else torch.zeros((), device=device)

        loss = (
            l_rank
            + train_cfg.lambda_irf * l_irf
            + lambda_vrex_eff * l_vrex
            + train_cfg.lambda_alpha_prior * l_alpha
        )
        return loss, {"l_rank": float(l_rank.item()),
                      "l_irf": float(l_irf.item()),
                      "l_vrex": float(l_vrex.item()),
                      "l_alpha": float(l_alpha.item()),
                      "n_envs": len(env_losses)}

    @torch.no_grad()
    def evaluate(idx):
        model.eval()
        T = x.shape[0]; N = x.shape[1]
        y_hat_all = np.zeros((T, N), dtype=np.float32)
        eval_mask = np.zeros((T, N), dtype=bool)
        alpha_log: list[float] = []
        mu_log: list[float] = []
        for t_idx in idx:
            out = forward_one_day(int(t_idx))
            if not out:
                continue
            y_hat_all[t_idx] = out["y_hat"].detach().cpu().numpy()
            eval_mask[t_idx] = mask[t_idx]
            ap = out["alpha_per_ticker"].detach().cpu().numpy()
            if ap.size > 0:
                alpha_log.append(float(np.mean(ap)))
            mu_t = out.get("mu_t")
            if mu_t is not None and mu_t.dim() == 0:
                mu_log.append(float(mu_t.item()))
        ic, _ = per_day_ic(y_hat_all, y, eval_mask, rank=False)
        rank_ic, _ = per_day_ic(y_hat_all, y, eval_mask, rank=True)
        ndcg10 = ndcg_at_k(y_hat_all, y, eval_mask, 10)
        ndcg50 = ndcg_at_k(y_hat_all, y, eval_mask, 50)
        cohort_ic = cohort_sliced_ic(
            y_hat_all, y, eval_mask, cohort_bkt,
            num_buckets=train_cfg.cohort_num_buckets,
        )
        return {"ic": ic, "rank_ic": rank_ic, "ndcg10": ndcg10, "ndcg50": ndcg50,
                "cohort_ic": cohort_ic,
                "alpha_mean": float(np.mean(alpha_log)) if alpha_log else 0.0,
                "mu_mean": float(np.mean(mu_log)) if mu_log else 0.0,
                "y_hat_all": y_hat_all, "eval_mask": eval_mask}

    step = 0; smoke_step_cap = 80
    for epoch in range(train_cfg.epochs):
        # Linear ramp: lambda_vrex 0 -> target across vrex_warmup_epochs.
        ramp = min(1.0, epoch / max(1, train_cfg.vrex_warmup_epochs))
        lambda_vrex_eff = train_cfg.lambda_vrex * ramp

        model.train()
        np.random.seed(seed + epoch)
        perm = np.random.permutation(train_idx)
        epoch_losses: list[float] = []
        epoch_rank: list[float] = []
        epoch_irf: list[float] = []
        epoch_vrex: list[float] = []
        epoch_alpha: list[float] = []
        for t_idx in perm:
            t_idx = int(t_idx)
            if t_idx < max(w, train_cfg.correlation_window):
                continue
            out = forward_one_day(t_idx)
            if not out:
                continue
            loss, parts = composite_loss(out, lambda_vrex_eff)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            scaler.step(optim); scaler.update(); scheduler.step()
            epoch_losses.append(float(loss.item()))
            epoch_rank.append(parts["l_rank"])
            epoch_irf.append(parts["l_irf"])
            epoch_vrex.append(parts["l_vrex"])
            epoch_alpha.append(parts["l_alpha"])
            step += 1
            if smoke and step >= smoke_step_cap:
                break

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        val_metrics = evaluate(val_idx)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_l_rank": float(np.mean(epoch_rank)) if epoch_rank else 0.0,
            "train_l_irf": float(np.mean(epoch_irf)) if epoch_irf else 0.0,
            "train_l_vrex": float(np.mean(epoch_vrex)) if epoch_vrex else 0.0,
            "train_l_alpha": float(np.mean(epoch_alpha)) if epoch_alpha else 0.0,
            "lambda_vrex_eff": lambda_vrex_eff,
            "val_ic": val_metrics["ic"],
            "val_rank_ic": val_metrics["rank_ic"],
            "val_alpha_mean": val_metrics["alpha_mean"],
            "val_mu_mean": val_metrics["mu_mean"],
        })
        print(f"[epiSTAR-SBP] epoch {epoch}: loss={train_loss:.4f} "
              f"rank={history[-1]['train_l_rank']:.4f} "
              f"irf={history[-1]['train_l_irf']:.4f} "
              f"vrex={history[-1]['train_l_vrex']:.4f} "
              f"alpha={history[-1]['train_l_alpha']:.4f} "
              f"lam_vrex={lambda_vrex_eff:.2f} "
              f"val_ic={val_metrics['ic']:.4f}")

        if val_metrics["ic"] > best_val_ic + 1e-5:
            best_val_ic = val_metrics["ic"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= train_cfg.early_stop_patience:
                print(f"[epiSTAR-SBP] early stop at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate(test_idx)
    val_metrics_final = evaluate(val_idx)

    out_dir = Path(train_cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / f"fold{fold}_seed{seed}_predictions.npz"
    np.savez_compressed(
        pred_path,
        y_hat=test_metrics["y_hat_all"], y_true=y,
        mask=test_metrics["eval_mask"],
        test_idx=np.asarray(test_idx, dtype=np.int64),
        tickers=np.asarray(tickers, dtype=str),
        dates=np.asarray([str(d) for d in dates], dtype=str),
        age_days=age_days.astype(np.int32),
        cohort_bucket=cohort_bkt.astype(np.int8),
    )
    if best_state is not None:
        torch.save(best_state, out_dir / f"fold{fold}_seed{seed}_ckpt.pt")
    out_path = out_dir / f"fold{fold}_seed{seed}.json"

    test_metrics.pop("y_hat_all", None); test_metrics.pop("eval_mask", None)
    val_metrics_final.pop("y_hat_all", None); val_metrics_final.pop("eval_mask", None)

    payload = {
        "fold": fold, "seed": seed, "model": "epiSTAR-SBP",
        "panel_start": train_cfg.panel_start, "panel_end": train_cfg.panel_end,
        "n_tickers": int(x.shape[1]), "n_dates": int(x.shape[0]),
        "n_train": int(len(train_idx)), "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "ic": test_metrics["ic"], "rank_ic": test_metrics["rank_ic"],
        "ndcg10": test_metrics["ndcg10"], "ndcg50": test_metrics["ndcg50"],
        "test_cohort_ic": test_metrics["cohort_ic"],
        "test_alpha_mean": test_metrics["alpha_mean"],
        "test_mu_mean": test_metrics["mu_mean"],
        "val_ic": val_metrics_final["ic"], "val_rank_ic": val_metrics_final["rank_ic"],
        "val_cohort_ic": val_metrics_final["cohort_ic"],
        "best_val_ic": best_val_ic,
        "irf_train_freq": irf_freq.detach().cpu().numpy().tolist(),
        "history": history,
        "config": {
            "train": asdict(train_cfg),
            "backbone": asdict(backbone_cfg),
            "memory": asdict(memory_cfg),
            "model": {
                "cohort_dim": sbp_cfg.cohort_dim,
                "dual_pool_m1": sbp_cfg.dual_pool_m1,
                "dual_pool_m2": sbp_cfg.dual_pool_m2,
                "tau_cohort": sbp_cfg.tau_cohort,
                "use_per_ticker_gate": sbp_cfg.use_per_ticker_gate,
                "use_two_head_xattn": sbp_cfg.use_two_head_xattn,
                "use_dual_pool": sbp_cfg.use_dual_pool,
                "disable_retrieval": sbp_cfg.disable_retrieval,
                "episode_value_dim": sbp_cfg.episode_value_dim,
            },
            "key_cols": key_cols,
        },
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"[epiSTAR-SBP] wrote {out_path}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/epistar_sbp.yaml")
    p.add_argument("--fold", type=int, choices=[1, 2, 3], required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.config, args.fold, args.seed, smoke=args.smoke)
