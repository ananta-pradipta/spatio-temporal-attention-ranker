"""Training loop for OW-epiSTAR v1.

Differs from train_epistar.py in three places:
    1. Builds an age-feature tensor and uses log1p_age + history_valid_ratio
       to construct a per-(day, ticker) IPO key.
    2. Applies correlation shrinkage when building the dynamic graph
       (rho_shrunk = rho * n_overlap / (n_overlap + tau=30)) so young
       nodes are not silently zero-correlated; soft replacement for the
       hard age cutoff that hurt epiDyReg-STAR's fold-1.
    3. Adds the per-(day, ticker) IPO analogue retrieval path with its
       own confidence gate (alpha_ipo) on top of the existing day-level
       gate (alpha_day).

All other components match epiSTAR-full: STAR backbone, day-level
EpisodeMemoryBank, cross-attention fusion, single rank head,
cross-sectional MSE loss on z-scored 5-day forward log returns.
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
from src.v2.data.age_features import AGE_COLS, AgeFeatureConfig, build_age_feature_tensor
from src.v2.data.episode_keys import (
    EpisodeKeyConfig, EPISODE_KEY_COLS, build_episode_keys,
)
from src.v2.model.episode_memory import EpisodeMemoryConfig
from src.v2.model.ipo_memory import IPOMemoryConfig
from src.v2.model.ow_epistar import OWEpiSTAR, OWEpiSTARConfig
from src.v2.model.star_backbone import STARBackboneConfig
from src.v2.training.folds import fold_indices


@dataclass
class TrainConfig:
    """Top-level hyperparameters."""

    fold: int = 1
    seed: int = 42
    panel_start: str = "2015-01-09"
    panel_end: str = "2022-12-31"
    horizon_days: int = 5
    universe_csv: str = "data/raw/biotech_universe_v1.csv"
    epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_steps: int = 500
    grad_clip: float = 1.0
    early_stop_patience: int = 3
    correlation_window: int = 60
    correlation_shrinkage_tau: float = 30.0
    min_overlap_soft: int = 5
    output_dir: str = "results/ow_epistar"


def set_seeds(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cs_mse_loss(y_hat, y_true, mask):
    m = mask.bool()
    yh = y_hat[m]; yt = y_true[m]
    if yt.numel() < 2:
        return torch.zeros((), device=y_hat.device, dtype=y_hat.dtype)
    mu = yt.mean(); sd = yt.std().clamp(min=1e-6)
    return ((yh - (yt - mu) / sd) ** 2).mean()


def per_day_ic(y_hat, y, mask, rank=False):
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
    return (np.nanmean(ics) if not np.all(np.isnan(ics)) else 0.0, ics)


def ndcg_at_k(y_hat, y, mask, k):
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


def standardize_features(x, mask, train_idx):
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
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))


def shrunk_correlation_neighbors(
    returns: np.ndarray, mask: np.ndarray, t: int,
    window: int, k: int, tau: float, min_overlap: int,
) -> np.ndarray:
    """Per-day top-K neighbors using shrunk rolling correlation.

    Soft replacement for the hard age-cutoff. Tickers with few overlap
    days get their correlation shrunk toward zero rather than excluded.
    """
    win_returns = returns[t - window + 1 : t + 1]
    win_mask = mask[t - window + 1 : t + 1]
    n = returns.shape[1]
    nan_filled = np.where(win_mask, np.where(np.isnan(win_returns), 0.0, win_returns), 0.0)
    valid_count = win_mask.sum(axis=0).astype(np.float32)
    # Rolling Pearson correlation on the masked-zero-filled window.
    x = nan_filled - nan_filled.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-8, 1e-8, sd)
    x_norm = x / sd
    rho = (x_norm.T @ x_norm) / window
    # Shrinkage based on overlapping count.
    overlap = np.minimum(valid_count[:, None], valid_count[None, :])
    shrinkage = overlap / (overlap + tau)
    rho_shrunk = rho * shrinkage
    rho_shrunk[overlap < min_overlap] = -np.inf
    np.fill_diagonal(rho_shrunk, -np.inf)
    rho_shrunk[~mask[t], :] = -np.inf
    rho_shrunk[:, ~mask[t]] = -np.inf
    top = np.full((n, k), -1, dtype=np.int64)
    if mask[t].sum() < 2:
        return top
    part = np.argpartition(-rho_shrunk, kth=min(k, n - 1), axis=1)[:, :k]
    for i in range(n):
        if not mask[t, i]:
            continue
        row_scores = rho_shrunk[i, part[i]]
        valid = np.isfinite(row_scores)
        chosen = part[i][valid]
        top[i, : len(chosen)] = chosen
    return top


IPO_KEY_COLS = [
    "log1p_age", "history_valid_ratio_60d", "log_market_cap", "cash_runway",
    "rd_intensity", "cash_to_mc", "rev_growth_yoy",
    "st_volume_24h", "st_bullish_ratio", "st_labeled_ratio",
    "rv_20d", "rv_60d", "log_return_5d", "log_return_20d",
]


def build_ipo_keys(x_raw: np.ndarray, age_feat: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per-(day, ticker) 14-dim IPO context key (raw, un-standardised).

    Indices into the 22-feature panel:
        0:log_return  1:log_return_5d  2:log_return_20d  3:log_volume
        4:log_volume_ratio_20d  5:rv_20d  6:rv_60d  7:hl_range
        8:close_to_high
        9:st_volume_24h  10:st_volume_change_30d  11:st_bullish_ratio
        12:st_sentiment_dispersion  13:st_labeled_ratio
        14:log_market_cap  15:cash_runway_q  16:rd_intensity
        17:revenue_growth_yoy  18:cash_to_mc  19:shares_outstanding_yoy
        20:total_assets_growth  21:has_fundamentals
    age_feat columns: 0:age_trading_days 1:log1p_age ... 7:history_valid_ratio_60d
    """
    t_total, n, _ = x_raw.shape
    keys = np.zeros((t_total, n, len(IPO_KEY_COLS)), dtype=np.float32)
    keys[..., 0] = age_feat[..., 1]            # log1p_age
    keys[..., 1] = age_feat[..., 7]            # history_valid_ratio_60d
    keys[..., 2] = x_raw[..., 14]              # log_market_cap
    keys[..., 3] = x_raw[..., 15]              # cash_runway_q
    keys[..., 4] = x_raw[..., 16]              # rd_intensity
    keys[..., 5] = x_raw[..., 18]              # cash_to_mc
    keys[..., 6] = x_raw[..., 17]              # rev_growth_yoy
    keys[..., 7] = x_raw[..., 9]               # st_volume_24h
    keys[..., 8] = x_raw[..., 11]              # st_bullish_ratio
    keys[..., 9] = x_raw[..., 13]              # st_labeled_ratio
    keys[..., 10] = x_raw[..., 5]              # rv_20d
    keys[..., 11] = x_raw[..., 6]              # rv_60d
    keys[..., 12] = x_raw[..., 1]              # log_return_5d
    keys[..., 13] = x_raw[..., 2]              # log_return_20d
    return keys


def main(cfg_path: str, fold: int, seed: int, smoke: bool = False) -> None:
    """Train OW-epiSTAR on one (fold, seed) pair."""
    with open(cfg_path) as f:
        raw = yaml.safe_load(f)
    train_cfg = TrainConfig(**{**raw.get("train", {}), "fold": fold, "seed": seed})
    backbone_cfg = STARBackboneConfig(**raw.get("backbone", {}))
    day_mem_cfg = EpisodeMemoryConfig(**raw.get("day_memory", {}))
    ipo_mem_cfg = IPOMemoryConfig(**raw.get("ipo_memory", {}))
    ow_cfg = OWEpiSTARConfig(
        backbone=backbone_cfg, day_memory=day_mem_cfg, ipo_memory=ipo_mem_cfg,
        episode_value_dim=raw.get("model", {}).get("episode_value_dim", 32),
        ipo_value_dim=raw.get("model", {}).get("ipo_value_dim", 32),
        cross_attn_heads=raw.get("model", {}).get("cross_attn_heads", 4),
        gate_hidden_dim=raw.get("model", {}).get("gate_hidden_dim", 64),
        head_hidden_dim=raw.get("model", {}).get("head_hidden_dim", 64),
        head_dropout=raw.get("model", {}).get("head_dropout", 0.1),
    )
    if smoke:
        train_cfg.epochs = 2

    set_seeds(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[OW-epiSTAR] fold={fold} seed={seed} device={device}")

    panel_cfg = EnrichedPanelConfig(
        start_date=pd.Timestamp(train_cfg.panel_start),
        end_date=pd.Timestamp(train_cfg.panel_end),
        horizon_days=train_cfg.horizon_days,
        universe_csv=Path(train_cfg.universe_csv),
    )
    panel, tickers, dates = build_enriched_panel(panel_cfg)
    tens = panel_to_tensors(panel, tickers, dates)
    x_raw = tens["x"]; y = tens["y"]; mask = tens["mask"]
    print(f"[OW-epiSTAR] panel: T={x_raw.shape[0]} N={x_raw.shape[1]} F={x_raw.shape[2]}")
    if x_raw.shape[1] < 50:
        raise RuntimeError(f"Panel has only {x_raw.shape[1]} active tickers; aborting.")

    train_idx, val_idx, test_idx = fold_indices(fold, dates)
    print(f"[OW-epiSTAR] fold {fold}: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    x = standardize_features(x_raw, mask, train_idx).astype(np.float32)

    age_feat, age_cols = build_age_feature_tensor(mask, AgeFeatureConfig())
    print(f"[OW-epiSTAR] age features: {len(age_cols)} dims")

    # Day-level keys + values (same as epiSTAR-full).
    day_keys, day_key_cols = build_episode_keys(
        dates=dates, log_returns=x_raw[..., 0], mask=mask, cfg=EpisodeKeyConfig(),
    )
    feature_idx = [0, 1, 5, 6]
    n_summary = 2 * len(feature_idx) + 1
    day_values = np.zeros((len(dates), day_keys.shape[1] + n_summary), dtype=np.float32)
    day_values[:, : day_keys.shape[1]] = day_keys
    for t in range(len(dates)):
        m = mask[t]
        if m.sum() < 5:
            continue
        for j, fi in enumerate(feature_idx):
            v = x_raw[t, m, fi]
            day_values[t, day_keys.shape[1] + 2 * j] = float(np.mean(v))
            day_values[t, day_keys.shape[1] + 2 * j + 1] = float(np.std(v))
        day_values[t, -1] = float(m.sum()) / 250.0
    print(f"[OW-epiSTAR] day-mem keys: {day_keys.shape[1]} dims, values: {day_values.shape[1]} dims")

    # IPO ticker-level keys + values.
    ipo_keys = build_ipo_keys(x_raw, age_feat, mask)
    # IPO value: the IPO key plus the realized return as a "what happened" tag.
    ipo_value_extras = np.zeros((len(dates), x_raw.shape[1], 1), dtype=np.float32)
    ipo_value_extras[..., 0] = y  # The 5-day forward return at (s, j).
    ipo_values = np.concatenate([ipo_keys, ipo_value_extras], axis=-1)
    print(f"[OW-epiSTAR] IPO keys: {ipo_keys.shape[-1]} dims, values: {ipo_values.shape[-1]} dims")

    # Build the IPO memory bank flat list: (day, ticker) entries with
    # age <= max_age_days, listed_mask=label_mask=True, training-fold only.
    ipo_eligible = (age_feat[..., 0] <= ipo_mem_cfg.max_age_days) & mask
    ipo_eligible[~mask] = False
    train_set = set(int(t) for t in train_idx.tolist())
    flat_keys = []; flat_values = []; flat_days = []; flat_tickers = []
    for t in range(len(dates)):
        if t not in train_set:
            continue
        active_tickers = np.where(ipo_eligible[t])[0]
        for i in active_tickers:
            flat_keys.append(ipo_keys[t, i])
            flat_values.append(ipo_values[t, i])
            flat_days.append(t)
            flat_tickers.append(int(i))
    flat_keys = np.array(flat_keys, dtype=np.float32) if flat_keys else np.zeros((0, ipo_keys.shape[-1]), dtype=np.float32)
    flat_values = np.array(flat_values, dtype=np.float32) if flat_values else np.zeros((0, ipo_values.shape[-1]), dtype=np.float32)
    flat_days = np.array(flat_days, dtype=np.int64) if flat_days else np.zeros((0,), dtype=np.int64)
    flat_tickers = np.array(flat_tickers, dtype=np.int64) if flat_tickers else np.zeros((0,), dtype=np.int64)
    print(f"[OW-epiSTAR] IPO memory entries (train-fold, age<={ipo_mem_cfg.max_age_days}d): {len(flat_keys)}")

    ow_cfg.episode_value_dim = day_values.shape[1]
    ow_cfg.ipo_value_dim = ipo_values.shape[-1]
    model = OWEpiSTAR(
        ow_cfg, day_key_dim=day_keys.shape[1], ipo_key_dim=ipo_keys.shape[-1],
    ).to(device)
    model.day_memory.populate(
        keys=day_keys, values=day_values, day_indices=np.arange(len(dates)),
        train_day_indices=train_idx,
    )
    model.day_memory.to(device)
    model.ipo_memory.populate(
        keys=flat_keys, values=flat_values,
        day_indices=flat_days, ticker_indices=flat_tickers,
        train_day_indices=train_idx,
    )
    model.ipo_memory.to(device)

    allowed_train = torch.from_numpy(train_idx).long().to(device)
    optim = AdamW(model.parameters(), lr=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay)
    total_steps = train_cfg.epochs * max(1, len(train_idx))
    scheduler = LambdaLR(optim, lr_lambda=lambda s: warmup_cosine_lr(s, train_cfg.warmup_steps, total_steps))
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

        # Shrunk-correlation dynamic neighbor list for this day.
        neigh = shrunk_correlation_neighbors(
            x_raw[..., 0], mask, int(t_idx),
            train_cfg.correlation_window, backbone_cfg.num_neighbors,
            train_cfg.correlation_shrinkage_tau, train_cfg.min_overlap_soft,
        )
        top_neighbors_day_t = torch.from_numpy(neigh).to(device)
        x_window = torch.from_numpy(x[t_idx - w + 1 : t_idx + 1]).to(device)
        mask_window = torch.from_numpy(mask[t_idx - w + 1 : t_idx + 1]).to(device)
        patches, patch_mask = build_patches(
            x_window=x_window, mask_window=mask_window,
            top_neighbors=top_neighbors_day_t, active_idx=active_idx,
        )

        day_query_key = torch.from_numpy(day_keys[t_idx]).float().to(device)
        regime_scalars = model.day_memory.standardize_query(day_query_key)[[0, 9]].clone()
        if torch.isnan(regime_scalars).any():
            regime_scalars = torch.zeros(2, device=device)

        # Per-active-ticker IPO query keys and gate features.
        active_idx_np = active_idx.detach().cpu().numpy()
        ipo_query_keys = torch.from_numpy(ipo_keys[t_idx, active_idx_np]).float().to(device)
        # Gate features: log1p_age and history_valid_ratio_60d per active ticker.
        ipo_gate_feats = torch.from_numpy(
            age_feat[t_idx, active_idx_np][:, [1, 7]]
        ).float().to(device)

        with autocast(enabled=use_amp, dtype=torch.float16):
            out = model.forward_day(
                patches=patches, patch_mask=patch_mask, active_mask=active_mask_t,
                day_query_key=day_query_key, ipo_query_keys=ipo_query_keys,
                ipo_gate_features=ipo_gate_feats,
                query_day_idx=int(t_idx), allowed_day_indices=allowed_train,
                gate_regime_scalars=regime_scalars,
            )
        out["y_hat"] = out["y_hat"].float()
        out["active_mask"] = active_mask_t
        out["t_idx"] = t_idx
        return out

    @torch.no_grad()
    def evaluate(idx):
        model.eval()
        T = x.shape[0]; N = x.shape[1]
        y_hat_all = np.zeros((T, N), dtype=np.float32)
        eval_mask = np.zeros((T, N), dtype=bool)
        for t_idx in idx:
            out = forward_one_day(int(t_idx))
            if not out:
                continue
            y_hat_all[t_idx] = out["y_hat"].detach().cpu().numpy()
            eval_mask[t_idx] = mask[t_idx]
        ic, ic_per_day = per_day_ic(y_hat_all, y, eval_mask, rank=False)
        rank_ic, _ = per_day_ic(y_hat_all, y, eval_mask, rank=True)
        ndcg10 = ndcg_at_k(y_hat_all, y, eval_mask, 10)
        ndcg50 = ndcg_at_k(y_hat_all, y, eval_mask, 50)
        return {"ic": ic, "rank_ic": rank_ic, "ndcg10": ndcg10, "ndcg50": ndcg50,
                "y_hat_all": y_hat_all, "eval_mask": eval_mask}

    step = 0; smoke_step_cap = 80
    for epoch in range(train_cfg.epochs):
        model.train()
        np.random.seed(seed + epoch)
        perm = np.random.permutation(train_idx)
        epoch_losses: list[float] = []
        for t_idx in perm:
            t_idx = int(t_idx)
            if t_idx < w:
                continue
            out = forward_one_day(t_idx)
            if not out:
                continue
            y_true_t = torch.from_numpy(y[t_idx]).to(device)
            mask_t = out["active_mask"]
            loss = cs_mse_loss(out["y_hat"], y_true_t, mask_t)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            scaler.step(optim); scaler.update(); scheduler.step()
            epoch_losses.append(float(loss.item()))
            step += 1
            if smoke and step >= smoke_step_cap:
                break

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        val_metrics = evaluate(val_idx)
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_ic": val_metrics["ic"], "val_rank_ic": val_metrics["rank_ic"]})
        print(f"[OW-epiSTAR] epoch {epoch}: train_loss={train_loss:.4f} "
              f"val_ic={val_metrics['ic']:.4f} val_rank_ic={val_metrics['rank_ic']:.4f}")
        if val_metrics["ic"] > best_val_ic + 1e-5:
            best_val_ic = val_metrics["ic"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= train_cfg.early_stop_patience:
                print(f"[OW-epiSTAR] early stop at epoch {epoch}")
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
        y_hat=test_metrics["y_hat_all"], y_true=y, mask=test_metrics["eval_mask"],
        test_idx=np.asarray(test_idx, dtype=np.int64),
        tickers=np.asarray(tickers, dtype=str),
        dates=np.asarray([str(d) for d in dates], dtype=str),
        age_days=age_feat[..., 0].astype(np.int32),
    )
    if best_state is not None:
        torch.save(best_state, out_dir / f"fold{fold}_seed{seed}_ckpt.pt")
    out_path = out_dir / f"fold{fold}_seed{seed}.json"
    payload = {
        "fold": fold, "seed": seed, "model": "OW-epiSTAR",
        "panel_start": train_cfg.panel_start, "panel_end": train_cfg.panel_end,
        "n_tickers": int(x.shape[1]), "n_dates": int(x.shape[0]),
        "n_train": int(len(train_idx)), "n_val": int(len(val_idx)), "n_test": int(len(test_idx)),
        "n_ipo_memory_entries": int(len(flat_keys)),
        "ic": test_metrics["ic"], "rank_ic": test_metrics["rank_ic"],
        "ndcg10": test_metrics["ndcg10"], "ndcg50": test_metrics["ndcg50"],
        "val_ic": val_metrics_final["ic"], "val_rank_ic": val_metrics_final["rank_ic"],
        "best_val_ic": best_val_ic, "history": history,
        "config": {"train": asdict(train_cfg), "backbone": asdict(backbone_cfg),
                   "day_memory": asdict(day_mem_cfg), "ipo_memory": asdict(ipo_mem_cfg),
                   "model": {"head_hidden_dim": ow_cfg.head_hidden_dim,
                             "head_dropout": ow_cfg.head_dropout,
                             "episode_value_dim": ow_cfg.episode_value_dim,
                             "ipo_value_dim": ow_cfg.ipo_value_dim}},
    }
    test_metrics.pop("y_hat_all", None); test_metrics.pop("eval_mask", None)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"[OW-epiSTAR] wrote {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/ow_epistar.yaml")
    p.add_argument("--fold", type=int, choices=[1, 2, 3], required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.config, args.fold, args.seed, smoke=args.smoke)
