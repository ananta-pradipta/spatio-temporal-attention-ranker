"""Shared baseline runner using the v2 protocol used by RAG-STAR.

All baselines (MASTER, StockMixer, DySTAGE, RSR) consume the SAME panel,
masks, fold definitions, embargo, seeds, and evaluation as
``src.v2.training.train_dow_epistar``. This guarantees fair comparison:
each baseline differs only in its model architecture, not in its data
pipeline or evaluation protocol.

Provides:
    build_panel       : 244-ticker biotech panel, 22 features, 2015-2022
    build_masks       : tradable / label / loss masks
    fold_split        : v2 fold_indices wrapper
    standardize       : feature standardisation with train-fold stats
    cs_mse_loss       : per-day cross-sectional MSE on z-scored target
    per_day_ic        : day-averaged Pearson IC
    per_day_rank_ic   : day-averaged Spearman rank IC
    ndcg_at_k         : NDCG@k for k in {10, 50}
    cohort_ic         : age-stratified IC (fresh_ipo_60d, young_public_252d, seasoned_253d)
    save_result       : JSON + NPZ output in the format used by train_dow_epistar
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import torch

from src.mtgn.training.panel_enriched import (
    EnrichedPanelConfig, build_enriched_panel, panel_to_tensors,
)
from src.v2.data.minimal_masks import (
    MinimalMaskConfig, build_minimal_masks, compute_age_from_tradable,
)
from src.v2.training.folds import fold_indices


# ============================================================================
# Configuration shared by every v2 baseline.
# ============================================================================

PANEL_START = "2015-01-09"
PANEL_END = "2022-12-31"
HORIZON_DAYS = 5
EMBARGO_DAYS = 5
UNIVERSE_CSV = "data/raw/biotech_universe_v1.csv"
SEEDS = (42, 43, 44, 45, 46)


@dataclass
class V2BaselineConfig:
    """Top-level baseline hyperparameters. Subclasses extend with model-specific keys."""

    fold: int = 1
    seed: int = 42
    panel_start: str = PANEL_START
    panel_end: str = PANEL_END
    horizon_days: int = HORIZON_DAYS
    universe_csv: str = UNIVERSE_CSV
    epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_steps: int = 500
    grad_clip: float = 1.0
    early_stop_patience: int = 3
    temporal_window: int = 20
    output_dir: str = "results/baselines_244"
    # Universal-panel switches added 2026-05-13. When panel_kind="lattice_native"
    # the runner builds the 26-col S&P 500 panel instead of the biotech-244
    # 22-col panel. When two_regime_val=True the fold split uses
    # fold_indices_two_regime_val (val=2017H2+2018H2, two-segment train).
    panel_kind: str = "biotech"   # "biotech" | "lattice_native"
    two_regime_val: bool = False


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# Panel + masks (identical to train_dow_epistar.py).
# ============================================================================


def build_panel(cfg: V2BaselineConfig):
    """Build the panel and return (x_raw, y, tickers, dates).

    When cfg.panel_kind=="biotech" (default): 244-ticker biotech panel, 22 features.
    When cfg.panel_kind=="lattice_native": ~600-ticker S&P 500 panel, 26 features
        (matches the universal RAG-STAR sweep panel).
    """
    if cfg.panel_kind == "lattice_native":
        from src.v2.data.lattice_native_panel import (
            LatticeNativePanelConfig, build_lattice_native_panel, FEATURE_COLS,
        )
        panel_cfg_l = LatticeNativePanelConfig(
            lattice_dir=Path("data/lattice/processed"),
            start_date=cfg.panel_start,
            end_date=cfg.panel_end,
            horizon_days=cfg.horizon_days,
        )
        panel, tickers, dates = build_lattice_native_panel(panel_cfg_l)
        # Build (T, N, F) tensors mirroring panel_to_tensors but for the
        # lattice_native schema. Reindex by (date, ticker) and then pivot
        # column-by-column to preserve the FEATURE_COLS order.
        T = len(dates); N = len(tickers); F = len(FEATURE_COLS)
        ticker_to_i = {tk: i for i, tk in enumerate(tickers)}
        date_to_t = {d: i for i, d in enumerate(dates)}
        x = np.zeros((T, N, F), dtype=np.float32)
        y = np.zeros((T, N), dtype=np.float32)
        # Vectorised fill via groupby on the long dataframe.
        panel = panel.copy()
        panel["t_idx"] = panel["date"].map(date_to_t).astype("int64")
        panel["n_idx"] = panel["ticker"].map(ticker_to_i).astype("int64")
        ti = panel["t_idx"].to_numpy()
        ni = panel["n_idx"].to_numpy()
        y[ti, ni] = panel["fwd_return_h"].astype(np.float32).to_numpy()
        for f, col in enumerate(FEATURE_COLS):
            x[ti, ni, f] = panel[col].astype(np.float32).to_numpy()
        return x, y, tickers, dates
    # Default: biotech-244, 22-feature panel.
    panel_cfg = EnrichedPanelConfig(
        start_date=pd.Timestamp(cfg.panel_start),
        end_date=pd.Timestamp(cfg.panel_end),
        horizon_days=cfg.horizon_days,
        universe_csv=Path(cfg.universe_csv),
    )
    panel, tickers, dates = build_enriched_panel(panel_cfg)
    tens = panel_to_tensors(panel, tickers, dates)
    return tens["x"], tens["y"], tickers, dates


def build_masks(cfg: V2BaselineConfig, dates: list, tickers: list):
    """Build tradable + label + loss masks. Mirrors train_dow_epistar mask config.

    For the universal panel, route to the extended S&P 500 prices parquet so
    F4/F5 dates have valid mask cells (matches train_dow_epistar.py logic).
    """
    mask_kwargs: dict = dict(horizon_days=cfg.horizon_days)
    if cfg.panel_kind == "lattice_native":
        ext_prices = Path("data/raw/sp500/prices_sp500_extended.parquet")
        canon_prices = Path("data/raw/sp500/prices_sp500.parquet")
        mask_kwargs["raw_prices_parquet"] = (
            ext_prices if ext_prices.exists() else canon_prices
        )
    return build_minimal_masks(dates, tickers, MinimalMaskConfig(**mask_kwargs))


def build_age_features(tradable_mask: np.ndarray, hist20: np.ndarray, hist60: np.ndarray) -> np.ndarray:
    """[T, N, 8] age features from cumsum-tradable mask."""
    age = compute_age_from_tradable(tradable_mask).astype(np.float32)
    out = np.zeros((*tradable_mask.shape, 8), dtype=np.float32)
    out[..., 0] = age
    out[..., 1] = np.log1p(age)
    out[..., 2] = ((age >= 1) & (age <= 20)).astype(np.float32)
    out[..., 3] = ((age > 20) & (age <= 60)).astype(np.float32)
    out[..., 4] = ((age > 60) & (age <= 252)).astype(np.float32)
    out[..., 5] = (age > 252).astype(np.float32)
    out[..., 6] = hist20
    out[..., 7] = hist60
    return out


def fold_split(cfg: V2BaselineConfig, dates: list):
    """Return (train_idx, val_idx, test_idx).

    For panel_kind="lattice_native" with two_regime_val=True, uses
    fold_indices_two_regime_val (val = 2017 H2 + 2018 H2, two-segment train).
    For panel_kind="lattice_native" without two_regime_val, uses the
    standard 5-fold lattice fold_indices.
    Otherwise falls back to the biotech 3-fold v2 fold_indices.
    """
    if cfg.panel_kind == "lattice_native":
        if cfg.two_regime_val:
            from src.lattice.data.folds import fold_indices_two_regime_val
            return fold_indices_two_regime_val(cfg.fold, dates)
        from src.lattice.data.folds import fold_indices as lattice_fold_indices
        return lattice_fold_indices(cfg.fold, dates)
    return fold_indices(cfg.fold, dates)


def standardize_features(x: np.ndarray, mask: np.ndarray, train_idx: np.ndarray) -> np.ndarray:
    """Per-feature standardisation using training-fold mean and std only."""
    out = np.zeros_like(x)
    flat_train_mask = mask[train_idx]
    x_train = x[train_idx]
    for f in range(x.shape[2]):
        vals = x_train[..., f][flat_train_mask]
        if vals.size < 2:
            mu, sd = 0.0, 1.0
        else:
            mu = float(np.mean(vals))
            sd = float(np.std(vals))
            if sd < 1e-6:
                sd = 1.0
        out[..., f] = (x[..., f] - mu) / sd
    return (out * mask[..., None]).astype(np.float32)


# ============================================================================
# Loss + metrics (identical to train_dow_epistar.py).
# ============================================================================


def cs_mse_loss(y_hat: torch.Tensor, y_true: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-day cross-sectional MSE on z-scored 5d forward log returns."""
    m = mask.bool()
    yh = y_hat[m]
    yt = y_true[m]
    if yt.numel() < 2:
        return torch.zeros((), device=y_hat.device, dtype=y_hat.dtype)
    mu = yt.mean()
    sd = yt.std().clamp(min=1e-6)
    return ((yh - (yt - mu) / sd) ** 2).mean()


def per_day_ic(y_hat: np.ndarray, y: np.ndarray, mask: np.ndarray, rank: bool = False) -> tuple[float, np.ndarray]:
    t_total = y_hat.shape[0]
    ics = np.full(t_total, np.nan, dtype=np.float64)
    for t in range(t_total):
        m = mask[t]
        if m.sum() < 5:
            continue
        a = y_hat[t, m]
        b = y[t, m]
        if rank:
            a = pd.Series(a).rank().to_numpy()
            b = pd.Series(b).rank().to_numpy()
        if a.std() < 1e-9 or b.std() < 1e-9:
            continue
        ics[t] = float(np.corrcoef(a, b)[0, 1])
    if np.all(np.isnan(ics)):
        return 0.0, ics
    return float(np.nanmean(ics)), ics


def ndcg_at_k(y_hat: np.ndarray, y: np.ndarray, mask: np.ndarray, k: int) -> float:
    out = []
    for t in range(y_hat.shape[0]):
        m = mask[t]
        if m.sum() < k + 1:
            continue
        scores = y_hat[t, m]
        rels = y[t, m]
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


def cohort_ic(y_hat: np.ndarray, y: np.ndarray, eval_mask: np.ndarray, age_days: np.ndarray) -> dict:
    """IC stratified by ticker age. Cohorts match train_dow_epistar.py."""
    out: dict[str, float] = {}
    cohorts = {
        "all": np.ones_like(eval_mask, dtype=bool),
        "fresh_ipo_60d": (age_days <= 60) & (age_days >= 1),
        "young_public_252d": (age_days > 60) & (age_days <= 252),
        "seasoned_253d": age_days > 252,
    }
    for label, cohort_mask in cohorts.items():
        ics = []
        for t in range(y_hat.shape[0]):
            m = eval_mask[t] & cohort_mask[t]
            if m.sum() < 5:
                continue
            a = y_hat[t, m]
            b = y[t, m]
            if a.std() < 1e-9 or b.std() < 1e-9:
                continue
            ics.append(float(np.corrcoef(a, b)[0, 1]))
        out[label] = float(np.mean(ics)) if ics else 0.0
    return out


def warmup_cosine_lr(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))


# ============================================================================
# Evaluation + result-saving (matches train_dow_epistar.py JSON format).
# ============================================================================


def evaluate_predictions(
    y_hat_all: np.ndarray, y: np.ndarray, eval_mask: np.ndarray, age_days: np.ndarray,
) -> dict:
    """Compute the headline metrics from per-(day, ticker) predictions."""
    ic, _ = per_day_ic(y_hat_all, y, eval_mask, rank=False)
    rank_ic, _ = per_day_ic(y_hat_all, y, eval_mask, rank=True)
    ndcg10 = ndcg_at_k(y_hat_all, y, eval_mask, 10)
    ndcg50 = ndcg_at_k(y_hat_all, y, eval_mask, 50)
    coh = cohort_ic(y_hat_all, y, eval_mask, age_days)
    return {
        "ic": ic,
        "rank_ic": rank_ic,
        "ndcg10": ndcg10,
        "ndcg50": ndcg50,
        "cohort_ic": coh,
    }


def save_result(
    out_dir: Path,
    fold: int,
    seed: int,
    model_name: str,
    test_metrics: dict,
    val_metrics: dict,
    test_y_hat: np.ndarray,
    test_eval_mask: np.ndarray,
    history: list,
    config: dict,
    n_panel: tuple[int, int, int],
    n_train: int,
    n_val: int,
    n_test: int,
    y_true: np.ndarray,
    tickers: list,
    dates: list,
    age_days: np.ndarray,
    tradable_mask: np.ndarray,
) -> Path:
    """Save per-run JSON + predictions NPZ in the format used by RAG-STAR."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / f"fold{fold}_seed{seed}_predictions.npz"
    np.savez_compressed(
        pred_path,
        y_hat=test_y_hat,
        y_true=y_true,
        loss_mask=test_eval_mask,
        tradable_mask=tradable_mask,
        tickers=np.asarray(tickers, dtype=str),
        dates=np.asarray([str(d) for d in dates], dtype=str),
        age_days=age_days.astype(np.int32),
    )
    out_path = out_dir / f"fold{fold}_seed{seed}.json"
    payload = {
        "fold": fold,
        "seed": seed,
        "model": model_name,
        "panel_T": int(n_panel[0]),
        "panel_N": int(n_panel[1]),
        "panel_F": int(n_panel[2]),
        "n_train": int(n_train),
        "n_val": int(n_val),
        "n_test": int(n_test),
        "ic": test_metrics["ic"],
        "rank_ic": test_metrics["rank_ic"],
        "ndcg10": test_metrics["ndcg10"],
        "ndcg50": test_metrics["ndcg50"],
        "test_cohort_ic": test_metrics["cohort_ic"],
        "val_ic": val_metrics["ic"],
        "val_rank_ic": val_metrics["rank_ic"],
        "val_cohort_ic": val_metrics["cohort_ic"],
        "history": history,
        "config": config,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    return out_path


__all__ = [
    "PANEL_START",
    "PANEL_END",
    "HORIZON_DAYS",
    "EMBARGO_DAYS",
    "UNIVERSE_CSV",
    "SEEDS",
    "V2BaselineConfig",
    "set_seeds",
    "build_panel",
    "build_masks",
    "build_age_features",
    "fold_split",
    "standardize_features",
    "cs_mse_loss",
    "per_day_ic",
    "ndcg_at_k",
    "cohort_ic",
    "warmup_cosine_lr",
    "evaluate_predictions",
    "save_result",
]
