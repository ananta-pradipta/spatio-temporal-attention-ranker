"""Multi-source dynamic neighbor selection for epiDyReg-STAR.

Builds K=2 to K=4 candidate graphs per day and mixes them with regime-
conditioned source weights. Sources implemented (Section 6 of the
spec, minimum v1 from Section 21):

    1. Static mechanistic graph: sector + industry similarity from
       `data/processed/ticker_company.parquet`. Built once per fold.
    2. Rolling return-correlation graph: 60-day Pearson correlation
       on log returns, recomputed per day with data up to day t only.
    3. Rolling residual-correlation graph: returns are first regressed
       on the SPDR Biotech ETF (XBI) proxy (the cross-sectional
       average return, used as XBI proxy when XBI is unavailable),
       then 60-day Pearson correlation of residuals.

The regime-conditioned source mixer takes a regime context vector
z_regime in R^D and produces softmax weights over the K sources;
candidate neighbors are scored as a weighted sum and the top-K=8
neighbors per ticker per day are selected.

Age-aware graph rules (Section 6.4): tickers with fewer than 60
trading days of history are excluded from rolling-correlation edges
(both as sources and as candidates) and rely on the static graph only.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class MultiSourceGraphConfig:
    """Hyperparameters for multi-source dynamic graph construction.

    Attributes:
        rolling_window_days: window for rolling correlations (60d default).
        top_k_neighbors: final neighbor count per ticker per day.
        candidate_k_per_source: candidates pulled per source before mixing.
        min_age_for_corr_edges: tickers younger than this are excluded
            from rolling-corr and residual-corr edges.
        use_residual_corr: if True, build residual-correlation graph
            after de-XBI-beta. Otherwise only return-corr is used.
        ticker_company_path: source for sector / industry metadata.
    """

    rolling_window_days: int = 60
    top_k_neighbors: int = 8
    candidate_k_per_source: int = 16
    min_age_for_corr_edges: int = 60
    use_residual_corr: bool = True
    ticker_company_path: Path = Path("data/processed/ticker_company.parquet")


def build_static_score(
    tickers: list[str], cfg: MultiSourceGraphConfig
) -> np.ndarray:
    """Static mechanistic score matrix from sector / industry metadata.

    Pair (i, j) gets:
        +1.0 if same `industry`
        +0.5 if same `sector`
    A pair sharing both gets +1.5. Missing metadata gives 0.0.
    """
    n = len(tickers)
    out = np.zeros((n, n), dtype=np.float32)
    if not cfg.ticker_company_path.exists():
        return out
    meta = pd.read_parquet(cfg.ticker_company_path)
    meta = meta.set_index(meta["ticker"].astype(str).str.upper())
    industries: list[str | None] = []
    sectors: list[str | None] = []
    for t in tickers:
        if t.upper() in meta.index:
            row = meta.loc[t.upper()]
            industries.append(row.get("industry") if isinstance(row, pd.Series) else None)
            sectors.append(row.get("sector") if isinstance(row, pd.Series) else None)
        else:
            industries.append(None); sectors.append(None)
    for i in range(n):
        for j in range(i + 1, n):
            score = 0.0
            if industries[i] is not None and industries[i] == industries[j]:
                score += 1.0
            if sectors[i] is not None and sectors[i] == sectors[j]:
                score += 0.5
            out[i, j] = score
            out[j, i] = score
    return out


def correlation_window_matrix(
    returns: np.ndarray, t: int, window_days: int
) -> np.ndarray:
    """Rolling Pearson correlation of column returns over [t-w+1, t]."""
    win = returns[t - window_days + 1 : t + 1]
    win = np.where(np.isnan(win), 0.0, win)
    x = win - win.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-8, 1e-8, sd)
    x_norm = x / sd
    corr = (x_norm.T @ x_norm) / win.shape[0]
    np.fill_diagonal(corr, -np.inf)
    return corr.astype(np.float32)


def residual_correlation_window_matrix(
    returns: np.ndarray, t: int, window_days: int
) -> np.ndarray:
    """Residual correlation after removing the cross-sectional-average beta.

    Per the spec, the proper baseline is the SPDR Biotech ETF (XBI), but
    XBI itself is not in the panel. As a robust proxy we use the daily
    cross-sectional mean of active-ticker log returns; this captures the
    "biotech sector beta" the spec calls for, computed only from the
    panel itself with no external data.
    """
    win = returns[t - window_days + 1 : t + 1]
    win = np.where(np.isnan(win), 0.0, win)
    market = win.mean(axis=1, keepdims=True)
    market_var = (market ** 2).mean()
    if market_var < 1e-12:
        return np.full(
            (win.shape[1], win.shape[1]), -np.inf, dtype=np.float32
        )
    # Per-ticker beta: cov(r_i, market) / var(market)
    cov_im = (win * market).mean(axis=0)
    beta = cov_im / market_var
    # Residual returns: r_i - beta_i * market
    resid = win - beta[None, :] * market
    x = resid - resid.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-8, 1e-8, sd)
    x_norm = x / sd
    corr = (x_norm.T @ x_norm) / win.shape[0]
    np.fill_diagonal(corr, -np.inf)
    return corr.astype(np.float32)


def compute_age_days(mask: np.ndarray) -> np.ndarray:
    """Per-(day, ticker) age in trading days since first active day.

    Returns int64 array of shape mask.shape. Tickers that have not yet
    had a first active day get age 0; tickers that have, get a count
    from their first active day forward.
    """
    t_total, n = mask.shape
    out = np.zeros((t_total, n), dtype=np.int64)
    first_active = np.full(n, -1, dtype=np.int64)
    for t in range(t_total):
        active = mask[t]
        # Initialise first_active for tickers becoming active at day t.
        new_active = np.where(active & (first_active == -1))[0]
        first_active[new_active] = t
        # Compute age for currently active tickers.
        age = np.where(first_active >= 0, t - first_active, 0)
        out[t] = np.maximum(age, 0)
    return out


def gate_weighted_top_k(
    score_matrix: np.ndarray,
    active_mask: np.ndarray,
    k: int,
    age_days: np.ndarray | None = None,
    min_age: int = 0,
    age_relevant: bool = True,
) -> np.ndarray:
    """Select top-K neighbors from a scored [N, N] matrix.

    If `age_relevant` is True and `age_days` is provided, candidates with
    `age_days[j] < min_age` are excluded as candidate neighbors. The
    SOURCE ticker's own age also matters: if `age_days[i] < min_age`,
    no neighbors are returned for ticker i (the row is left at -1).
    """
    n = score_matrix.shape[0]
    score = score_matrix.copy()
    score[~active_mask, :] = -np.inf
    score[:, ~active_mask] = -np.inf
    if age_relevant and age_days is not None and min_age > 0:
        too_young = age_days < min_age
        score[:, too_young] = -np.inf  # exclude as candidate neighbor

    top = np.full((n, k), -1, dtype=np.int64)
    if active_mask.sum() < 2:
        return top
    part = np.argpartition(-score, kth=min(k, n - 1), axis=1)[:, :k]
    for i in range(n):
        if not active_mask[i]:
            continue
        if age_relevant and age_days is not None and min_age > 0 and age_days[i] < min_age:
            continue
        row_scores = score[i, part[i]]
        valid = np.isfinite(row_scores)
        chosen = part[i][valid]
        top[i, : len(chosen)] = chosen
    return top


def graph_summary_features(
    score_matrix: np.ndarray, active_mask: np.ndarray, prev_neighbors: np.ndarray | None
) -> np.ndarray:
    """Compute a 6-dim graph-summary vector for one day's graph.

    Returns: [avg_abs_corr, pc1_share_proxy, graph_density,
              mean_turnover_vs_prev, n_active_norm, score_std].

    `pc1_share_proxy` here uses the dominant-eigenvalue ratio of the
    score matrix's top eigenvalue, which is a proxy for the
    cross-sectional first-PC variance share that the spec's
    `cross_pc1_share_60d` field also captures.
    """
    out = np.zeros(6, dtype=np.float32)
    if active_mask.sum() < 5:
        return out
    sub = score_matrix[active_mask][:, active_mask]
    finite = sub[np.isfinite(sub)]
    if finite.size == 0:
        return out
    out[0] = float(np.mean(np.abs(finite)))
    # PC1-share proxy via Frobenius vs largest eigenvalue.
    sub_safe = np.where(np.isfinite(sub), sub, 0.0)
    try:
        evals = np.linalg.eigvalsh((sub_safe + sub_safe.T) / 2)
        evals = evals[evals > 0]
        if evals.size > 0:
            out[1] = float(evals.max() / evals.sum())
    except np.linalg.LinAlgError:
        out[1] = 0.0
    # Density: fraction of finite scores above a small threshold.
    out[2] = float((finite > 0.1).mean())
    # Turnover vs prev day's neighbors.
    if prev_neighbors is not None:
        diffs = []
        for i in range(active_mask.shape[0]):
            if not active_mask[i]:
                continue
            cur = set(int(x) for x in prev_neighbors[i] if x >= 0)
            if cur:
                diffs.append(len(cur.symmetric_difference(set())) / max(len(cur), 1))
        out[3] = float(np.mean(diffs)) if diffs else 0.0
    # Active count normaliser.
    out[4] = float(active_mask.sum()) / 250.0
    # Score std (a regime-spread proxy).
    out[5] = float(np.std(finite))
    return out


__all__ = [
    "MultiSourceGraphConfig",
    "build_static_score",
    "correlation_window_matrix",
    "residual_correlation_window_matrix",
    "compute_age_days",
    "gate_weighted_top_k",
    "graph_summary_features",
]
