"""InVAR evaluation metrics.

  - daily_ic           : per-day Pearson IC; mean across days (NaN-safe).
  - daily_rank_ic      : per-day Spearman IC; mean across days.
  - ndcg_at_k          : per-day NDCG at k in {10, 50}.
  - cohort_stratified_ic: per-cohort mean rank IC for sector / size /
                          age cohorts.
  - long_short_sharpe  : annualised Sharpe of a long-short portfolio
                          formed from the top and bottom decile of y_hat,
                          with a transaction-cost haircut in basis points.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def daily_ic(predictions: pd.DataFrame) -> dict:
    """Per-day Pearson IC and mean across days.

    Args:
        predictions: DataFrame with columns ``(date, ticker, y_hat,
            y_true)`` and at least 5 active rows per day.
    """
    ics = []
    for date, group in predictions.groupby("date"):
        if len(group) < 5:
            continue
        y_hat = group["y_hat"].to_numpy()
        y_true = group["y_true"].to_numpy()
        if y_hat.std() < 1e-9 or y_true.std() < 1e-9:
            continue
        ics.append(float(np.corrcoef(y_hat, y_true)[0, 1]))
    if not ics:
        return {"mean": float("nan"), "std": float("nan"), "n_days": 0}
    return {
        "mean": float(np.mean(ics)),
        "std": float(np.std(ics)),
        "n_days": len(ics),
    }


def daily_rank_ic(predictions: pd.DataFrame) -> dict:
    """Per-day Spearman rank IC across days."""
    from scipy.stats import spearmanr
    ics = []
    for date, group in predictions.groupby("date"):
        if len(group) < 5:
            continue
        y_hat = group["y_hat"].to_numpy()
        y_true = group["y_true"].to_numpy()
        if y_hat.std() < 1e-9 or y_true.std() < 1e-9:
            continue
        rho, _ = spearmanr(y_hat, y_true)
        if np.isfinite(rho):
            ics.append(float(rho))
    if not ics:
        return {"mean": float("nan"), "std": float("nan"), "n_days": 0}
    return {
        "mean": float(np.mean(ics)),
        "std": float(np.std(ics)),
        "n_days": len(ics),
    }


def ndcg_at_k(predictions: pd.DataFrame, k: int = 10) -> dict:
    """Per-day NDCG at k.

    Treat y_true as relevance (clipped to >= 0); compute DCG and IDCG
    over the top-k by y_hat and y_true respectively.
    """
    scores = []
    for date, group in predictions.groupby("date"):
        if len(group) < k + 1:
            continue
        y_hat = group["y_hat"].to_numpy()
        y_true = group["y_true"].to_numpy()
        rel = np.clip(y_true, a_min=0.0, a_max=None)
        order_pred = np.argsort(-y_hat)[:k]
        order_true = np.argsort(-y_true)[:k]
        gains_pred = (2.0 ** rel[order_pred] - 1.0)
        gains_true = (2.0 ** rel[order_true] - 1.0)
        denom = np.log2(np.arange(2, k + 2))
        dcg = float((gains_pred / denom).sum())
        idcg = float((gains_true / denom).sum())
        if idcg <= 1e-9:
            continue
        scores.append(dcg / idcg)
    if not scores:
        return {"mean": float("nan"), "n_days": 0}
    return {"mean": float(np.mean(scores)), "n_days": len(scores)}


def cohort_stratified_ic(
    predictions: pd.DataFrame, axis: str,
) -> dict:
    """Per-cohort mean rank IC for axis in {sector_id, size_decile, age_bucket}.

    Computes spearman per (date, cohort) cell, averages across days
    within each cohort.
    """
    from scipy.stats import spearmanr
    out: dict[int | str, dict] = {}
    for cohort, df_c in predictions.groupby(axis):
        if df_c.empty:
            continue
        ics = []
        for date, group in df_c.groupby("date"):
            if len(group) < 5:
                continue
            yh = group["y_hat"].to_numpy()
            yt = group["y_true"].to_numpy()
            if yh.std() < 1e-9 or yt.std() < 1e-9:
                continue
            rho, _ = spearmanr(yh, yt)
            if np.isfinite(rho):
                ics.append(float(rho))
        if ics:
            out[cohort] = {
                "mean": float(np.mean(ics)),
                "std": float(np.std(ics)),
                "n_days": len(ics),
                "tickers_avg": int(df_c.groupby("date").size().mean()),
            }
    return out


def long_short_sharpe(
    predictions: pd.DataFrame, top_pct: float = 0.1, tc_bps: float = 5.0,
    annualisation: float = 252.0,
) -> dict:
    """Annualised Sharpe of a long-short portfolio.

    Forms an equal-weight long basket of the top ``top_pct`` of tickers
    by y_hat each day and a short basket of the bottom ``top_pct``;
    portfolio return is mean of (long return) minus mean of (short
    return) on the same-day y_true. Subtracts a tc_bps round-trip cost
    haircut at the daily level (treating each day as fully rebalanced).
    """
    rets = []
    for date, group in predictions.groupby("date"):
        if len(group) < 20:
            continue
        n = len(group)
        k = max(1, int(top_pct * n))
        order = np.argsort(-group["y_hat"].to_numpy())
        long_ret = float(group["y_true"].iloc[order[:k]].mean())
        short_ret = float(group["y_true"].iloc[order[-k:]].mean())
        gross = long_ret - short_ret
        net = gross - 2.0 * (tc_bps / 10000.0)
        rets.append(net)
    if not rets:
        return {"sharpe": float("nan"), "annual_return": float("nan"),
                "annual_vol": float("nan"), "n_days": 0,
                "max_drawdown": float("nan")}
    rets = np.asarray(rets)
    mu_d = float(rets.mean())
    sd_d = float(rets.std())
    sharpe = mu_d / sd_d * np.sqrt(annualisation) if sd_d > 1e-9 else float("nan")
    cum = np.cumsum(rets)
    high = np.maximum.accumulate(cum)
    drawdown = high - cum
    return {
        "sharpe": float(sharpe),
        "annual_return": float(mu_d * annualisation),
        "annual_vol": float(sd_d * np.sqrt(annualisation)),
        "max_drawdown": float(drawdown.max()) if drawdown.size else float("nan"),
        "n_days": len(rets),
    }


__all__ = [
    "daily_ic", "daily_rank_ic", "ndcg_at_k",
    "cohort_stratified_ic", "long_short_sharpe",
]
