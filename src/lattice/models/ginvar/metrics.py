"""G-InVAR evaluation metrics.

Reuses the InVAR metrics module (daily IC, rank IC, NDCG@k, cohort
stratified IC, long-short Sharpe). Re-exported here so users of the
``ginvar`` package have a single import surface.
"""
from src.invar.evaluation.metrics import (
    daily_ic, daily_rank_ic, ndcg_at_k,
    cohort_stratified_ic, long_short_sharpe,
)

__all__ = [
    "daily_ic", "daily_rank_ic", "ndcg_at_k",
    "cohort_stratified_ic", "long_short_sharpe",
]
