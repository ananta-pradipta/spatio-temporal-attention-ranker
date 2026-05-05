"""Evaluation metrics for stock ranking models.

Provides IC, Rank IC, and long-short portfolio return computation.
"""

from typing import Dict, Optional

import numpy as np
from scipy import stats


def information_coefficient(
    predictions: np.ndarray,
    targets: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> float:
    """Compute Information Coefficient (Pearson correlation).

    Args:
        predictions: Predicted scores.
        targets: Actual forward returns or ranks.
        mask: Optional boolean mask for valid entries.

    Returns:
        Pearson correlation coefficient. Returns 0.0 if computation fails.
    """
    if mask is not None:
        predictions = predictions[mask]
        targets = targets[mask]

    valid = ~(np.isnan(predictions) | np.isnan(targets))
    if valid.sum() < 3:
        return 0.0

    corr, _ = stats.pearsonr(predictions[valid], targets[valid])
    if np.isnan(corr):
        return 0.0
    return float(corr)


def rank_information_coefficient(
    predictions: np.ndarray,
    targets: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> float:
    """Compute Rank IC (Spearman rank correlation).

    Args:
        predictions: Predicted scores.
        targets: Actual forward returns or ranks.
        mask: Optional boolean mask for valid entries.

    Returns:
        Spearman correlation coefficient. Returns 0.0 if computation fails.
    """
    if mask is not None:
        predictions = predictions[mask]
        targets = targets[mask]

    valid = ~(np.isnan(predictions) | np.isnan(targets))
    if valid.sum() < 3:
        return 0.0

    corr, _ = stats.spearmanr(predictions[valid], targets[valid])
    if np.isnan(corr):
        return 0.0
    return float(corr)


def long_short_return(
    predictions: np.ndarray,
    forward_returns: np.ndarray,
    top_k: int = 5,
    bottom_k: int = 5,
    mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute long-short portfolio return for a single snapshot.

    Longs the top_k predicted stocks, shorts the bottom_k.

    Args:
        predictions: Predicted scores.
        forward_returns: Actual next-period returns.
        top_k: Number of stocks in the long portfolio.
        bottom_k: Number of stocks in the short portfolio.
        mask: Optional boolean mask.

    Returns:
        Dictionary with long_return, short_return, long_short_return.
    """
    if mask is not None:
        predictions = predictions[mask]
        forward_returns = forward_returns[mask]

    valid = ~(np.isnan(predictions) | np.isnan(forward_returns))
    pred_valid = predictions[valid]
    ret_valid = forward_returns[valid]

    n = len(pred_valid)
    if n < top_k + bottom_k:
        return {"long_return": 0.0, "short_return": 0.0, "long_short_return": 0.0}

    sorted_idx = np.argsort(pred_valid)

    # Bottom k (short) and top k (long)
    short_idx = sorted_idx[:bottom_k]
    long_idx = sorted_idx[-top_k:]

    long_ret = float(np.mean(ret_valid[long_idx]))
    short_ret = float(np.mean(ret_valid[short_idx]))
    ls_ret = long_ret - short_ret

    return {
        "long_return": long_ret,
        "short_return": short_ret,
        "long_short_return": ls_ret,
    }


def compute_all_metrics(
    predictions: np.ndarray,
    target_ranks: np.ndarray,
    forward_returns: np.ndarray,
    top_k: int = 5,
    bottom_k: int = 5,
) -> Dict[str, float]:
    """Compute all evaluation metrics for a single snapshot.

    Args:
        predictions: Model output scores.
        target_ranks: Normalized rank targets.
        forward_returns: Raw forward returns.
        top_k: Stocks in long portfolio.
        bottom_k: Stocks in short portfolio.

    Returns:
        Dictionary of metric name to value.
    """
    mask = ~(np.isnan(target_ranks) | np.isnan(forward_returns))

    ic = information_coefficient(predictions, forward_returns, mask)
    ric = rank_information_coefficient(predictions, forward_returns, mask)
    ls = long_short_return(predictions, forward_returns, top_k, bottom_k, mask)

    return {
        "ic": ic,
        "rank_ic": ric,
        **ls,
    }
