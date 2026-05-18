"""InVAR Phase 3 statistical tests.

  - paired_t_test_per_fold(model_ic_per_seed, baseline_ic_per_seed):
        4 dof, two-sided.
  - diebold_mariano(model_daily_ic, baseline_daily_ic):
        DM test on daily-IC differences.
  - wilcoxon_signed_rank(model_per_seed, baseline_per_seed):
        non-parametric backup.
  - bootstrap_sharpe_ci(daily_returns, n_resamples=1000, alpha=0.05):
        95% CI on long-short Sharpe.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TestResult:
    statistic: float
    pvalue: float
    method: str
    note: str = ""


def paired_t_test_per_fold(
    model_per_seed: list[float], baseline_per_seed: list[float],
) -> TestResult:
    """Paired t-test across 5 seeds within a fold (4 degrees of freedom)."""
    from scipy import stats
    a = np.asarray(model_per_seed, dtype=float)
    b = np.asarray(baseline_per_seed, dtype=float)
    if a.shape != b.shape or a.size < 2:
        return TestResult(float("nan"), float("nan"), "paired_t",
                            "insufficient data")
    res = stats.ttest_rel(a, b, alternative="two-sided")
    return TestResult(float(res.statistic), float(res.pvalue), "paired_t")


def wilcoxon_signed_rank(
    model_per_seed: list[float], baseline_per_seed: list[float],
) -> TestResult:
    """Wilcoxon signed-rank, non-parametric backup."""
    from scipy import stats
    a = np.asarray(model_per_seed, dtype=float)
    b = np.asarray(baseline_per_seed, dtype=float)
    if a.shape != b.shape or a.size < 2:
        return TestResult(float("nan"), float("nan"), "wilcoxon",
                            "insufficient data")
    diffs = a - b
    if (np.abs(diffs) < 1e-12).all():
        return TestResult(float("nan"), float("nan"), "wilcoxon",
                            "all differences zero")
    res = stats.wilcoxon(diffs, alternative="two-sided")
    return TestResult(float(res.statistic), float(res.pvalue), "wilcoxon")


def diebold_mariano(
    model_daily_ic: np.ndarray, baseline_daily_ic: np.ndarray,
    h: int = 1,
) -> TestResult:
    """Diebold-Mariano test on daily-IC differences.

    The "loss" is one minus IC; squared-loss differential is taken pointwise.
    Reference: Diebold and Mariano (1995).
    """
    a = np.asarray(model_daily_ic, dtype=float)
    b = np.asarray(baseline_daily_ic, dtype=float)
    if a.shape != b.shape or a.size < 5:
        return TestResult(float("nan"), float("nan"), "DM", "insufficient days")
    d = (1.0 - a) ** 2 - (1.0 - b) ** 2
    n = d.size
    mean_d = float(d.mean())
    var_d = float(d.var(ddof=1))
    if var_d < 1e-12:
        return TestResult(float("nan"), float("nan"), "DM", "zero variance")
    dm_stat = mean_d / np.sqrt(var_d / n)
    # Two-sided p-value under N(0, 1)
    from scipy.stats import norm
    pvalue = 2.0 * (1.0 - norm.cdf(abs(dm_stat)))
    return TestResult(float(dm_stat), float(pvalue), "DM",
                        f"n={n}, h={h}")


def bootstrap_sharpe_ci(
    daily_returns: np.ndarray, n_resamples: int = 1000, alpha: float = 0.05,
    annualisation: float = 252.0, seed: int = 0,
) -> dict:
    """Bootstrap 95% CI on the annualised Sharpe of a daily-return series.

    Args:
        daily_returns: 1-D array of long-short daily returns.
        n_resamples: number of bootstrap resamples (1000 default per spec).
    """
    r = np.asarray(daily_returns, dtype=float)
    if r.size < 10:
        return {
            "point": float("nan"),
            "ci_lo": float("nan"),
            "ci_hi": float("nan"),
            "n_resamples": 0,
        }
    rng = np.random.default_rng(seed)
    sharpes = []
    n = r.size
    for _ in range(n_resamples):
        idx = rng.integers(low=0, high=n, size=n)
        sample = r[idx]
        sd = sample.std()
        if sd > 1e-9:
            sharpes.append(float(sample.mean() / sd * np.sqrt(annualisation)))
    if not sharpes:
        return {"point": float("nan"), "ci_lo": float("nan"),
                "ci_hi": float("nan"), "n_resamples": 0}
    sd_full = r.std()
    point = float(r.mean() / sd_full * np.sqrt(annualisation)) if sd_full > 1e-9 else float("nan")
    sharpes.sort()
    lo = sharpes[int(np.floor(alpha / 2.0 * len(sharpes)))]
    hi = sharpes[int(np.ceil((1 - alpha / 2.0) * len(sharpes))) - 1]
    return {
        "point": point,
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "n_resamples": len(sharpes),
        "alpha": alpha,
    }


__all__ = [
    "TestResult",
    "paired_t_test_per_fold",
    "wilcoxon_signed_rank",
    "diebold_mariano",
    "bootstrap_sharpe_ci",
]
