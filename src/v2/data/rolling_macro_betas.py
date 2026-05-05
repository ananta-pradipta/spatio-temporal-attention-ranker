"""Rolling macro-beta features per (day, ticker) for DOW-epiSTAR v2.

Per the spec Section B, estimate trailing-window sensitivities of each
ticker's daily log return to:

    xbi_return       sector beta
    qqq_return       broad-Nasdaq beta
    spy_return       broad-market beta
    rate_shock       daily change in 10y yield
    credit_shock     daily change in HY (BAA10Y proxy) spread

Two windows: 60d and 120d. Estimation is multivariate ridge OLS:

    r_i[tau] = b_xbi * xbi[tau] + b_qqq * qqq[tau] + b_spy * spy[tau]
              + b_rate * rate_shock[tau] + b_credit * credit_shock[tau]
              + epsilon_i[tau]

with ridge_alpha = 1e-3 and min_obs = 30 (60d) / 60 (120d). Tickers
that fail the min_obs threshold get all betas = 0 and
beta_valid_ratio = 0.

Output: data/processed/rolling_macro_betas.parquet
    [date, ticker,
     rolling_xbi_beta_60d, rolling_qqq_beta_60d, rolling_spy_beta_60d,
     rolling_rate_beta_60d, rolling_credit_beta_60d,
     rolling_xbi_beta_120d, rolling_rate_beta_120d, rolling_credit_beta_120d,
     beta_valid_ratio_60d, beta_valid_ratio_120d]
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class RollingBetaConfig:
    """Hyperparameters for rolling beta estimation."""

    raw_prices_parquet: Path = Path("data/raw/prices_universe.parquet")
    macro_duration_parquet: Path = Path("data/processed/macro_duration_features.parquet")
    output_path: Path = Path("data/processed/rolling_macro_betas.parquet")
    panel_start: str = "2014-09-01"   # 120d warmup before 2015-01-09
    panel_end: str = "2023-01-15"
    window_60d: int = 60
    window_120d: int = 120
    min_obs_60d: int = 30
    min_obs_120d: int = 60
    ridge_alpha: float = 1e-3


def _ridge_solve(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Solve ridge OLS: beta = (X.T X + alpha I)^-1 X.T y.

    Returns NaN-filled vector if the system is singular or ``y`` is
    constant.
    """
    if x.shape[0] < x.shape[1] + 1 or y.std() < 1e-9:
        return np.full(x.shape[1], np.nan, dtype=np.float32)
    xtx = x.T @ x + alpha * np.eye(x.shape[1])
    try:
        beta = np.linalg.solve(xtx, x.T @ y)
    except np.linalg.LinAlgError:
        return np.full(x.shape[1], np.nan, dtype=np.float32)
    return beta.astype(np.float32)


def _rolling_betas_for_ticker(
    ticker_ret: np.ndarray,
    factor_mat: np.ndarray,
    window: int,
    min_obs: int,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute rolling betas of ticker_ret on factor_mat.

    Args:
        ticker_ret: [T] daily log returns. NaN for non-tradable days.
        factor_mat: [T, F] daily factor returns/shocks.
        window: trailing window length.
        min_obs: minimum non-NaN overlap to estimate.
        alpha: ridge regularisation.

    Returns:
        betas: [T, F] rolling beta matrix.
        valid: [T] fraction of non-NaN obs in trailing window.
    """
    t_total, f = factor_mat.shape
    betas = np.full((t_total, f), 0.0, dtype=np.float32)
    valid = np.zeros(t_total, dtype=np.float32)
    for t in range(window - 1, t_total):
        lo = t - window + 1
        y_win = ticker_ret[lo : t + 1]
        x_win = factor_mat[lo : t + 1]
        ok = ~np.isnan(y_win) & ~np.isnan(x_win).any(axis=1)
        n_obs = int(ok.sum())
        valid[t] = n_obs / float(window)
        if n_obs < min_obs:
            continue
        b = _ridge_solve(x_win[ok], y_win[ok], alpha)
        if not np.isnan(b).any():
            betas[t] = b
    return betas, valid


def build_rolling_betas(cfg: RollingBetaConfig | None = None) -> pd.DataFrame:
    """Build the per-(day, ticker) rolling beta panel.

    Reuses the macro_duration_features parquet as the factor source
    (xbi_ret_1d, qqq_ret_5d-derived 1d, spy 1d, rate_shock_1d derived
    from dgs10 diff, credit_shock_1d derived from hy_spread diff).
    """
    cfg = cfg or RollingBetaConfig()

    raw = pd.read_parquet(cfg.raw_prices_parquet).copy()
    raw["date"] = pd.to_datetime(raw["date"]).dt.normalize()
    raw = raw[(raw["date"] >= pd.Timestamp(cfg.panel_start))
              & (raw["date"] <= pd.Timestamp(cfg.panel_end))]
    raw = raw.sort_values(["ticker", "date"]).reset_index(drop=True)
    raw["log_return"] = raw.groupby("ticker", sort=False)["close"].transform(
        lambda s: np.log(s / s.shift(1))
    )

    macro = pd.read_parquet(cfg.macro_duration_parquet)
    macro.index = pd.to_datetime(macro.index).normalize()
    panel_dates = pd.DatetimeIndex(sorted(set(raw["date"]).union(set(macro.index))))
    panel_dates = panel_dates[
        (panel_dates >= pd.Timestamp(cfg.panel_start))
        & (panel_dates <= pd.Timestamp(cfg.panel_end))
    ]

    macro_aligned = macro.reindex(panel_dates).ffill(limit=5)
    # 1d factor returns/shocks (5 columns).
    xbi_1d = macro_aligned["xbi_ret_1d"].to_numpy(dtype=np.float32)
    # qqq, spy 1d returns derived from 5d (rough proxy).
    qqq_close = (macro_aligned["qqq_ret_5d"] / 5.0).to_numpy(dtype=np.float32)
    spy_close = (macro_aligned["spy_ret_5d"] / 5.0).to_numpy(dtype=np.float32)
    rate_shock_1d = macro_aligned["dgs10"].diff().to_numpy(dtype=np.float32)
    credit_shock_1d = macro_aligned["hy_spread"].diff().to_numpy(dtype=np.float32)
    factor_mat = np.stack([xbi_1d, qqq_close, spy_close, rate_shock_1d, credit_shock_1d], axis=1)

    rows: list[pd.DataFrame] = []
    tickers = sorted(raw["ticker"].unique())
    for tk in tickers:
        sub = raw[raw["ticker"] == tk].set_index("date").reindex(panel_dates)
        ret = sub["log_return"].to_numpy(dtype=np.float32)
        b60, v60 = _rolling_betas_for_ticker(
            ret, factor_mat, cfg.window_60d, cfg.min_obs_60d, cfg.ridge_alpha,
        )
        b120, v120 = _rolling_betas_for_ticker(
            ret, factor_mat[:, [0, 3, 4]], cfg.window_120d, cfg.min_obs_120d, cfg.ridge_alpha,
        )
        out = pd.DataFrame({
            "date": panel_dates, "ticker": tk,
            "rolling_xbi_beta_60d": b60[:, 0],
            "rolling_qqq_beta_60d": b60[:, 1],
            "rolling_spy_beta_60d": b60[:, 2],
            "rolling_rate_beta_60d": b60[:, 3],
            "rolling_credit_beta_60d": b60[:, 4],
            "rolling_xbi_beta_120d": b120[:, 0],
            "rolling_rate_beta_120d": b120[:, 1],
            "rolling_credit_beta_120d": b120[:, 2],
            "beta_valid_ratio_60d": v60,
            "beta_valid_ratio_120d": v120,
        })
        rows.append(out)
    long = pd.concat(rows, ignore_index=True)
    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    long.to_parquet(cfg.output_path, index=False)
    print(f"[rolling_betas] wrote {cfg.output_path}: shape={long.shape}, "
          f"tickers={long['ticker'].nunique()}")
    return long


ROLLING_BETA_COLS = [
    "rolling_xbi_beta_60d", "rolling_qqq_beta_60d", "rolling_spy_beta_60d",
    "rolling_rate_beta_60d", "rolling_credit_beta_60d",
    "rolling_xbi_beta_120d", "rolling_rate_beta_120d", "rolling_credit_beta_120d",
    "beta_valid_ratio_60d", "beta_valid_ratio_120d",
]


def betas_to_tensor(
    long: pd.DataFrame, panel_dates: list[pd.Timestamp], tickers: list[str],
) -> np.ndarray:
    """Pivot the long-format betas to a [T, N, 10] tensor aligned with
    the panel."""
    panel_index = pd.DatetimeIndex(pd.to_datetime(panel_dates).normalize())
    ticker_to_idx = {t: i for i, t in enumerate(tickers)}
    date_to_idx = {pd.Timestamp(d): i for i, d in enumerate(panel_index)}
    t_total, n = len(panel_index), len(tickers)
    out = np.zeros((t_total, n, len(ROLLING_BETA_COLS)), dtype=np.float32)

    sub = long[long["ticker"].isin(tickers)].copy()
    sub["date"] = pd.to_datetime(sub["date"]).dt.normalize()
    sub = sub[sub["date"].isin(panel_index)]
    di = sub["date"].map(date_to_idx).to_numpy()
    ti = sub["ticker"].map(ticker_to_idx).to_numpy()
    valid = ~pd.isna(di) & ~pd.isna(ti)
    di = di[valid].astype(np.int64); ti = ti[valid].astype(np.int64)
    sub_v = sub[valid].reset_index(drop=True)
    for j, col in enumerate(ROLLING_BETA_COLS):
        out[di, ti, j] = sub_v[col].fillna(0.0).to_numpy(dtype=np.float32)
    return out


__all__ = [
    "RollingBetaConfig",
    "ROLLING_BETA_COLS",
    "build_rolling_betas",
    "betas_to_tensor",
]
