"""Risk-aware global features (per-day, shared across all stocks).

Seven features used by STAR's Feature-wise Linear Modulation (FiLM)
conditioning and optionally by MARS as auxiliary global context:

  1. CBOE Volatility Index (VIX)
  2. NASDAQ 100 Volatility Index (VXN)
  3. Volatility of VIX (VVIX)
  4. VIX term slope = VIX3M - VIX  (positive = calm/contango)
  5. SPDR Biotech ETF (XBI) realized volatility 20d (annualized)
  6. XBI realized volatility 60d (annualized)
  7. VIX 5-day change

Already loaded: VIX, VXN, VVIX in `data/raw/volatility_indices.parquet`.
To fetch: VIX9D (^VIX9D), VIX3M (^VIX3M), XBI (XBI) daily closes.

Usage:
    from src.mtgn.data.risk_features import build_risk_features, standardize_risk_features
    df = build_risk_features("2015-01-01", "2023-01-01")
    df.to_parquet("data/processed/risk_features.parquet")

Standardization must use train-slice statistics only (caller responsibility).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RISK_COLS = (
    "vix", "vxn", "vvix", "vix_term_slope",
    "xbi_rv_20d", "xbi_rv_60d", "vix_5d_change",
)


def build_risk_features(start_date: str, end_date: str,
                        cache_path: Path = Path("data/processed/risk_features.parquet")) -> pd.DataFrame:
    """Returns a DataFrame indexed by trading day with the 7 risk columns.

    Uses existing `data/raw/volatility_indices.parquet` for VIX/VXN/VVIX,
    fetches VIX9D/VIX3M/XBI from Yahoo Finance.
    """
    if cache_path.exists():
        cached = pd.read_parquet(cache_path)
        cached.index = pd.to_datetime(cached.index)
        # Tolerance of 7 calendar days at the start (first trading day may be
        # later than the requested calendar start) and at the end; accept the
        # cache if it covers the business-day range of the request.
        start_ok = cached.index.min() <= pd.Timestamp(start_date) + pd.Timedelta(days=7)
        end_ok   = cached.index.max() >= pd.Timestamp(end_date)   - pd.Timedelta(days=7)
        if start_ok and end_ok:
            return cached.loc[start_date:end_date]

    import yfinance as yf
    vol = pd.read_parquet("data/raw/volatility_indices.parquet")
    vol.index = pd.to_datetime(vol.index)
    vol = vol.loc[start_date:end_date]

    extra = {}
    for sym, name in [("^VIX9D", "vix9d"), ("^VIX3M", "vix3m"), ("XBI", "xbi")]:
        raw = yf.download(sym, start=start_date, end=end_date,
                          progress=False, auto_adjust=False)
        close = raw["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        extra[name] = close

    df = pd.DataFrame({
        "vix": vol["VIX"],
        "vxn": vol["VXN"],
        "vvix": vol["VVIX"],
    }).join(pd.concat(extra, axis=1), how="left")

    df["vix_term_slope"] = df["vix3m"] - df["vix"]
    xbi_log_ret = np.log(df["xbi"] / df["xbi"].shift(1))
    df["xbi_rv_20d"] = xbi_log_ret.rolling(20, min_periods=5).std() * np.sqrt(252)
    df["xbi_rv_60d"] = xbi_log_ret.rolling(60, min_periods=10).std() * np.sqrt(252)
    df["vix_5d_change"] = df["vix"] - df["vix"].shift(5)

    # Causal forward-looking auxiliary supervision target for STAR.
    # At day t, mean(|xbi_log_ret[t+1..t+5]|). Uses ONLY future returns
    # relative to day t, so no overlap with features available at t.
    abs_ret = xbi_log_ret.abs()
    fwd5 = sum(abs_ret.shift(-k) for k in range(1, 6)) / 5.0
    df["xbi_fwd_abs_ret_5d"] = fwd5

    out_cols = list(RISK_COLS) + ["xbi_fwd_abs_ret_5d"]
    out = df[out_cols].copy()
    out = out.ffill().dropna(how="all")
    out.index.name = "date"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache_path)
    return out


def standardize_risk_features(df: pd.DataFrame, train_start: str,
                              train_end: str) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Standardize using train-slice mean and std. Returns (df_std, mu, sd)."""
    train = df.loc[train_start:train_end]
    mu = train.mean()
    sd = train.std().replace(0, 1.0)
    return (df - mu) / (sd + 1e-6), mu, sd


__all__ = ["build_risk_features", "standardize_risk_features", "RISK_COLS"]
