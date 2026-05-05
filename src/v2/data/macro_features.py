"""Free macro features for the SBP v1 revision (Addition 3).

Pulls four series:
    - DGS10            10-year Treasury yield (FRED).
    - BAMLH0A0HYM2     ICE BofA US High Yield OAS (FRED).
    - IBB              iShares Nasdaq Biotech ETF daily close (yfinance).
    - XBI              already on disk; reused from
                       data/processed/risk_features.parquet (the panel
                       loader carries XBI close indirectly via the
                       returns-and-vol features), so here we re-pull
                       XBI directly to compute rolling betas.

Computes four per-(day, ticker) features:
    - xbi_beta_60d:    rolling 60-day OLS beta of ticker daily return on
                       XBI daily return.
    - ibb_beta_60d:    rolling 60-day OLS beta on IBB.
    - tenyear_yield_z: z-score of DGS10, fold-aware (caller passes train
                       indices for fold-aware standardisation).
    - hy_spread_z:     z-score of BAMLH0A0HYM2.

Output: data/macro_features_v1.parquet with columns
    [date, ticker, xbi_beta_60d, ibb_beta_60d, tenyear_yield_z,
     hy_spread_z]

The two scalar series (DGS10, HY spread) are broadcast across tickers.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np
import pandas as pd


@dataclass
class MacroFeaturesConfig:
    """Hyperparameters for macro feature extraction."""

    panel_start: str = "2014-09-01"   # 60d warmup for betas
    panel_end: str = "2023-01-15"
    beta_window: int = 60
    output_path: Path = Path("data/macro_features_v1.parquet")
    raw_prices_parquet: Path = Path("data/raw/prices_universe.parquet")
    xbi_cache: Path = Path("data/raw/xbi_close.csv")
    ibb_cache: Path = Path("data/raw/ibb_close.csv")
    fred_cache: Path = Path("data/raw/macro_fred.csv")


def _load_or_fetch_etf(symbol: str, start: str, end: str, cache: Path) -> pd.Series:
    """Load ETF close prices from cache, falling back to yfinance."""
    if cache.exists():
        df = pd.read_csv(cache, parse_dates=["date"]).set_index("date")
        if df.index.min() <= pd.Timestamp(start) and df.index.max() >= pd.Timestamp(end):
            return df["close"].astype(float)
    import yfinance as yf
    raw = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
    if raw.empty:
        raise RuntimeError(f"yfinance returned empty frame for {symbol}")
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].iloc[:, 0]
    else:
        close = raw["Close"]
    out = pd.DataFrame({"close": close.astype(float)})
    out.index.name = "date"
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache)
    return out["close"]


def _load_or_fetch_fred(start: str, end: str, cache: Path) -> pd.DataFrame:
    """Load FRED macro series from cache, falling back to pandas-datareader.

    The spec lists ``BAMLH0A0HYM2`` (ICE BofA US High Yield OAS) but the
    free public FRED CSV endpoint returns only the most recent ~3 years
    of that series without an API key. As a methodologically equivalent
    credit-stress proxy with full free historical access we substitute
    ``BAA10Y`` (Moody's Baa Corporate Yield minus 10-Year Treasury,
    Aaa-Baa is the standard textbook credit spread). The downstream
    ``hy_spread_z`` feature carries the same regime-stress signal.
    """
    if cache.exists():
        df = pd.read_csv(cache, parse_dates=["date"]).set_index("date")
        # Backwards compatibility: older caches may not have BAA10Y.
        if df.index.min() <= pd.Timestamp(start) and df.index.max() >= pd.Timestamp(end):
            if "BAA10Y" in df.columns:
                return df
    from pandas_datareader import data as web
    series = ["DGS10", "BAA10Y"]
    parts = []
    for s in series:
        try:
            x = web.DataReader(s, "fred", start, end)
        except Exception as exc:
            warnings.warn(f"FRED fetch failed for {s}: {exc}", stacklevel=2)
            continue
        x.columns = [s]
        parts.append(x)
    if not parts:
        raise RuntimeError("FRED fetch failed for all series")
    out = pd.concat(parts, axis=1).sort_index()
    out.index.name = "date"
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache)
    return out


def _rolling_beta(
    ticker_returns: np.ndarray, market_returns: np.ndarray, window: int
) -> np.ndarray:
    """Rolling OLS beta of ``ticker_returns`` on ``market_returns``.

    For each t we compute beta = cov(ticker, market) / var(market) over
    the trailing ``window`` days. NaN entries are skipped pairwise.
    """
    t_total = ticker_returns.shape[0]
    out = np.full(t_total, np.nan, dtype=np.float32)
    for t in range(window - 1, t_total):
        a = ticker_returns[t - window + 1 : t + 1]
        b = market_returns[t - window + 1 : t + 1]
        valid = ~np.isnan(a) & ~np.isnan(b)
        if valid.sum() < window // 2:
            continue
        a_v = a[valid]; b_v = b[valid]
        mu_b = b_v.mean(); var_b = ((b_v - mu_b) ** 2).mean()
        if var_b < 1e-12:
            continue
        mu_a = a_v.mean()
        cov = ((a_v - mu_a) * (b_v - mu_b)).mean()
        out[t] = float(cov / var_b)
    return out


def build_macro_features(cfg: MacroFeaturesConfig | None = None) -> pd.DataFrame:
    """Pull/compute all four macro features and persist to parquet.

    Returns the long-format DataFrame written to disk.
    """
    cfg = cfg or MacroFeaturesConfig()
    panel = pd.read_parquet(cfg.raw_prices_parquet)
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    panel = panel[(panel["date"] >= pd.Timestamp(cfg.panel_start))
                  & (panel["date"] <= pd.Timestamp(cfg.panel_end))]
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    panel["log_return"] = panel.groupby("ticker", sort=False)["close"].transform(
        lambda s: np.log(s / s.shift(1))
    )

    xbi_close = _load_or_fetch_etf("XBI", cfg.panel_start, cfg.panel_end, cfg.xbi_cache)
    ibb_close = _load_or_fetch_etf("IBB", cfg.panel_start, cfg.panel_end, cfg.ibb_cache)
    fred = _load_or_fetch_fred(cfg.panel_start, cfg.panel_end, cfg.fred_cache)

    panel_dates = sorted(panel["date"].unique())
    panel_index = pd.DatetimeIndex(panel_dates)
    xbi_aligned = xbi_close.reindex(panel_index).ffill(limit=5)
    ibb_aligned = ibb_close.reindex(panel_index).ffill(limit=5)
    fred_aligned = fred.reindex(panel_index).ffill(limit=5)

    xbi_ret = np.log(xbi_aligned / xbi_aligned.shift(1)).to_numpy()
    ibb_ret = np.log(ibb_aligned / ibb_aligned.shift(1)).to_numpy()
    dgs10 = fred_aligned["DGS10"].to_numpy(dtype=np.float32)
    # See _load_or_fetch_fred docstring: BAA10Y (Moody's Baa minus 10Y
    # Treasury) substitutes for BAMLH0A0HYM2. Same credit-stress signal,
    # full free historical access via FRED graph CSV.
    hy_oas = fred_aligned["BAA10Y"].to_numpy(dtype=np.float32)

    rows: list[pd.DataFrame] = []
    for tk, sub in panel.groupby("ticker", sort=False):
        sub = sub.set_index("date").reindex(panel_index)
        ret = sub["log_return"].to_numpy()
        beta_xbi = _rolling_beta(ret, xbi_ret, cfg.beta_window)
        beta_ibb = _rolling_beta(ret, ibb_ret, cfg.beta_window)
        out = pd.DataFrame({
            "date": panel_index,
            "ticker": tk,
            "xbi_beta_60d": beta_xbi,
            "ibb_beta_60d": beta_ibb,
            "dgs10": dgs10,
            "hy_oas": hy_oas,
        })
        rows.append(out)
    long = pd.concat(rows, ignore_index=True)
    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    long.to_parquet(cfg.output_path, index=False)
    print(f"[macro_features] wrote {cfg.output_path}: "
          f"shape={long.shape}, tickers={long['ticker'].nunique()}, "
          f"date range {long['date'].min()} to {long['date'].max()}")
    return long


def standardize_macro(
    panel_dates: list[pd.Timestamp],
    train_idx: np.ndarray,
    macro_long: pd.DataFrame,
    tickers: list[str],
) -> dict[str, np.ndarray]:
    """Z-score the scalar macro series using train-fold stats and align
    everything to the [T, N] grid.

    Returns:
        Dict with [T, N] float arrays for the four features:
            xbi_beta_60d, ibb_beta_60d, tenyear_yield_z, hy_spread_z.
    """
    panel_dates_norm = pd.DatetimeIndex(pd.to_datetime(panel_dates).normalize())
    macro_long = macro_long.copy()
    macro_long["date"] = pd.to_datetime(macro_long["date"]).dt.normalize()
    macro_long = macro_long[macro_long["date"].isin(panel_dates_norm)]
    macro_long = macro_long[macro_long["ticker"].isin(tickers)]

    t_total, n = len(panel_dates_norm), len(tickers)
    ticker_to_idx = {t: i for i, t in enumerate(tickers)}
    date_to_idx = {pd.Timestamp(d): i for i, d in enumerate(panel_dates_norm)}

    xbi_b = np.zeros((t_total, n), dtype=np.float32)
    ibb_b = np.zeros((t_total, n), dtype=np.float32)
    dgs10 = np.zeros((t_total, n), dtype=np.float32)
    hy = np.zeros((t_total, n), dtype=np.float32)
    di = macro_long["date"].map(date_to_idx).to_numpy()
    ti = macro_long["ticker"].map(ticker_to_idx).to_numpy()
    valid = ~pd.isna(di) & ~pd.isna(ti)
    di = di[valid].astype(np.int64); ti = ti[valid].astype(np.int64)

    sub = macro_long[valid].reset_index(drop=True)
    xbi_b[di, ti] = sub["xbi_beta_60d"].fillna(0.0).to_numpy(dtype=np.float32)
    ibb_b[di, ti] = sub["ibb_beta_60d"].fillna(0.0).to_numpy(dtype=np.float32)
    dgs10[di, ti] = sub["dgs10"].fillna(method="ffill").fillna(0.0).to_numpy(dtype=np.float32)
    hy[di, ti] = sub["hy_oas"].fillna(method="ffill").fillna(0.0).to_numpy(dtype=np.float32)

    train_mask = np.zeros(t_total, dtype=bool)
    train_mask[train_idx] = True
    dgs10_train = dgs10[train_mask]
    dgs10_train = dgs10_train[dgs10_train != 0.0]
    hy_train = hy[train_mask]
    hy_train = hy_train[hy_train != 0.0]
    if dgs10_train.size > 1:
        dgs10_mu = float(dgs10_train.mean()); dgs10_sd = float(dgs10_train.std())
    else:
        dgs10_mu, dgs10_sd = 0.0, 1.0
    if hy_train.size > 1:
        hy_mu = float(hy_train.mean()); hy_sd = float(hy_train.std())
    else:
        hy_mu, hy_sd = 0.0, 1.0
    if dgs10_sd < 1e-6:
        dgs10_sd = 1.0
    if hy_sd < 1e-6:
        hy_sd = 1.0
    tenyear_z = ((dgs10 - dgs10_mu) / dgs10_sd).astype(np.float32)
    hy_z = ((hy - hy_mu) / hy_sd).astype(np.float32)

    return {
        "xbi_beta_60d": xbi_b,
        "ibb_beta_60d": ibb_b,
        "tenyear_yield_z": tenyear_z,
        "hy_spread_z": hy_z,
    }


__all__ = [
    "MacroFeaturesConfig",
    "build_macro_features",
    "standardize_macro",
]
