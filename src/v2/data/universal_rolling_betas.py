"""Universal rolling-beta builder for the S&P 500 validation.

Parallel to ``src/v2/data/rolling_macro_betas.py`` with one substitution
that preserves the 10-d shape:

    Slot 1 (rolling_xbi_beta_60d)   <-  rolling_sector_etf_beta_60d
    Slot 6 (rolling_xbi_beta_120d)  <-  rolling_sector_etf_beta_120d

Where each ticker's sector ETF is determined from the GICS sector in
``sp500_constituents_history.parquet``. Mapping (per spec 3b):

    XLK  Information Technology
    XLF  Financials
    XLV  Health Care
    XLE  Energy
    XLY  Consumer Discretionary
    XLP  Consumer Staples
    XLI  Industrials
    XLU  Utilities
    XLB  Materials
    XLRE Real Estate
    XLC  Communication Services

XLRE was launched 2015-10-08, XLC was launched 2018-06-19. For dates
before the launch we fall back to SPY (broad-market beta) so the slot
remains populated rather than NaN-filled.

Output column names are kept identical to the biotech panel
(``rolling_xbi_beta_60d``, ``rolling_xbi_beta_120d``) so the existing
model code reads transparently. The values are sector-ETF betas.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.v2.data.rolling_macro_betas import (
    ROLLING_BETA_COLS, _rolling_betas_for_ticker,
)


GICS_TO_ETF = {
    "Information Technology": "XLK",
    "Financials":              "XLF",
    "Health Care":             "XLV",
    "Energy":                  "XLE",
    "Consumer Discretionary":  "XLY",
    "Consumer Staples":        "XLP",
    "Industrials":             "XLI",
    "Utilities":               "XLU",
    "Materials":               "XLB",
    "Real Estate":             "XLRE",
    "Communication Services":  "XLC",
}

ETF_LAUNCH_FALLBACK = {
    "XLRE": ("2015-10-08", "SPY"),
    "XLC":  ("2018-06-19", "SPY"),
}


@dataclass
class UniversalRollingBetaConfig:
    """Hyperparameters for universal rolling-beta estimation."""

    raw_prices_parquet: Path = Path("data/raw/sp500/prices_sp500.parquet")
    sector_etfs_parquet: Path = Path("data/raw/sp500/sector_etfs.parquet")
    constituents_parquet: Path = Path("data/raw/sp500/sp500_constituents_history.parquet")
    macro_duration_parquet: Path = Path("data/processed/macro_duration_features.parquet")
    output_path: Path = Path("data/processed/sp500_rolling_betas.parquet")
    panel_start: str = "2014-09-01"
    panel_end: str = "2023-01-15"
    window_60d: int = 60
    window_120d: int = 120
    min_obs_60d: int = 30
    min_obs_120d: int = 60
    ridge_alpha: float = 1e-3


def _build_etf_returns(cfg: UniversalRollingBetaConfig, dates: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    """Build a {etf_name -> [T] daily log return vector} mapping for the panel dates."""
    etfs = pd.read_parquet(cfg.sector_etfs_parquet)
    etfs["date"] = pd.to_datetime(etfs["date"]).dt.normalize()
    etfs = etfs.sort_values(["ticker", "date"])
    etfs["log_ret_1d"] = etfs.groupby("ticker")["close"].transform(
        lambda s: np.log(s / s.shift(1))
    )
    out: dict[str, np.ndarray] = {}
    for tk, sub in etfs.groupby("ticker"):
        s = sub.set_index("date")["log_ret_1d"].reindex(dates).ffill(limit=5)
        out[tk] = s.to_numpy(dtype=np.float32)
    return out


def _resolve_per_ticker_etf(ticker: str, sector: str, dates: pd.DatetimeIndex,
                            etf_returns: dict[str, np.ndarray]) -> np.ndarray:
    """Return [T] daily log returns of the ticker's sector ETF, with launch-
    date fallback to SPY for XLRE/XLC pre-launch dates."""
    primary = GICS_TO_ETF.get(sector, "SPY")
    series = etf_returns.get(primary)
    if series is None or primary not in ETF_LAUNCH_FALLBACK:
        return series if series is not None else etf_returns["SPY"]
    launch_str, fallback = ETF_LAUNCH_FALLBACK[primary]
    launch_ts = pd.Timestamp(launch_str).normalize()
    pre = dates < launch_ts
    out = series.copy()
    fb = etf_returns.get(fallback)
    if fb is not None and pre.any():
        out[pre] = fb[pre]
    return out


def build_universal_rolling_betas(cfg: UniversalRollingBetaConfig | None = None) -> pd.DataFrame:
    cfg = cfg or UniversalRollingBetaConfig()

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

    # Universal factor matrix: sector_etf_1d (per ticker), qqq, spy, rate_shock, credit_shock.
    qqq_1d = (macro_aligned["qqq_ret_5d"] / 5.0).to_numpy(dtype=np.float32)
    spy_1d = (macro_aligned["spy_ret_5d"] / 5.0).to_numpy(dtype=np.float32)
    rate_shock_1d   = macro_aligned["dgs10"].diff().to_numpy(dtype=np.float32)
    credit_shock_1d = macro_aligned["hy_spread"].diff().to_numpy(dtype=np.float32)

    etf_returns = _build_etf_returns(cfg, panel_dates)

    # Per-ticker GICS sector
    hist = pd.read_parquet(cfg.constituents_parquet)
    sector_map = (hist.drop_duplicates("ticker")
                       .set_index("ticker")["gics_sector"].to_dict())

    rows: list[pd.DataFrame] = []
    tickers = sorted(raw["ticker"].unique())
    for i, tk in enumerate(tickers, 1):
        sub = raw[raw["ticker"] == tk].set_index("date").reindex(panel_dates)
        ret = sub["log_return"].to_numpy(dtype=np.float32)
        sector = sector_map.get(tk)
        sector_etf_1d = _resolve_per_ticker_etf(tk, sector, panel_dates, etf_returns)
        factor_mat = np.stack(
            [sector_etf_1d, qqq_1d, spy_1d, rate_shock_1d, credit_shock_1d], axis=1,
        )
        b60, v60 = _rolling_betas_for_ticker(
            ret, factor_mat, cfg.window_60d, cfg.min_obs_60d, cfg.ridge_alpha,
        )
        b120, v120 = _rolling_betas_for_ticker(
            ret, factor_mat[:, [0, 3, 4]], cfg.window_120d, cfg.min_obs_120d, cfg.ridge_alpha,
        )
        out = pd.DataFrame({
            "date": panel_dates, "ticker": tk,
            # Same column names as biotech; the values are now sector_etf_beta
            "rolling_xbi_beta_60d":   b60[:, 0],
            "rolling_qqq_beta_60d":   b60[:, 1],
            "rolling_spy_beta_60d":   b60[:, 2],
            "rolling_rate_beta_60d":  b60[:, 3],
            "rolling_credit_beta_60d": b60[:, 4],
            "rolling_xbi_beta_120d":  b120[:, 0],
            "rolling_rate_beta_120d": b120[:, 1],
            "rolling_credit_beta_120d": b120[:, 2],
            "beta_valid_ratio_60d":   v60,
            "beta_valid_ratio_120d":  v120,
        })
        rows.append(out)
        if i % 100 == 0:
            print(f"  [{i}/{len(tickers)}] tickers processed", flush=True)

    long = pd.concat(rows, ignore_index=True)
    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    long.to_parquet(cfg.output_path, index=False)
    print(f"[universal_rolling_betas] wrote {cfg.output_path}: "
          f"shape={long.shape}, tickers={long['ticker'].nunique()}",
          flush=True)
    return long


__all__ = [
    "UniversalRollingBetaConfig",
    "GICS_TO_ETF",
    "ETF_LAUNCH_FALLBACK",
    "build_universal_rolling_betas",
]
