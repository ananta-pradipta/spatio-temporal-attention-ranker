"""LATTICE Phase 1 macro state (24-d).

Per spec Section 5.3, the 24 columns are:

    vix, vix_term_slope, move_proxy
    dgs2, dgs10, slope_2s10s, slope_3m10y, breakeven_10y
    dxy_5d_ret
    hyg_5d_ret, tlt_5d_ret, gld_5d_ret
    spy_5d_ret, qqq_5d_ret, iwm_5d_ret
    xlk_5d_ret, xlf_5d_ret, xle_5d_ret, xlv_5d_ret
    xly_5d_ret, xlp_5d_ret, xlu_5d_ret, xlre_5d_ret
    market_breadth_proxy

Sources:
    FRED                        DGS2, DGS10, BAA10Y (already in v2 macro CSV)
                                T10YIE, DTWEXBGS    (in macro_fred_extra.csv)
    risk_features.parquet       VIX, VVIX, vix_term_slope (from v2)
    macro_etfs.parquet          11 sector ETFs + SPY + QQQ
    macro_etfs_extra.parquet    IWM, HYG, TLT, GLD
    market breadth              fraction of S&P 500 above 50-day MA
                                (computed from prices_sp500.parquet)

Output: data/lattice/processed/macro_state.parquet
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _load_etfs(*paths: Path) -> dict[str, pd.Series]:
    """Concatenate all ETF parquets into a per-ticker close-price Series dict."""
    frames = []
    for p in paths:
        if p.exists():
            df = pd.read_parquet(p)
            df["date"] = pd.to_datetime(df["date"]).dt.normalize()
            frames.append(df)
    full = pd.concat(frames, ignore_index=True)
    out = {}
    for tk, sub in full.groupby("ticker"):
        sub = sub.sort_values("date")
        out[tk] = sub.set_index("date")["close"]
    return out


def build_macro_state(
    fred_main_csv: Path = Path("data/lattice/raw/macro_fred.csv"),
    fred_extra_csv: Path = Path("data/lattice/raw/macro_fred_extra.csv"),
    risk_features_parquet: Path = Path("data/processed/risk_features.parquet"),
    etf_main_parquet: Path = Path("data/lattice/raw/macro_etfs.parquet"),
    etf_extra_parquet: Path = Path("data/lattice/raw/macro_etfs_extra.parquet"),
    prices_parquet: Path = Path("data/lattice/raw/prices_sp500.parquet"),
    out_path: Path = Path("data/lattice/processed/macro_state.parquet"),
    panel_start: str = "2015-01-09",
    panel_end: str = "2022-12-31",
) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fred_main = pd.read_csv(fred_main_csv, parse_dates=["date"]).set_index("date")
    fred_extra = pd.read_csv(fred_extra_csv, parse_dates=["date"]).set_index("date")
    fred = fred_main.join(fred_extra, how="outer").sort_index()

    risk = pd.read_parquet(risk_features_parquet)
    risk.index = pd.to_datetime(risk.index)

    etfs = _load_etfs(etf_main_parquet, etf_extra_parquet)
    print(f"[lattice macro] etfs loaded: {sorted(etfs.keys())}", flush=True)

    # Common date index from any ETF (they share the same trading calendar)
    sample_etf = etfs["SPY"]
    dates = pd.DatetimeIndex(sample_etf.index)
    dates = dates[(dates >= panel_start) & (dates <= panel_end)]

    fred_a = fred.reindex(dates).ffill(limit=5)
    risk_a = risk.reindex(dates).ffill(limit=5)

    out = pd.DataFrame(index=dates)
    out.index.name = "date"

    # Vol
    out["vix"] = risk_a["vix"]
    if "vix_term_slope" in risk_a.columns:
        out["vix_term_slope"] = risk_a["vix_term_slope"]
    else:
        out["vix_term_slope"] = 0.0
    out["move_proxy"] = fred_a["DGS10"].diff().rolling(20, min_periods=5).std() * np.sqrt(252)

    # Yields
    out["dgs2"] = fred_a["DGS2"]
    out["dgs10"] = fred_a["DGS10"]
    out["slope_2s10s"] = fred_a["DGS10"] - fred_a["DGS2"]
    out["slope_3m10y"] = fred_a["DGS10"] - fred_a["DGS3MO"]
    out["breakeven_10y"] = fred_a.get("T10YIE", pd.Series(np.nan, index=dates))

    # USD index (5d return)
    dxy = fred_a.get("DTWEXBGS", pd.Series(np.nan, index=dates))
    out["dxy_5d_ret"] = np.log(dxy / dxy.shift(5))

    # ETF 5-day log returns
    def _ret_5d(tk):
        s = etfs.get(tk)
        if s is None:
            return pd.Series(np.nan, index=dates)
        s_a = s.reindex(dates).ffill(limit=5)
        return np.log(s_a / s_a.shift(5))

    for tk, col in [
        ("HYG", "hyg_5d_ret"), ("TLT", "tlt_5d_ret"), ("GLD", "gld_5d_ret"),
        ("SPY", "spy_5d_ret"), ("QQQ", "qqq_5d_ret"), ("IWM", "iwm_5d_ret"),
        ("XLK", "xlk_5d_ret"), ("XLF", "xlf_5d_ret"),
        ("XLE", "xle_5d_ret"), ("XLV", "xlv_5d_ret"), ("XLY", "xly_5d_ret"),
        ("XLP", "xlp_5d_ret"), ("XLU", "xlu_5d_ret"), ("XLRE", "xlre_5d_ret"),
    ]:
        out[col] = _ret_5d(tk)

    # Market breadth: fraction of S&P 500 active tickers with close above their 50-day MA
    prices = pd.read_parquet(prices_parquet, columns=["ticker", "date", "close"])
    prices["date"] = pd.to_datetime(prices["date"]).dt.normalize()
    prices = prices[(prices["date"] >= panel_start) & (prices["date"] <= panel_end)]
    prices = prices.sort_values(["ticker", "date"])
    prices["ma50"] = prices.groupby("ticker", sort=False)["close"].transform(
        lambda x: x.rolling(50, min_periods=10).mean()
    )
    prices["above_ma"] = (prices["close"] > prices["ma50"]).astype(float)
    breadth = prices.groupby("date")["above_ma"].mean()
    out["market_breadth_proxy"] = breadth.reindex(dates).ffill(limit=5)

    out = out.reset_index()
    out.to_parquet(out_path, index=False)
    print(f"[lattice macro] wrote {out_path}: shape={out.shape}", flush=True)
    print(f"  date range: {out.date.min()} -> {out.date.max()}", flush=True)
    print(f"  null rate per col:")
    for c in out.columns[1:]:
        print(f"    {c:25s}  {out[c].isna().mean()*100:5.1f}%", flush=True)
    return {
        "rows": len(out),
        "cols": list(out.columns),
        "dates": (out.date.min().strftime("%Y-%m-%d"), out.date.max().strftime("%Y-%m-%d")),
    }


__all__ = ["build_macro_state"]
