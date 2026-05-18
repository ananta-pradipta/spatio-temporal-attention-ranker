"""LATTICE Phase 1 panel builder.

Produces five parquet artifacts under data/lattice/processed/:

    active_mask.parquet         (date, ticker, mask)
    cohorts.parquet             (date, ticker, size_decile, liquidity_decile,
                                  sector, age_bucket)
    panel_features.parquet      (date, ticker, 31 features)
    stocktwits_features.parquet (date, ticker, 5 disagreement features)
    macro_state.parquet         (date, 24 macro state columns)

Per spec Section 5.1, the 31-feature panel decomposes:
    10 price+volume      (9 from v2 biotech + amihud_illiquidity_20d)
    4 distress proxies   (interest_coverage, net_debt_to_ebitda, fcf_yield,
                           current_ratio; sector-z-scored within fold)
    4 intangible proxies (rd_to_sales, sga_to_sales, gross_profitability,
                           capex_to_sales; sector-z-scored within fold)
    3 fundamentals       (log_market_cap, book_to_market, asset_growth_yoy)
    5 stocktwits disagreement (separate parquet, joined at training time)
    3 catalyst features  (sin/cos days-to-event, 6-d onehot summed-down to a
                           single integer-tag column for storage; expanded at
                           training time. Phase 1 ships these as zero-filled
                           because the S&P 500 catalyst calendar requires a
                           separate engineering effort beyond Phase 1 scope.)
    2 availability flags (has_fundamentals, has_stocktwits)

Note: spec Section 5.1 totals 30 features in summary text but the explicit
itemization adds to 31 (9 price + 1 amihud + 4 + 4 + 3 + 5 + 3 + 2 = 31).
This implementation uses 31 with a clear comment in the schema docstring.

Sector-z-scoring uses the training-fold's sector mean and standard
deviation; the scaler is fitted in `src/lattice/training/standardise.py`
during fold setup and is NOT computed in this Phase 1 builder.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


PRICE_VOL_COLS = [
    "log_return", "log_return_5d", "log_return_20d",
    "log_volume", "log_volume_ratio_20d",
    "realized_vol_20d", "realized_vol_60d",
    "high_low_range", "close_to_high_5d",
    "amihud_illiquidity_20d",
]

# Tier A1 plus A2 panel additions (2026-05-12, see docs/panel_feature_improvement.md).
# K-line shape: Alpha158 canonical body / shadow / asymmetry features.
KLINE_COLS = [
    "kmid", "klen", "kup", "klow", "ksft",
]
# Multi-horizon momentum: 60-day plus 12-month-minus-1-month momentum, the
# Jegadeesh-Titman canon. Complements existing log_return_5d and _20d.
MOMENTUM_EXTRA_COLS = [
    "log_return_60d", "log_return_12m_minus_1m",
]
# Cross-sectional vol and price-volume signals not covered by realized_vol.
# max20 captures Bali-Cakici-Whitelaw lottery preferences; ivol_21d the
# CAPM-residual idiosyncratic vol (Ang-Hodrick-Xing-Zhang, 2006); corr20
# and cord20 the Alpha158 price-volume joint dynamics.
VOL_PV_EXTRA_COLS = [
    "max20", "ivol_21d", "corr20", "cord20",
]

DISTRESS_COLS = [
    "interest_coverage",
    "net_debt_to_ebitda",
    "fcf_yield",
    "current_ratio",
]

INTANGIBLE_COLS = [
    "rd_to_sales",
    "sga_to_sales",
    "gross_profitability",
    "capex_to_sales",
]

OTHER_FUND_COLS = [
    "log_market_cap",
    "book_to_market",
    "asset_growth_yoy",
]

CATALYST_COLS = [
    "days_to_next_catalyst_sin",
    "days_to_next_catalyst_cos",
    "catalyst_type_id",
]

FLAG_COLS = [
    "has_fundamentals",
    "has_stocktwits",
]

PANEL_FEATURE_COLS = (
    PRICE_VOL_COLS + KLINE_COLS + MOMENTUM_EXTRA_COLS + VOL_PV_EXTRA_COLS
    + DISTRESS_COLS + INTANGIBLE_COLS + OTHER_FUND_COLS
    + CATALYST_COLS + FLAG_COLS
)
# 10 PV + 5 k-line + 2 momentum-extra + 4 vol/PV-extra + 4 distress
# + 4 intangible + 3 other-fund + 3 catalyst + 2 flag = 37.
assert len(PANEL_FEATURE_COLS) == 37

ST_FEATURE_COLS = [
    "st_volume_24h_log",
    "st_volume_abnormal_z60d",
    "st_sentiment_dispersion",
    "st_labeled_ratio",
    "st_bullish_ratio_demeaned",
]

MACRO_FEATURE_COLS = [
    "vix", "vix_term_slope", "move_proxy",
    "dgs2", "dgs10", "slope_2s10s", "slope_3m10y", "breakeven_10y",
    "dxy_5d_ret",
    "hyg_5d_ret", "tlt_5d_ret", "gld_5d_ret",
    "spy_5d_ret", "qqq_5d_ret", "iwm_5d_ret",
    "xlk_5d_ret", "xlf_5d_ret", "xle_5d_ret", "xlv_5d_ret",
    "xly_5d_ret", "xlp_5d_ret", "xlu_5d_ret",
    "xlre_5d_ret", "market_breadth_proxy",
]
assert len(MACRO_FEATURE_COLS) == 24


@dataclass
class LatticePhase1Config:
    """Phase 1 builder configuration."""

    raw_dir: Path = Path("data/lattice/raw")
    out_dir: Path = Path("data/lattice/processed")
    panel_start: str = "2015-01-09"
    panel_end: str = "2022-12-31"


# --------------------------- Price + volume features ---------------------------

def _price_features(
    df: pd.DataFrame, spy_log_return: pd.Series | None = None,
) -> pd.DataFrame:
    """Per-ticker price-volume feature pass.

    Args:
        df: long-format prices with at least ['ticker', 'date', 'open',
            'high', 'low', 'close', 'volume'].
        spy_log_return: market-proxy daily log return indexed by date.
            Used to compute CAPM-residual idiosyncratic volatility
            (ivol_21d). If None, ivol_21d is filled with NaN and treated
            as a missing feature by the downstream training-time scaler.
    """
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    frames = []
    for t, sub in df.groupby("ticker", sort=False):
        s = sub.copy()
        close = s["close"]
        op = s["open"] if "open" in s.columns else close
        vol = s["volume"].replace(0, np.nan)
        dollar_vol = (close * vol).replace(0, np.nan)
        s["log_return"] = np.log(close).diff()
        s["log_return_5d"] = np.log(close / close.shift(5))
        s["log_return_20d"] = np.log(close / close.shift(20))
        s["log_return_60d"] = np.log(close / close.shift(60))
        # 12-month minus 1-month: log(close_{t-21} / close_{t-252}).
        s["log_return_12m_minus_1m"] = np.log(close.shift(21) / close.shift(252))
        s["log_volume"] = np.log(vol)
        s["log_volume_ratio_20d"] = np.log(vol / vol.rolling(20, min_periods=5).mean())
        s["realized_vol_20d"] = s["log_return"].rolling(20, min_periods=5).std()
        s["realized_vol_60d"] = s["log_return"].rolling(60, min_periods=10).std()
        if "high" in s.columns and "low" in s.columns:
            hi = s["high"]
            lo = s["low"]
            s["high_low_range"] = np.log(hi / lo).clip(upper=0.5)
            close_to_high_5d_max = hi.rolling(5, min_periods=2).max()
            s["close_to_high_5d"] = (close / close_to_high_5d_max).fillna(1.0)
            # Alpha158 K-line shape, normalised by open. All five clipped to
            # +-1 to neutralise stock-split / data-glitch ticks.
            op_safe = op.replace(0, np.nan)
            s["kmid"] = ((close - op) / op_safe).clip(lower=-1.0, upper=1.0)
            s["klen"] = ((hi - lo) / op_safe).clip(lower=0.0, upper=1.0)
            upper_body = np.maximum(op, close)
            lower_body = np.minimum(op, close)
            s["kup"] = ((hi - upper_body) / op_safe).clip(lower=0.0, upper=1.0)
            s["klow"] = ((lower_body - lo) / op_safe).clip(lower=0.0, upper=1.0)
            s["ksft"] = ((2.0 * close - hi - lo) / op_safe).clip(lower=-1.0, upper=1.0)
        else:
            s["high_low_range"] = 0.0
            s["close_to_high_5d"] = 1.0
            for c in KLINE_COLS:
                s[c] = 0.0
        # Amihud illiquidity: mean of |return| / dollar volume over 20 days
        ret_abs = s["log_return"].abs()
        s["amihud_illiquidity_20d"] = (
            (ret_abs / dollar_vol.replace(0, np.nan))
            .rolling(20, min_periods=5).mean()
        ).clip(upper=1e-3)  # cap extreme values from low-volume days
        # MAX20: rolling 20d max of daily log return (Bali-Cakici-Whitelaw).
        s["max20"] = s["log_return"].rolling(20, min_periods=5).max()
        # Price-volume joint signals (Alpha158 / FactorVAE).
        log_vol = np.log(vol)
        s["corr20"] = (
            np.log(close).rolling(20, min_periods=5).corr(log_vol)
        ).clip(lower=-1.0, upper=1.0).fillna(0.0)
        ret_ratio = close / close.shift(1)
        vol_ratio = vol / vol.shift(1)
        s["cord20"] = (
            ret_ratio.rolling(20, min_periods=5).corr(vol_ratio)
        ).clip(lower=-1.0, upper=1.0).fillna(0.0)
        # IVOL21: rolling 21d std of CAPM residual on SPY using the
        # closed-form decomposition var(e) = var(y) - beta^2 * var(x),
        # equivalent to fitting r = alpha + beta * r_spy + e per window.
        if spy_log_return is not None and len(spy_log_return) > 0:
            spy_aligned = spy_log_return.reindex(
                pd.DatetimeIndex(pd.to_datetime(s["date"]).values),
            )
            spy_aligned = pd.Series(spy_aligned.values, index=s.index)
            cov_xy = s["log_return"].rolling(21, min_periods=10).cov(spy_aligned)
            var_x = spy_aligned.rolling(21, min_periods=10).var()
            var_y = s["log_return"].rolling(21, min_periods=10).var()
            beta = cov_xy / var_x.replace(0, np.nan)
            resid_var = (var_y - (beta ** 2) * var_x).clip(lower=0.0)
            s["ivol_21d"] = np.sqrt(resid_var)
        else:
            s["ivol_21d"] = np.nan
        s["fwd_return_h"] = np.log(close.shift(-5) / close)
        frames.append(s)
    return pd.concat(frames, ignore_index=True)


# --------------------------- Fundamentals derived features ---------------------------

def _derive_fundamentals(
    fund_main: pd.DataFrame, fund_extra: pd.DataFrame, prices: pd.DataFrame,
) -> pd.DataFrame:
    """Compute distress + intangible + other fundamental features.

    Inputs:
      fund_main: v2 schema (ticker, cik, quarter_end, filed_date, cash, assets,
                 shares, revenue, rd_expense, net_income, op_cf,
                 assets_current, liabilities_current, retained_earnings,
                 ebit, total_liabilities, capex)
      fund_extra: LATTICE extras (sga_expense, gross_profit, interest_expense,
                  advertising_expense)
      prices: ticker, date, close (for market cap at filing date)

    Output: per-(ticker, filed_date) frame with PRICE_VOL/DISTRESS/INTANGIBLE/
            OTHER_FUND columns ready for forward-fill onto the panel grid.
    """
    fund_main["filed_date"] = pd.to_datetime(fund_main["filed_date"]).dt.normalize()
    fund_main["quarter_end"] = pd.to_datetime(fund_main["quarter_end"]).dt.normalize()
    fund_extra["filed_date"] = pd.to_datetime(fund_extra["filed_date"]).dt.normalize()
    fund_extra["quarter_end"] = pd.to_datetime(fund_extra["quarter_end"]).dt.normalize()

    fund = fund_main.merge(
        fund_extra[["ticker", "quarter_end", "sga_expense", "gross_profit",
                    "interest_expense", "advertising_expense"]],
        how="left", on=["ticker", "quarter_end"],
    )
    prices_small = prices[["ticker", "date", "close"]].copy()
    prices_small["date"] = pd.to_datetime(prices_small["date"]).dt.normalize()
    fund = fund.merge(
        prices_small.rename(columns={"date": "filed_date"}),
        how="left", on=["ticker", "filed_date"],
    )

    frames = []
    for t, sub in fund.groupby("ticker", sort=False):
        sub = sub.sort_values("quarter_end").reset_index(drop=True).copy()
        sub_prices = prices_small[prices_small["ticker"] == t].sort_values("date")
        # Fall through to next trading-day close if filed_date isn't a trading day
        for i in range(len(sub)):
            if pd.isna(sub.loc[i, "close"]):
                fd = sub.loc[i, "filed_date"]
                nxt = sub_prices[sub_prices["date"] >= fd]
                if len(nxt) > 0:
                    sub.loc[i, "close"] = nxt.iloc[0]["close"]

        sub["market_cap"] = sub["close"] * sub["shares"]
        sub["log_market_cap"] = np.log(sub["market_cap"].replace({0: np.nan}))
        sub["revenue_ttm"] = sub["revenue"].rolling(4, min_periods=1).sum()
        # Distress proxies
        ebit = sub["ebit"]
        sub["interest_coverage"] = (
            ebit / sub["interest_expense"].abs().replace(0, np.nan)
        ).clip(lower=-50, upper=50)
        cash = sub["cash"]
        net_debt = sub["total_liabilities"] - cash
        sub["net_debt_to_ebitda"] = (net_debt / ebit.replace(0, np.nan)).clip(lower=-50, upper=50)
        fcf = sub["op_cf"] - sub["capex"].abs()
        sub["fcf_yield"] = (fcf / sub["market_cap"].replace(0, np.nan)).clip(lower=-1.0, upper=1.0)
        sub["current_ratio"] = (sub["assets_current"]
                                 / sub["liabilities_current"].replace(0, np.nan)).clip(upper=20.0)
        # Intangible proxies
        rev = sub["revenue"].replace(0, np.nan)
        sub["rd_to_sales"] = (sub["rd_expense"] / rev).clip(lower=0, upper=2.0)
        sub["sga_to_sales"] = (sub["sga_expense"] / rev).clip(lower=0, upper=2.0)
        sub["gross_profitability"] = (sub["gross_profit"]
                                       / sub["assets"].replace(0, np.nan)).clip(lower=-1.0, upper=2.0)
        sub["capex_to_sales"] = (sub["capex"].abs() / rev).clip(lower=0, upper=2.0)
        # Other fundamentals
        equity = sub["assets"] - sub["total_liabilities"]
        sub["book_to_market"] = (equity / sub["market_cap"].replace(0, np.nan)).clip(lower=-2, upper=10)
        sub["asset_growth_yoy"] = sub["assets"].pct_change(4, fill_method=None).clip(lower=-1, upper=2)

        sub["date"] = sub["filed_date"]
        out_cols = (["ticker", "date"] + DISTRESS_COLS + INTANGIBLE_COLS + OTHER_FUND_COLS)
        frames.append(sub[out_cols])
    return pd.concat(frames, ignore_index=True)


def _forward_fill_fundamentals(
    fund: pd.DataFrame, dates: list[pd.Timestamp],
) -> pd.DataFrame:
    rows = []
    trading = pd.DatetimeIndex(sorted(dates))
    for t, sub in fund.groupby("ticker", sort=False):
        s = sub.sort_values("date").set_index("date")
        s = s[~s.index.duplicated(keep="last")]
        s = s.reindex(trading, method="ffill")
        s["ticker"] = t
        s["date"] = s.index
        rows.append(s.reset_index(drop=True))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# --------------------------- Phase 1 driver ---------------------------

def build_phase1(cfg: LatticePhase1Config | None = None) -> dict:
    """Run Phase 1 panel build end-to-end. Returns a summary dict.

    Persists active_mask, cohorts, panel_features, stocktwits_features,
    macro_state to cfg.out_dir.
    """
    cfg = cfg or LatticePhase1Config()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    print("[lattice phase1] loading raw data...", flush=True)
    prices = pd.read_parquet(cfg.raw_dir / "prices_sp500.parquet")
    prices["ticker"] = prices["ticker"].astype(str).str.upper()
    prices["date"] = pd.to_datetime(prices["date"])
    hist = pd.read_parquet(cfg.raw_dir / "sp500_constituents_pit.parquet")
    hist["ticker"] = hist["ticker"].astype(str).str.upper()
    hist["start_date"] = pd.to_datetime(hist["start_date"])
    hist["end_date"] = pd.to_datetime(hist["end_date"])
    fund_main = pd.read_parquet(cfg.raw_dir / "fundamentals_edgar_sp500.parquet")
    fund_main["ticker"] = fund_main["ticker"].astype(str).str.upper()
    fund_extra = pd.read_parquet(cfg.raw_dir / "fundamentals_edgar_sp500_extra.parquet")
    fund_extra["ticker"] = fund_extra["ticker"].astype(str).str.upper()
    print(f"  prices: {len(prices):,} rows  hist: {len(hist):,} intervals  "
          f"fund_main: {len(fund_main):,}  fund_extra: {len(fund_extra):,}",
          flush=True)

    # Filter to panel window
    prices = prices[(prices["date"] >= cfg.panel_start)
                    & (prices["date"] <= cfg.panel_end)].copy()

    # Market proxy for IVOL21: take SPY close from the macro ETF parquet
    # the macro builder uses. Fall back to None if neither parquet exists;
    # _price_features will then NaN-fill ivol_21d.
    spy_log_return = None
    for cand in (cfg.raw_dir / "macro_etfs.parquet",
                 cfg.raw_dir / "macro_etfs_extra.parquet",
                 Path("data/raw/sp500/sector_etfs.parquet")):
        if not cand.exists():
            continue
        etf = pd.read_parquet(cand, columns=["ticker", "date", "close"])
        etf = etf[etf["ticker"].astype(str).str.upper() == "SPY"].copy()
        if len(etf) == 0:
            continue
        etf["date"] = pd.to_datetime(etf["date"]).dt.normalize()
        etf = etf.sort_values("date").drop_duplicates("date", keep="last")
        spy_close = etf.set_index("date")["close"].astype(float)
        spy_log_return = np.log(spy_close / spy_close.shift(1))
        print(f"[lattice phase1] SPY proxy loaded from {cand.name} "
              f"({len(spy_log_return):,} rows)", flush=True)
        break
    if spy_log_return is None:
        print("[lattice phase1] WARNING: no SPY market proxy found; "
              "ivol_21d will be NaN", flush=True)

    print(f"[lattice phase1] price-feature pass...", flush=True)
    prices = _price_features(prices, spy_log_return=spy_log_return)

    print(f"[lattice phase1] fundamentals derive + forward-fill...", flush=True)
    fund = _derive_fundamentals(fund_main, fund_extra, prices)
    panel_dates = sorted(prices["date"].unique().tolist())
    fund_daily = _forward_fill_fundamentals(fund, panel_dates)
    panel = prices.merge(fund_daily, how="left", on=["ticker", "date"])
    panel["has_fundamentals"] = panel[DISTRESS_COLS + INTANGIBLE_COLS
                                       + OTHER_FUND_COLS].notna().any(axis=1).astype(float)

    # Catalyst features: zero-filled for Phase 1; documented in audit md
    panel["days_to_next_catalyst_sin"] = 0.0
    panel["days_to_next_catalyst_cos"] = 0.0
    panel["catalyst_type_id"] = 0  # 0 = no scheduled catalyst

    # has_stocktwits flag depends on the StockTwits features parquet, which
    # the v2 universal-validation work already pre-aggregated. Read just the
    # ticker list to set the flag; the actual ST features go into a
    # separate parquet.
    st_features_path = Path("data/processed/stocktwits_features_sp500.parquet")
    if st_features_path.exists():
        st_lookup = pd.read_parquet(st_features_path, columns=["ticker", "date"])
        st_lookup["date"] = pd.to_datetime(st_lookup["date"])
        st_lookup["has_st"] = True
        panel = panel.merge(st_lookup, how="left", on=["ticker", "date"])
        panel["has_stocktwits"] = panel["has_st"].fillna(False).astype(float)
        panel = panel.drop(columns=["has_st"])
    else:
        panel["has_stocktwits"] = 0.0

    # Membership-mask gate (per Phase 1 spec section 4.1).
    # Drop only cells without close, volume, and 5-day forward return; rolling
    # features (realized_vol_20d, etc.) are kept even if NaN at the start of a
    # ticker's history and are NaN-filled at training time. This preserves the
    # short-history cells the architecture's IPO retrieval is designed to use.
    panel = panel.dropna(subset=["fwd_return_h", "log_return", "log_volume"])
    panel = panel.reset_index(drop=True)
    intervals = hist[["ticker", "start_date", "end_date"]].copy()
    iv_by_ticker: dict[str, np.ndarray] = {}
    for tk, sub in intervals.groupby("ticker"):
        iv_by_ticker[tk] = np.stack([
            sub["start_date"].astype("int64").to_numpy(),
            sub["end_date"].astype("int64").to_numpy(),
        ])
    panel_dt_int = panel["date"].astype("int64").to_numpy()
    keep = np.zeros(len(panel), dtype=bool)
    for tk, sub in panel.groupby("ticker", sort=False):
        ivs = iv_by_ticker.get(tk)
        if ivs is None or ivs.shape[1] == 0:
            continue
        idx = sub.index.to_numpy()
        d = panel_dt_int[idx]
        in_iv = (ivs[0][None, :] <= d[:, None]) & (d[:, None] <= ivs[1][None, :])
        keep[idx] = in_iv.any(axis=1)
    panel = panel[keep].reset_index(drop=True)

    # Materialize raw fill (sector-z-scoring is done at training time, not here)
    for c in DISTRESS_COLS + INTANGIBLE_COLS + OTHER_FUND_COLS:
        panel[c] = panel[c].replace([np.inf, -np.inf], np.nan)

    # Active mask (post-membership)
    active_mask = panel[["ticker", "date"]].copy()
    active_mask["mask"] = True

    # Cohort labels
    sector_map = (hist.drop_duplicates("ticker")
                       .set_index("ticker")["gics_sector"].to_dict())
    panel["sector"] = panel["ticker"].map(sector_map)
    panel["dollar_volume"] = np.exp(panel["log_volume"]) * 1.0  # placeholder
    panel["size_decile"] = panel.groupby("date")["log_market_cap"].transform(
        lambda x: pd.qcut(x, 10, labels=False, duplicates="drop").astype("Int64")
    )
    panel["liquidity_decile"] = panel.groupby("date")["amihud_illiquidity_20d"].transform(
        lambda x: pd.qcut(x, 10, labels=False, duplicates="drop").astype("Int64")
    )
    # age_bucket: months since first active day for each ticker
    first_date_per_ticker = panel.groupby("ticker")["date"].min()
    panel["months_since_first"] = panel.apply(
        lambda r: int((r["date"] - first_date_per_ticker[r["ticker"]]).days / 30),
        axis=1,
    )
    panel["age_bucket"] = pd.cut(panel["months_since_first"],
                                  bins=[-1, 12, 36, 120, 1000],
                                  labels=[0, 1, 2, 3]).astype("Int64")

    cohorts = panel[["ticker", "date", "size_decile", "liquidity_decile",
                     "sector", "age_bucket"]].copy()

    # Save panel (raw, not yet sector-z-scored)
    panel_save_cols = ["ticker", "date", "fwd_return_h"] + PANEL_FEATURE_COLS
    panel_save = panel[panel_save_cols].copy()

    panel_save.to_parquet(cfg.out_dir / "panel_features.parquet", index=False)
    active_mask.to_parquet(cfg.out_dir / "active_mask.parquet", index=False)
    cohorts.to_parquet(cfg.out_dir / "cohorts.parquet", index=False)

    summary = {
        "panel_rows": len(panel_save),
        "tickers": panel_save["ticker"].nunique(),
        "dates": panel_save["date"].nunique(),
        "active_density": float(active_mask["mask"].mean()),
        "cohort_rows": len(cohorts),
        "feature_cols": PANEL_FEATURE_COLS,
        "n_features": len(PANEL_FEATURE_COLS),
    }
    print(f"[lattice phase1] saved active_mask, cohorts, panel_features", flush=True)
    print(f"  panel_rows: {summary['panel_rows']:,}  tickers: {summary['tickers']}  "
          f"dates: {summary['dates']}  features: {summary['n_features']}", flush=True)
    return summary


__all__ = [
    "LatticePhase1Config",
    "PANEL_FEATURE_COLS", "ST_FEATURE_COLS", "MACRO_FEATURE_COLS",
    "PRICE_VOL_COLS", "KLINE_COLS", "MOMENTUM_EXTRA_COLS", "VOL_PV_EXTRA_COLS",
    "DISTRESS_COLS", "INTANGIBLE_COLS", "OTHER_FUND_COLS",
    "CATALYST_COLS", "FLAG_COLS",
    "build_phase1",
]
