"""Manual delisting CSV with Shumway-Warther imputation (SBP Addition 2).

The full WRDS-CRSP delisting feed is not available; the spec asks for
hand-curated coverage of the panel's delistings using public sources.
This module provides:

    - ``detect_delisting_candidates``: scans the raw price panel for
      tickers whose tradable history ends before the panel's last day
      and never resumes. Emits a candidate CSV (template) with the
      auto-classified reason set to ``MnA_unknown`` so the user can
      hand-correct.
    - ``apply_delisting_imputation``: reads the curated CSV and writes
      the Shumway-Warther terminal returns onto the last 5 trading days
      of each delisted ticker's panel slice. Performance-related
      delistings get -0.55 (Shumway and Warther 1999), voluntary get
      0.0, M&A entries use the curated terminal return when present.

CSV schema (``data/delisting_log_v1.csv``):
    ticker,delisting_date,reason,imputed_terminal_return,source_note
    reason in {performance, MnA, voluntary, MnA_unknown}.
    imputed_terminal_return: float or empty (NaN). Empty -> default
        per reason: performance -> -0.55, voluntary -> 0.0, MnA_unknown
        -> 0.0 (treat as voluntary if user has not curated it).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


SHUMWAY_WARTHER_PERFORMANCE = -0.55
DEFAULT_REASON_RETURN = {
    "performance": SHUMWAY_WARTHER_PERFORMANCE,
    "voluntary": 0.0,
    "MnA_unknown": 0.0,
    "MnA": 0.0,  # only used when curated terminal return is missing
}


@dataclass
class DelistingConfig:
    """Hyperparameters for delisting detection / imputation."""

    raw_prices_parquet: Path = Path("data/raw/prices_universe.parquet")
    delisting_log_path: Path = Path("data/delisting_log_v1.csv")
    panel_end: str = "2022-12-31"
    final_window_days: int = 5  # last N trading days to apply terminal return
    early_exit_threshold_days: int = 21  # ticker must stop trading >= 21 days before panel_end
    performance_drop_threshold: float = 0.5  # final close < 50% of 6-month-ago = "performance"


def detect_delisting_candidates(cfg: DelistingConfig | None = None) -> pd.DataFrame:
    """Scan the raw price panel for tickers whose history ends early.

    Returns a DataFrame with one row per candidate, classified into
    {performance, voluntary, MnA_unknown} by the most defensible
    auto-rule (price drawdown threshold). Reasons should be hand-
    corrected against StockAnalysis / Wikipedia / news for the final
    curated CSV.
    """
    cfg = cfg or DelistingConfig()
    raw = pd.read_parquet(cfg.raw_prices_parquet).copy()
    raw["date"] = pd.to_datetime(raw["date"]).dt.normalize()
    panel_end = pd.Timestamp(cfg.panel_end).normalize()
    rows: list[dict] = []

    for tk, sub in raw.groupby("ticker", sort=False):
        sub = sub.sort_values("date")
        last_date = sub["date"].max()
        if (panel_end - last_date).days <= cfg.early_exit_threshold_days:
            continue
        last_close = float(sub["close"].iloc[-1])
        # Reference: close ~6 months earlier (approx 126 trading days back).
        if len(sub) >= 126:
            ref_close = float(sub["close"].iloc[-126])
        else:
            ref_close = float(sub["close"].iloc[0])
        if ref_close > 0 and last_close / ref_close < cfg.performance_drop_threshold:
            reason = "performance"
        else:
            reason = "MnA_unknown"
        imputed = DEFAULT_REASON_RETURN.get(reason, 0.0) if reason != "MnA_unknown" else None
        rows.append({
            "ticker": tk,
            "delisting_date": last_date.strftime("%Y-%m-%d"),
            "reason": reason,
            "imputed_terminal_return": imputed,
            "source_note": "auto-detect (curate manually)",
        })
    cols = ["ticker", "delisting_date", "reason",
            "imputed_terminal_return", "source_note"]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows).sort_values("delisting_date").reset_index(drop=True)


def write_template_csv(cfg: DelistingConfig | None = None) -> pd.DataFrame:
    """Write the auto-detected candidates to the delisting log path.

    If the file already exists it is NOT overwritten (manual curation
    is preserved). Returns the DataFrame either way.
    """
    cfg = cfg or DelistingConfig()
    cands = detect_delisting_candidates(cfg)
    if cfg.delisting_log_path.exists():
        existing = pd.read_csv(cfg.delisting_log_path)
        return existing
    cfg.delisting_log_path.parent.mkdir(parents=True, exist_ok=True)
    cands.to_csv(cfg.delisting_log_path, index=False)
    return cands


def load_delisting_log(cfg: DelistingConfig | None = None) -> pd.DataFrame:
    """Load the curated CSV; fall back to auto-detect on first run."""
    cfg = cfg or DelistingConfig()
    if not cfg.delisting_log_path.exists():
        return write_template_csv(cfg)
    return pd.read_csv(cfg.delisting_log_path)


def apply_delisting_imputation(
    y: np.ndarray,
    label_mask: np.ndarray,
    dates: list[pd.Timestamp],
    tickers: list[str],
    cfg: DelistingConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Apply Shumway-Warther terminal returns to the last 5 trading days
    before each delisted ticker's exit.

    Args:
        y: [T, N] 5-day forward log returns (will be modified in place
            on the affected cells).
        label_mask: [T, N] bool. Cells with imputed terminal returns are
            set to True so the loss can pick them up.
        dates: panel trading days.
        tickers: panel tickers.
        cfg: DelistingConfig.

    Returns:
        y_out: [T, N] modified.
        label_out: [T, N] modified.
        diag: dict with imputation counts per reason.
    """
    cfg = cfg or DelistingConfig()
    log = load_delisting_log(cfg)
    if "ticker" not in log.columns or len(log) == 0:
        return y, label_mask, {"applied_cells": 0, "tickers_imputed": 0, "by_reason": {}}

    panel_dates = pd.DatetimeIndex(pd.to_datetime(dates).normalize())
    ticker_to_idx = {t: i for i, t in enumerate(tickers)}
    date_to_idx = {pd.Timestamp(d): i for i, d in enumerate(panel_dates)}
    y_out = y.copy()
    label_out = label_mask.copy()
    counts = {"performance": 0, "voluntary": 0, "MnA_unknown": 0, "MnA": 0}
    applied = 0

    for _, row in log.iterrows():
        tk = str(row["ticker"]).strip()
        if tk not in ticker_to_idx:
            continue
        ti = ticker_to_idx[tk]
        d_str = str(row.get("delisting_date", "")).strip()
        if not d_str or d_str.lower() == "nan":
            continue
        try:
            d_ts = pd.Timestamp(d_str).normalize()
        except Exception:
            continue
        # Find the panel-date index at or just before d_ts.
        if d_ts not in date_to_idx:
            cand = panel_dates[panel_dates <= d_ts]
            if len(cand) == 0:
                continue
            d_idx = date_to_idx[cand[-1]]
        else:
            d_idx = date_to_idx[d_ts]
        reason = str(row.get("reason", "voluntary")).strip()
        ret_raw = row.get("imputed_terminal_return")
        if pd.isna(ret_raw) or ret_raw is None or str(ret_raw).strip() == "":
            ret = DEFAULT_REASON_RETURN.get(reason, 0.0)
        else:
            try:
                ret = float(ret_raw)
            except Exception:
                ret = DEFAULT_REASON_RETURN.get(reason, 0.0)
        # Apply to the last `final_window_days` cells before d_idx (inclusive).
        lo = max(0, d_idx - cfg.final_window_days + 1)
        hi = d_idx + 1
        per_day_ret = ret / max(1, hi - lo)
        for t in range(lo, hi):
            y_out[t, ti] = per_day_ret
            label_out[t, ti] = True
            applied += 1
        counts[reason] = counts.get(reason, 0) + 1

    return y_out, label_out, {
        "applied_cells": applied,
        "tickers_imputed": int(sum(counts.values())),
        "by_reason": counts,
    }


__all__ = [
    "SHUMWAY_WARTHER_PERFORMANCE",
    "DelistingConfig",
    "detect_delisting_candidates",
    "write_template_csv",
    "load_delisting_log",
    "apply_delisting_imputation",
]
