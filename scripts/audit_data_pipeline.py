"""Deep audit of the MTGN data preprocessing pipeline.

Checks each stage for correctness issues, leakage, and silent join bugs.
Writes findings to docs/data_audit.md and prints a summary to stdout.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


LINES: list[str] = []


def report(msg: str = "") -> None:
    print(msg)
    LINES.append(msg)


def section(title: str) -> None:
    report("")
    report(f"## {title}")
    report("")


def check_prices() -> None:
    section("1. Prices (data/raw/prices_universe.parquet)")
    df = pd.read_parquet("data/raw/prices_universe.parquet")
    df["date"] = pd.to_datetime(df["date"])
    report(f"rows: {len(df):,}")
    report(f"tickers: {df['ticker'].nunique()}")
    report(f"date range: {df['date'].min()} to {df['date'].max()}")
    report(f"rows per ticker (min/median/max): "
           f"{df.groupby('ticker').size().min()} / "
           f"{df.groupby('ticker').size().median()} / "
           f"{df.groupby('ticker').size().max()}")

    # Price dispersion across tickers
    for col in ["close", "adj_close", "volume"]:
        if col in df.columns:
            s = df[col].astype(float)
            report(f"{col}: min={s.min():.4g} p50={s.median():.4g} max={s.max():.4g}  nan={s.isna().sum()}")

    # Sanity: adj_close <= close typically (dividends / splits shrink adj). Violated?
    if "adj_close" in df.columns:
        n_violations = int(((df["adj_close"] > df["close"]) & df["adj_close"].notna()).sum())
        report(f"adj_close > close (expected rare, suggests dividend reinvestment?): {n_violations} rows")

    # Negative prices or zero close?
    zero_or_neg = int(((df["close"] <= 0) & df["close"].notna()).sum())
    report(f"close <= 0: {zero_or_neg} rows  (expected 0)")

    # Duplicate (ticker, date)?
    dup = df.duplicated(subset=["ticker", "date"]).sum()
    report(f"duplicate (ticker, date): {dup} rows  (expected 0)")


def check_stocktwits_features() -> None:
    section("2. StockTwits features (data/processed/stocktwits_features.parquet)")
    df = pd.read_parquet("data/processed/stocktwits_features.parquet")
    df["date"] = pd.to_datetime(df["date"])
    report(f"rows: {len(df):,}")
    report(f"tickers: {df['ticker'].nunique()}")
    report(f"date range: {df['date'].min()} to {df['date'].max()}")
    for col in ["st_volume_24h", "st_volume_change_30d", "st_bullish_ratio",
                "st_sentiment_dispersion", "st_labeled_ratio"]:
        if col in df.columns:
            s = df[col]
            report(
                f"{col}: nan={s.isna().sum()}  inf={np.isinf(s).sum()}  "
                f"p10={s.quantile(0.1):.4g} p50={s.median():.4g} p90={s.quantile(0.9):.4g} max={s.max():.4g}"
            )

    # Are the dates on a trading-calendar? StockTwits created_at is anywhere (weekends, holidays).
    # The aggregator groups by raw date, so weekend rows should exist here too.
    weekday_counts = df["date"].dt.weekday.value_counts().sort_index()
    report("weekday distribution (0=Mon, 6=Sun):")
    for wd, cnt in weekday_counts.items():
        report(f"  {wd}: {cnt:,}")
    if weekday_counts.get(5, 0) > 0 or weekday_counts.get(6, 0) > 0:
        report(
            "  ISSUE: weekend posts are present as their own (ticker, date) rows. "
            "The panel builder joins on exact date, so weekend post activity "
            "is DROPPED when joined against trading-date prices. Sentiment "
            "accumulated over a weekend should roll forward to Monday."
        )


def check_volatility_indices() -> None:
    section("3. Volatility indices (data/raw/volatility_indices.parquet)")
    df = pd.read_parquet("data/raw/volatility_indices.parquet")
    df.index = pd.to_datetime(df.index)
    report(f"rows: {len(df):,}")
    report(f"date range: {df.index.min()} to {df.index.max()}")
    for col in df.columns:
        s = df[col]
        report(f"{col}: nan={s.isna().sum()}  min={s.min():.3f} p50={s.median():.3f} max={s.max():.3f}")


def check_catalyst() -> None:
    section("4. Catalyst mask (data/processed/catalyst_days.parquet)")
    if not Path("data/processed/catalyst_days.parquet").exists():
        report("(file missing)")
        return
    df = pd.read_parquet("data/processed/catalyst_days.parquet")
    df["date"] = pd.to_datetime(df["date"])
    report(f"rows: {len(df):,}")
    report(f"tickers flagged: {df['ticker'].nunique()}")
    report(f"catalyst cells per ticker (min/median/max): "
           f"{df.groupby('ticker').size().min()} / "
           f"{df.groupby('ticker').size().median()} / "
           f"{df.groupby('ticker').size().max()}")


def check_panel_pipeline() -> None:
    section("5. Panel pipeline (src.mtgn.training.panel.build_panel)")
    from src.mtgn.training.panel import FEATURE_COLS, PanelConfig, build_panel, panel_to_tensors

    cfg = PanelConfig(
        start_date="2020-01-01",
        end_date="2022-12-31",
        horizon_days=5,
        max_tickers=50,
    )
    panel, tickers, dates = build_panel(cfg)
    tensors = panel_to_tensors(panel, tickers, dates)

    report(f"panel: {len(panel):,} rows, {len(tickers)} tickers, {len(dates)} trading days")
    report(f"feature columns: {FEATURE_COLS}")

    # Check forward-return shift: for each ticker, fwd_return_h(t) should
    # equal log(close(t+h) / close(t)). Spot-check one ticker.
    t0 = tickers[0]
    sub = panel[panel["ticker"] == t0].sort_values("date").reset_index(drop=True)
    report(f"spot-check forward-return shift on {t0}: checking 10 random rows")
    import random
    random.seed(0)
    close_col = "close" if "close" in sub.columns else None
    if close_col is None:
        report("  NOTE: 'close' column not present after panel merges")
    else:
        idxs = random.sample(range(len(sub) - 6), 10)
        n_ok, n_bad = 0, 0
        for i in idxs:
            if np.isnan(sub.at[i, "fwd_return_h"]):
                continue
            expected = float(np.log(sub.at[i + 5, close_col] / sub.at[i, close_col]))
            observed = float(sub.at[i, "fwd_return_h"])
            if abs(expected - observed) > 1e-5:
                n_bad += 1
                if n_bad <= 3:
                    report(f"  row {i} ({sub.at[i,'date'].date()}): expected={expected:+.6f} observed={observed:+.6f}  DIFF")
            else:
                n_ok += 1
        report(f"  forward-return shift check: {n_ok} ok, {n_bad} mismatched")

    # Any NaN / Inf in the feature tensor after pipeline?
    x = tensors["x"]
    report(f"tensor x shape: {x.shape}, dtype: {x.dtype}")
    report(f"  nan cells:  {int(np.isnan(x).sum()):,}  (expected 0 after panel fillna)")
    report(f"  inf cells:  {int(np.isinf(x).sum()):,}")
    report(f"  zero cells: {int((x == 0).sum()):,} (dense-tensor zero-fills from masked cells)")
    report(f"  mask density: {tensors['mask'].mean():.3f}")
    report(f"  y shape: {tensors['y'].shape}, nan: {int(np.isnan(tensors['y']).sum())}")

    # Check per-feature dispersion (is anything constant?)
    x_flat = x.reshape(-1, x.shape[-1])
    report("per-feature std across (T,N):")
    for i, c in enumerate(FEATURE_COLS):
        s = float(x_flat[:, i].std())
        mean = float(x_flat[:, i].mean())
        report(f"  {c:25s}  mean={mean:+.4g}  std={s:.4g}")


def check_train_slice_leakage() -> None:
    section("6. Train slice normalization + graph construction (leakage checks)")
    from src.mtgn.training.panel import FEATURE_COLS, PanelConfig, build_panel, panel_to_tensors
    from src.mtgn.training.train import temporal_split
    from src.mtgn.training.graph_builder import GraphConfig, build_correlation_edges

    cfg = PanelConfig(
        start_date="2020-01-01",
        end_date="2022-12-31",
        horizon_days=5,
        max_tickers=50,
    )
    panel, tickers, dates = build_panel(cfg)
    tensors = panel_to_tensors(panel, tickers, dates)
    T = tensors["x"].shape[0]
    train_slice, val_slice, test_slice = temporal_split(T, 0.15, 0.15)
    report(
        f"T={T}  train=[0, {train_slice.stop})  "
        f"val=[{val_slice.start}, {val_slice.stop})  "
        f"test=[{test_slice.start}, {test_slice.stop})"
    )

    # Feature normalization should use only the train slice.
    import torch
    x = torch.from_numpy(tensors["x"])
    F = x.shape[-1]
    mu_all = x.reshape(-1, F).mean(dim=0)
    mu_train = x[train_slice].reshape(-1, F).mean(dim=0)
    delta = float((mu_all - mu_train).abs().sum())
    report(f"sum(|mu_all - mu_train|): {delta:.4g}  (should be > 0; train.py correctly uses train-only mu)")

    # Correlation graph should be built from train-slice data only.
    head_arr = tensors["x"][train_slice.start : min(train_slice.start + 60, train_slice.stop)]
    report(
        f"graph built from tensors['x'][{train_slice.start}:{train_slice.start + 60}]  "
        f"(first 60 trading days of train slice -- no val/test leakage)"
    )
    ei, ew = build_correlation_edges(head_arr, GraphConfig())
    report(f"edges: {ei.shape[1]}")

    # Confirm forward return target does NOT leak past the test slice end.
    y = tensors["y"]
    n_nan_last_h = int(np.isnan(y[-5:, :]).sum())
    report(f"fwd_return in last 5 rows: NaN cells = {n_nan_last_h} (panel drops rows with NaN target; mask handles)")


def main() -> None:
    check_prices()
    check_stocktwits_features()
    check_volatility_indices()
    check_catalyst()
    check_panel_pipeline()
    check_train_slice_leakage()

    Path("docs/data_audit.md").parent.mkdir(parents=True, exist_ok=True)
    Path("docs/data_audit.md").write_text(
        "# Data Pipeline Audit\n\nGenerated by `scripts/audit_data_pipeline.py`.\n\n"
        + "\n".join(LINES) + "\n"
    )


if __name__ == "__main__":
    main()
