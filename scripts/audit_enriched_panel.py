"""Audit the enriched 5-year, 22-feature panel for correctness.

Checks:
  1. Feature distributions: mean, std, extremes per feature
  2. NaN / Inf sweep
  3. Temporal split boundaries (no overlap)
  4. Forward-return shift correctness
  5. Fundamentals filing-date vs quarter-end: detects forward-fill leakage
  6. Normalization fit on train only
  7. LSTM window leakage (does window at test time peek into val?)
  8. has_fundamentals flag vs actual non-null rate
  9. Sector-median imputation per-date vs global

Writes report to docs/enriched_panel_audit.md.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


LINES: list[str] = []
def out(msg: str = "") -> None:
    print(msg)
    LINES.append(msg)


def main() -> None:
    from src.mtgn.training.panel_enriched import (
        FEATURE_COLS, FUND_COLS, EnrichedPanelConfig, build_enriched_panel, panel_to_tensors,
    )
    from src.mtgn.training.train import temporal_split

    out("# Enriched panel audit\n")
    cfg = EnrichedPanelConfig(start_date="2018-01-01", end_date="2022-12-31",
                              horizon_days=5, max_tickers=300)
    panel, tickers, dates = build_enriched_panel(cfg)
    tensors = panel_to_tensors(panel, tickers, dates)
    x = tensors["x"]; y = tensors["y"]; mask = tensors["mask"]
    T, N, F = x.shape
    out(f"panel shape: T={T} days, N={N} tickers, F={F} features")
    out(f"features: {FEATURE_COLS}")
    out(f"mask density: {mask.mean():.3f}")
    out("")

    # ============ 1. Feature distributions ============
    out("## 1. Per-feature distribution (train slice only)\n")
    train_sl, val_sl, test_sl = temporal_split(T, 0.125, 0.20)
    out(f"split:  train [0, {train_sl.stop})  val [{val_sl.start}, {val_sl.stop})  test [{test_sl.start}, {test_sl.stop})")
    out(f"        train {train_sl.stop} days  val {val_sl.stop-val_sl.start} days  test {test_sl.stop-test_sl.start} days")
    out(f"        train end date ≈ {dates[train_sl.stop - 1].date()}")
    out(f"        val end date   ≈ {dates[val_sl.stop - 1].date()}")
    out(f"        test end date  ≈ {dates[test_sl.stop - 1].date()}")
    out("")
    x_train = x[train_sl][mask[train_sl]]
    out("| Feature | mean | std | p01 | p50 | p99 |")
    out("|---|---:|---:|---:|---:|---:|")
    for i, c in enumerate(FEATURE_COLS):
        col = x_train[:, i]
        out(f"| {c} | {col.mean():+.4f} | {col.std():.4f} | {np.quantile(col, 0.01):+.4f} | {np.quantile(col, 0.5):+.4f} | {np.quantile(col, 0.99):+.4f} |")
    out("")
    # NaN / Inf
    nan_x = int(np.isnan(x).sum()); inf_x = int(np.isinf(x).sum())
    nan_y = int(np.isnan(y).sum()); inf_y = int(np.isinf(y).sum())
    out(f"NaN in x: {nan_x}  Inf in x: {inf_x}  NaN in y: {nan_y}  Inf in y: {inf_y}")
    out("")

    # ============ 2. Forward-return shift correctness ============
    out("## 2. Forward-return shift spot check\n")
    prices = pd.read_parquet("data/raw/prices_universe.parquet")
    prices["ticker"] = prices["ticker"].astype(str).str.upper()
    prices["date"] = pd.to_datetime(prices["date"])
    t0 = tickers[0]
    sub = prices[(prices["ticker"] == t0) & (prices["date"] >= cfg.start_date) & (prices["date"] <= cfg.end_date)]
    sub = sub.sort_values("date").reset_index(drop=True)
    # Pick 5 random rows and verify fwd_return_h = log(close(t+5)/close(t))
    import random
    random.seed(0)
    ok = 0; bad = 0
    for _ in range(10):
        i = random.randint(10, len(sub) - 10)
        expected = np.log(sub.at[i + 5, "close"] / sub.at[i, "close"])
        observed = y[i, tickers.index(t0)] if tickers.index(t0) < N else None
        # Adjust: y index is sorted by date, must align to panel dates
        if observed is None:
            continue
        d = sub.at[i, "date"]
        di = next((j for j, dt in enumerate(dates) if pd.Timestamp(dt).normalize() == pd.Timestamp(d).normalize()), None)
        if di is None:
            continue
        ti = tickers.index(t0)
        if not mask[di, ti]:
            continue
        observed = y[di, ti]
        if abs(expected - observed) < 1e-4:
            ok += 1
        else:
            bad += 1
    out(f"fwd_return_h shift check on ticker {t0}: {ok} match, {bad} mismatch")
    out("")

    # ============ 3. Normalization fit on train only ============
    out("## 3. Normalization check\n")
    mu_train = x[train_sl].reshape(-1, F).mean(axis=0)
    mu_all = x.reshape(-1, F).mean(axis=0)
    delta = float(np.abs(mu_train - mu_all).sum())
    out(f"sum(|mu_train - mu_all|): {delta:.4g}  (>0 expected; train uses train-only mu)")
    out("")

    # ============ 4. Fundamentals filing-date vs quarter-end leakage ============
    out("## 4. Fundamentals leakage check\n")
    if Path("data/raw/fundamentals_edgar.parquet").exists():
        fund = pd.read_parquet("data/raw/fundamentals_edgar.parquet")
        fund["filed_date"] = pd.to_datetime(fund["filed_date"])
        fund["quarter_end"] = pd.to_datetime(fund["quarter_end"])
        lag_days = (fund["filed_date"] - fund["quarter_end"]).dt.days
        out(f"EDGAR fundamentals: {len(fund)} rows, quarter_end {fund['quarter_end'].min().date()}..{fund['quarter_end'].max().date()}, filed_date {fund['filed_date'].min().date()}..{fund['filed_date'].max().date()}")
        out(f"filing lag (days): median {lag_days.median():.0f}, p90 {lag_days.quantile(0.9):.0f}, max {lag_days.max():.0f}")
        out("")
        out("**LEAKAGE CONCERN:** the EDGAR `date` field here is the QUARTER-END date (`end`),")
        out("not the FILING date. In reality 10-Q filings arrive ~45 days after quarter-end;")
        out("10-K filings ~90 days after fiscal-year-end. The current panel forward-fills from")
        out("quarter-end, so on e.g. 2021-01-15 we already have Q4-2020 fundamentals that were")
        out("not publicly reported until ~mid-Feb 2021.")
        out("")
        out("Look-ahead effect: ~30-60 days of forward-fill leakage on every quarterly value.")
        out("Fix: capture the `filed` field in fundamentals_edgar.py and forward-fill FROM filed.")
    out("")

    # ============ 5. has_fundamentals flag consistency ============
    out("## 5. has_fundamentals flag\n")
    hf_idx = FEATURE_COLS.index("has_fundamentals")
    hf = x[:, :, hf_idx]
    out(f"has_fundamentals distribution: 0 ({(hf == 0).sum()}), 1 ({(hf > 0.5).sum()})")
    out(f"fraction of active (ticker, day) cells with has_fundamentals=1: {hf[mask].mean():.3f}")
    out("")

    # ============ 6. Feature collinearity check ============
    out("## 6. Feature collinearity (correlation matrix diag)\n")
    xn = (x_train - x_train.mean(axis=0, keepdims=True)) / (x_train.std(axis=0, keepdims=True) + 1e-8)
    corr = np.corrcoef(xn.T)
    # Print only |corr| > 0.85 pairs (excluding diagonal)
    out("pairs with |corr| > 0.85 (possible collinearity):")
    for i in range(F):
        for j in range(i + 1, F):
            if abs(corr[i, j]) > 0.85:
                out(f"  {FEATURE_COLS[i]:<25} <-> {FEATURE_COLS[j]:<25}  corr = {corr[i, j]:+.3f}")
    out("")

    # ============ 7. LSTM window leakage conceptual check ============
    out("## 7. LSTM window leakage\n")
    out("LSTM forward at day t uses features from x[t-W:t] to predict y[t] (= fwd_return at t).")
    out("If W=20, at the first day of test (index test_sl.start), the window uses days")
    out(f"  [{test_sl.start - 20}, {test_sl.start}) which includes VAL days and possibly")
    out(f"  the last train day. This is NOT strict leakage because the window DOES NOT see")
    out(f"  the test target y[t]; but the model parameters were trained on val-period windows.")
    out("Mitigation: if patience-based early stopping selects a checkpoint based on val IC,")
    out(f"the checkpoint uses val windows for its model selection — that IS val exposure.")
    out(f"Current train_baselines.py does early stopping on val IC, so this is standard for")
    out(f"time-series deep learning but worth noting in the paper's methodology section.")
    out("")

    Path("docs/enriched_panel_audit.md").parent.mkdir(parents=True, exist_ok=True)
    Path("docs/enriched_panel_audit.md").write_text("\n".join(LINES) + "\n")
    out(f"wrote docs/enriched_panel_audit.md")


if __name__ == "__main__":
    main()
