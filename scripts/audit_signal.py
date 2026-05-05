"""Signal audit: walk the pipeline and measure where predictive signal lives.

Philosophy: if our final IC is near zero, either (a) our features have no
signal to begin with, (b) the target is dominated by noise, (c) the
model is failing to extract existing signal, or (d) there's a bug in
the data pipeline. Before blaming the architecture, check (a)-(d)
empirically with minimum-assumption diagnostics.

Diagnostics run in order:

  1. PER-FEATURE CROSS-SECTIONAL IC
     For each feature f_k, compute daily Pearson IC between f_k(t) and
     fwd_return_h(t) across the cross-section. Averaged over days.
     Interpretation: raw predictive content of each individual feature.

  2. NAIVE BASELINES (UNTRAINED)
     Predict fwd_return from simple rules: momentum (use log_return
     itself), reversal (negate), realized_vol-carry. Compare to zero.

  3. LINEAR REGRESSION R^2
     Fit a simple ridge on (features -> fwd_return) using train slice,
     evaluate on test. Floor for a perfectly-specified linear model.

  4. GRAPH-NEIGHBOR PREDICTIVENESS
     Does the average return of a ticker's correlation-graph neighbors
     predict its own fwd_return better than features alone?

  5. TARGET VARIANCE BREAKDOWN
     Decompose fwd_return variance into cross-sectional (same-day) and
     time-series (same-ticker) components. If cross-sectional variance
     dominates, ranking is the right framing; if time-series dominates,
     regression is.

Writes a terse report to docs/signal_audit.md.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


LINES: list[str] = []


def say(msg: str = "") -> None:
    print(msg)
    LINES.append(msg)


def daily_ic(x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    ics = []
    ricks = []
    for t in range(x.shape[0]):
        m = mask[t]
        if m.sum() < 3:
            continue
        a = x[t, m]
        b = y[t, m]
        if a.std() < 1e-8 or b.std() < 1e-8:
            continue
        ics.append(float(np.corrcoef(a, b)[0, 1]))
        rho, _ = spearmanr(a, b)
        if np.isfinite(rho):
            ricks.append(float(rho))
    return float(np.mean(ics)) if ics else float("nan"), float(np.mean(ricks)) if ricks else float("nan")


def main() -> None:
    from src.mtgn.training.panel import FEATURE_COLS, PanelConfig, build_panel, panel_to_tensors
    from src.mtgn.training.graph_builder import GraphConfig, build_correlation_edges
    from src.mtgn.training.train import temporal_split

    say("# Signal audit\n")
    cfg = PanelConfig(
        start_date="2020-01-01",
        end_date="2022-12-31",
        horizon_days=5,
        max_tickers=300,
    )
    panel, tickers, dates = build_panel(cfg)
    tensors = panel_to_tensors(panel, tickers, dates)
    x = tensors["x"]          # [T, N, F]
    y = tensors["y"]          # [T, N]
    mask = tensors["mask"]    # [T, N]
    T, N, F = x.shape
    say(f"panel: T={T} N={N} F={F}")
    say(f"feature columns: {FEATURE_COLS}")
    say(f"mask density: {mask.mean():.3f}")
    say("")

    train_slice, val_slice, test_slice = temporal_split(T, 0.15, 0.15)

    # ------------------------------------------------------------
    say("## 1. Per-feature cross-sectional IC on test slice\n")
    say("Raw feature f_k(t) vs fwd_return(t). No normalization, no")
    say("training. Measures direct signal strength per feature.\n")
    say("| Feature | IC | RankIC |")
    say("|---|---:|---:|")
    for i, col in enumerate(FEATURE_COLS):
        ic, ric = daily_ic(x[test_slice, :, i], y[test_slice], mask[test_slice])
        say(f"| {col} | {ic:+.4f} | {ric:+.4f} |")
    say("")

    # ------------------------------------------------------------
    say("## 2. Naive baselines on test slice\n")
    # Momentum: use log_return itself as y_hat
    lr = x[:, :, FEATURE_COLS.index("log_return")]
    ic_m, ric_m = daily_ic(lr[test_slice], y[test_slice], mask[test_slice])
    say(f"momentum (sign of log_return)   IC {ic_m:+.4f}   RankIC {ric_m:+.4f}")
    ic_r, ric_r = daily_ic(-lr[test_slice], y[test_slice], mask[test_slice])
    say(f"reversal (-log_return)          IC {ic_r:+.4f}   RankIC {ric_r:+.4f}")
    # Cross-sectional vol-carry
    rv = x[:, :, FEATURE_COLS.index("realized_vol")]
    ic_v, ric_v = daily_ic(rv[test_slice], y[test_slice], mask[test_slice])
    say(f"realized_vol-carry              IC {ic_v:+.4f}   RankIC {ric_v:+.4f}")
    ic_nv, ric_nv = daily_ic(-rv[test_slice], y[test_slice], mask[test_slice])
    say(f"anti-realized_vol               IC {ic_nv:+.4f}   RankIC {ric_nv:+.4f}")
    # StockTwits bullish ratio
    bu = x[:, :, FEATURE_COLS.index("st_bullish_ratio")]
    ic_b, ric_b = daily_ic(bu[test_slice], y[test_slice], mask[test_slice])
    say(f"st_bullish_ratio                IC {ic_b:+.4f}   RankIC {ric_b:+.4f}")
    say("")

    # ------------------------------------------------------------
    say("## 3. Linear regression (ridge) on features -> fwd_return\n")
    from sklearn.linear_model import Ridge
    # Build train design matrix: flatten [T_train * N, F]
    def flatten(sl):
        xs = x[sl].reshape(-1, F)
        ys = y[sl].reshape(-1)
        ms = mask[sl].reshape(-1)
        return xs[ms], ys[ms]
    Xtr, ytr = flatten(train_slice)
    Xte, yte = flatten(test_slice)
    # Normalize on train
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0).clip(min=1e-6)
    Xtr_n = (Xtr - mu) / sd
    Xte_n = (Xte - mu) / sd
    say(f"Train samples: {len(Xtr):,}   Test samples: {len(Xte):,}")
    for alpha in [1.0, 10.0, 100.0, 1000.0]:
        model = Ridge(alpha=alpha)
        model.fit(Xtr_n, ytr)
        y_hat_te = model.predict(Xte_n)
        # Report sign-only accuracy + correlation
        corr = float(np.corrcoef(y_hat_te, yte)[0, 1])
        r2 = 1 - ((y_hat_te - yte) ** 2).sum() / ((yte - yte.mean()) ** 2).sum()
        say(f"Ridge alpha={alpha:<6g}  train R^2={model.score(Xtr_n, ytr):+.4f}  test R^2={r2:+.4f}  test corr={corr:+.4f}")
    # Coefficients for alpha=10
    m = Ridge(alpha=10.0).fit(Xtr_n, ytr)
    say("")
    say("coefficients at alpha=10 (normalized inputs):")
    for c, w in zip(FEATURE_COLS, m.coef_):
        say(f"  {c:25s}  {w:+.5f}")
    say("")
    # Also compute daily cross-sectional IC of the ridge prediction
    yhat_flat = np.zeros_like(y)
    x_flat_all = ((x.reshape(-1, F) - mu) / sd).reshape(T, N, F)
    for t in range(T):
        for i in range(N):
            yhat_flat[t, i] = float(m.predict(x_flat_all[t, i : i + 1])[0])
    ic_r, ric_r = daily_ic(yhat_flat[test_slice], y[test_slice], mask[test_slice])
    say(f"ridge alpha=10 daily cross-sectional test IC {ic_r:+.4f}   RankIC {ric_r:+.4f}")
    say("")

    # ------------------------------------------------------------
    say("## 4. Graph-neighbor predictiveness\n")
    # Build correlation graph from first 60 days of train
    head = x[train_slice.start : min(train_slice.start + 60, train_slice.stop)]
    ei, ew = build_correlation_edges(head, GraphConfig())
    say(f"correlation graph: {ei.shape[1]} edges")
    # For each (t, i), average log_return of neighbors in the graph
    src, dst = ei[0], ei[1]
    neigh_return = np.zeros_like(lr)
    for t in range(T):
        row_lr = lr[t]
        agg = np.zeros(N)
        cnt = np.zeros(N)
        np.add.at(agg, dst, row_lr[src])
        np.add.at(cnt, dst, 1)
        neigh_return[t] = np.where(cnt > 0, agg / np.maximum(cnt, 1), 0.0)
    ic_n, ric_n = daily_ic(neigh_return[test_slice], y[test_slice], mask[test_slice])
    say(f"avg-neighbor-log-return daily IC {ic_n:+.4f}   RankIC {ric_n:+.4f}")
    # Combined: own + neighbor
    combined = lr + 0.5 * neigh_return
    ic_c, ric_c = daily_ic(combined[test_slice], y[test_slice], mask[test_slice])
    say(f"own + 0.5 * neighbor          IC {ic_c:+.4f}   RankIC {ric_c:+.4f}")
    say("")

    # ------------------------------------------------------------
    say("## 5. Target variance decomposition\n")
    y_masked = np.where(mask, y, np.nan)
    y_cross_std = np.nanstd(y_masked, axis=1)   # per-day cross-sectional std
    y_time_std = np.nanstd(y_masked, axis=0)    # per-ticker time-series std
    say(f"mean cross-sectional std of fwd_return_h: {np.nanmean(y_cross_std):.4f}")
    say(f"mean time-series       std of fwd_return_h: {np.nanmean(y_time_std):.4f}")
    say(f"cross / time ratio: {np.nanmean(y_cross_std) / np.nanmean(y_time_std):.3f}")
    say("")

    # ------------------------------------------------------------
    Path("docs/signal_audit.md").write_text("\n".join(LINES) + "\n")
    say("wrote docs/signal_audit.md")


if __name__ == "__main__":
    main()
