"""Long-short portfolio backtest + decile-monotonicity analysis.

Reads prediction npz files (y_hat, y_true, mask) per (model, fold, seed),
averages predictions across seeds, then computes:

Part 1 -- Long-short portfolio metrics per fold:
  - Each day t: rank tickers by y_hat, long top-K, short bottom-K equal-weight
  - Held-period return = 5-day forward (y_true is the 5-day forward log return)
  - Daily LS "portfolio return" (overlapping 5-day): mean(top-K y_true) - mean(bot-K y_true)
  - Annualised return = mean(daily LS) * 252
  - Annualised Sharpe = mean / std * sqrt(252)
  - Maximum drawdown of cumulative LS curve
  - Turnover = average daily fraction of single-side positions changing
  - Net Sharpe after 5 bps round-trip TC
  Reported for K=10 and K=25.

Part 2 -- Decile-monotonicity / quintile-portfolio analysis:
  - Each day t: sort tickers by y_hat, form 5 equal-weighted quintile portfolios
    Q1 = bottom (lowest predicted), Q5 = top (highest predicted)
  - Average 5-day forward return per quintile per fold
  - Plot per-fold quintile bar chart for RAG-STAR vs strongest baselines

Outputs:
  results/exports/portfolio_metrics.csv
  results/exports/quintile_returns.csv
  drafts/paper_aaai/figures/quintile_portfolios.pdf  (and KDD)
"""
from __future__ import annotations

import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT_DIR = Path("results/exports")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "RAG-STAR":   "results/dow_epistar_v23_no_rate_memory",
    "MASTER":     "results/baselines_244/master_v2",
    "FactorVAE":  "results/baselines_244/factorvae_v2",
    "DySTAGE":    "results/baselines_244/dystage_v2",
    "StockMixer": "results/baselines_244/stockmixer_v2",
}

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def load_avg_predictions(model_dir: str, fold: int):
    pattern = f"{model_dir}/fold{fold}_seed*_predictions.npz"
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    yhat_seeds = []
    y_true = mask = dates = None
    for f in files:
        d = np.load(f, allow_pickle=True)
        yhat_seeds.append(d["y_hat"])
        if y_true is None:
            y_true = d["y_true"]
            for name in ("loss_mask", "eval_mask", "mask"):
                if name in d:
                    mask = d[name].astype(bool); break
            if mask is None:
                raise RuntimeError(f"no mask in {f}")
            dates = d["dates"] if "dates" in d else None
    yhat = np.mean(np.stack(yhat_seeds), axis=0)
    return yhat, y_true, mask, dates


def long_short_pnl(yhat, y_true, mask, K):
    """Daily long-short portfolio return time series.

    For each test day t with at least 2K active tickers:
      sort active tickers by y_hat[t], long top K, short bot K
      portfolio_return[t] = mean(y_true[t, top_K]) - mean(y_true[t, bot_K])
    """
    T = yhat.shape[0]
    pnl = []
    pos_long, pos_short = None, None  # for turnover
    turnovers = []
    for t in range(T):
        m = mask[t]
        if m.sum() < 2 * K:
            continue
        active_idx = np.flatnonzero(m)
        scores = yhat[t, active_idx]
        order = np.argsort(scores)
        top_idx = set(active_idx[order[-K:]].tolist())
        bot_idx = set(active_idx[order[:K]].tolist())
        rets = y_true[t, list(top_idx)].mean() - y_true[t, list(bot_idx)].mean()
        pnl.append(rets)
        if pos_long is not None:
            tot_change = len(pos_long.symmetric_difference(top_idx)) + len(pos_short.symmetric_difference(bot_idx))
            turnovers.append(tot_change / (4 * K))  # both sides, both directions
        pos_long, pos_short = top_idx, bot_idx
    pnl = np.asarray(pnl)
    turnovers = np.asarray(turnovers) if turnovers else np.array([0.0])
    return pnl, turnovers


def portfolio_metrics(pnl, turnovers, tc_bps=5, horizon=5):
    """Annualised metrics. y_true is the 5-day forward log return, so each
    daily-entry observation in `pnl` is a 5-day return. We annualise treating
    these as non-overlapping 5-day-equivalent observations: factor 252/horizon
    for return, sqrt(252/horizon) for vol. This gives Sharpe = mean/std *
    sqrt(252/h) which is robust to the overlap autocorrelation.
    """
    if len(pnl) == 0:
        return dict.fromkeys(["ann_ret", "ann_vol", "sharpe", "max_dd", "turnover", "net_sharpe"], np.nan)
    mean = float(pnl.mean())
    vol = float(pnl.std(ddof=1)) if len(pnl) > 1 else 0.0
    factor_ret = 252.0 / horizon            # ~50.4
    factor_vol = np.sqrt(factor_ret)        # ~7.1
    ann_ret = mean * factor_ret
    ann_vol = vol * factor_vol
    sharpe = ann_ret / ann_vol if ann_vol > 1e-12 else 0.0
    cum = np.cumsum(pnl)
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak)
    max_dd = float(dd.min()) if len(dd) > 0 else 0.0
    avg_turnover = float(turnovers.mean())
    # Transaction cost: round-trip = 2 * single-side cost; tc_bps applied to traded fraction
    tc_per_obs = 2 * (tc_bps / 1e4) * avg_turnover
    net_pnl = pnl - tc_per_obs
    net_mean = float(net_pnl.mean())
    net_vol = float(net_pnl.std(ddof=1)) if len(net_pnl) > 1 else 0.0
    net_sharpe = (net_mean * factor_ret) / (net_vol * factor_vol) if net_vol > 1e-12 else 0.0
    return dict(ann_ret=ann_ret, ann_vol=ann_vol, sharpe=sharpe, max_dd=max_dd,
                turnover=avg_turnover, net_sharpe=net_sharpe)


def quintile_returns(yhat, y_true, mask, n_q=5):
    """For each test day, sort active tickers into n_q quintiles by yhat,
    compute equal-weighted mean of y_true per quintile, average across days.
    """
    T = yhat.shape[0]
    rets = [[] for _ in range(n_q)]
    for t in range(T):
        m = mask[t]
        n = int(m.sum())
        if n < n_q * 2:
            continue
        active_idx = np.flatnonzero(m)
        scores = yhat[t, active_idx]
        order = np.argsort(scores)
        sorted_active = active_idx[order]
        # Split into n_q roughly equal chunks (small chunks at the end if uneven)
        chunks = np.array_split(sorted_active, n_q)
        for q, ch in enumerate(chunks):
            if len(ch) > 0:
                rets[q].append(y_true[t, ch].mean())
    return [float(np.mean(r)) if r else np.nan for r in rets]


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


def main():
    # Part 1: long-short portfolio table
    rows = []
    for fold in (1, 2, 3):
        for model_name, model_dir in MODELS.items():
            res = load_avg_predictions(model_dir, fold)
            if res is None:
                continue
            yhat, y_true, mask, _ = res
            for K in (10, 25):
                pnl, turn = long_short_pnl(yhat, y_true, mask, K=K)
                m = portfolio_metrics(pnl, turn)
                rows.append({
                    "fold": fold, "model": model_name, "K": K,
                    "n_days": len(pnl),
                    **m,
                })
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "portfolio_metrics.csv", index=False)
    print(f"Saved {OUT_DIR / 'portfolio_metrics.csv'} with {len(df)} rows")

    # Print summary table by fold and K
    for K in (10, 25):
        print(f"\n=== Long-Short K={K} ===")
        sub = df[df.K == K].pivot_table(index="model", columns="fold",
                                         values=["ann_ret", "sharpe", "max_dd", "turnover", "net_sharpe"],
                                         aggfunc="mean")
        # Print compact
        print(f"{'Model':<12} {'F1 ann_ret':<10} {'F1 Sharpe':<10} {'F2 ann_ret':<10} {'F2 Sharpe':<10} {'F3 ann_ret':<10} {'F3 Sharpe':<10}")
        for model in MODELS:
            if model not in sub.index: continue
            row = sub.loc[model]
            line = f"{model:<12} "
            for f in (1, 2, 3):
                line += f"{row[('ann_ret', f)]:+.3f}    {row[('sharpe', f)]:+.2f}      "
            print(line)

    # Part 2: quintile-monotonicity table
    qrows = []
    for fold in (1, 2, 3):
        for model_name, model_dir in MODELS.items():
            res = load_avg_predictions(model_dir, fold)
            if res is None: continue
            yhat, y_true, mask, _ = res
            qrets = quintile_returns(yhat, y_true, mask, n_q=5)
            for q, r in enumerate(qrets):
                qrows.append({"fold": fold, "model": model_name, "quintile": q + 1, "mean_ret": r})
    qdf = pd.DataFrame(qrows)
    qdf.to_csv(OUT_DIR / "quintile_returns.csv", index=False)
    print(f"\nSaved {OUT_DIR / 'quintile_returns.csv'} with {len(qdf)} rows")

    # Plot quintile portfolios per fold
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    fold_titles = {1: "Fold 1 (2020)", 2: "Fold 2 (2021-H2 to 2022-H1)", 3: "Fold 3 (2022-H2)"}
    bar_width = 0.16
    colors = {"RAG-STAR": "#1f3b6b", "MASTER": "#9c4221", "FactorVAE": "#2e7d32",
              "DySTAGE": "#7c3aed", "StockMixer": "#666666"}
    for ax, fold in zip(axes, (1, 2, 3)):
        sub = qdf[qdf.fold == fold]
        models = list(MODELS.keys())
        x = np.arange(5)
        for i, model in enumerate(models):
            mvals = sub[sub.model == model].sort_values("quintile")["mean_ret"].values
            if len(mvals) == 5:
                offset = (i - len(models) / 2 + 0.5) * bar_width
                ax.bar(x + offset, mvals * 100, bar_width, label=model, color=colors.get(model, "#888"))
        ax.axhline(0, color="black", lw=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(["Q1\n(bottom)", "Q2", "Q3", "Q4", "Q5\n(top)"])
        ax.set_title(fold_titles[fold])
        ax.set_ylabel("Mean 5-d forward return (\\%)" if fold == 1 else "")
        if fold == 1:
            ax.legend(loc="lower left", fontsize=8, ncol=1)
        ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    for out_dir in [Path("drafts/paper_aaai/figures"), Path("drafts/paper_kdd/figures")]:
        out_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_dir / "quintile_portfolios.pdf", bbox_inches="tight")
    plt.close()
    print(f"\nSaved quintile_portfolios.pdf to 2 dirs")


if __name__ == "__main__":
    main()
