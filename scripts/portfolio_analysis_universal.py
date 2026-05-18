"""Long-short portfolio backtest + quintile-monotonicity for the UNIVERSAL panel.

Universal-panel analogue of scripts/portfolio_analysis.py. Reads prediction
npz files per (model, fold, seed) for the S&P 500 Universal Ticker panel,
averages across seeds, and computes:

Part 1 -- Long-short top-25/bottom-25 portfolio metrics, 5-fold mean:
  annualised return, gross/net Sharpe (5 bps round-trip TC), max drawdown,
  turnover. Annualisation factor sqrt(252/5) for the 5-day forward horizon.

Part 2 -- Quintile-portfolio mean 5-day forward return per fold (Q1=bottom
predicted, Q5=top), plotted as a 5-panel figure.

Outputs:
  results/exports/portfolio_metrics_universal.csv
  results/exports/quintile_returns_universal.csv
  drafts/universal_paper_aaai/figures/quintile_portfolios.pdf  (and KDD)
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
    "RAG-STAR":   "results/rag_star_universe_v2_no_ipo",
    "MASTER":     "results/baselines_universal_two_regime_val/master",
    "FactorVAE":  "results/baselines_universal_two_regime_val/factorvae",
    "iTransformer": "results/baselines_universal_two_regime_val/itransformer",
    "StockMixer": "results/baselines_universal_two_regime_val/stockmixer",
}
FOLDS = (1, 2, 3, 4, 5)


def load_avg_predictions(model_dir: str, fold: int):
    files = sorted(glob.glob(f"{model_dir}/fold{fold}_seed*_predictions.npz"))
    if not files:
        return None
    yhat_seeds = []
    y_true = mask = None
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
    yhat = np.mean(np.stack(yhat_seeds), axis=0)
    return yhat, y_true, mask


def long_short_pnl(yhat, y_true, mask, K):
    T = yhat.shape[0]
    pnl = []
    pos_long, pos_short = None, None
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
            turnovers.append(tot_change / (4 * K))
        pos_long, pos_short = top_idx, bot_idx
    pnl = np.asarray(pnl)
    turnovers = np.asarray(turnovers) if turnovers else np.array([0.0])
    return pnl, turnovers


def portfolio_metrics(pnl, turnovers, tc_bps=5, horizon=5):
    """AR / IR / Net-IR for a dollar-neutral long-short book.

    Each daily entry in `pnl` is a 5-day forward return; we annualise with
    the non-overlapping factor 252/horizon (return) and sqrt(252/horizon)
    (vol). For a market-neutral book IR == Sharpe (portfolio return is the
    active return). Max drawdown is computed on a NON-OVERLAPPING 5-day
    rebalanced equity curve (take every horizon-th daily obs and compound
    in log space) so it is economically interpretable rather than an
    overlap-summation artifact.
    """
    if len(pnl) == 0:
        return dict.fromkeys(["ann_ret", "ann_vol", "ir", "max_dd", "turnover", "net_ir"], np.nan)
    mean = float(pnl.mean())
    vol = float(pnl.std(ddof=1)) if len(pnl) > 1 else 0.0
    factor_ret = 252.0 / horizon
    factor_vol = np.sqrt(factor_ret)
    ann_ret = mean * factor_ret
    ann_vol = vol * factor_vol
    ir = ann_ret / ann_vol if ann_vol > 1e-12 else 0.0
    # Non-overlapping 5-day-rebalanced equity curve for an interpretable DD.
    nonoverlap = pnl[::horizon]
    eq = np.cumsum(nonoverlap)  # log-return space
    peak = np.maximum.accumulate(eq)
    max_dd = float((eq - peak).min()) if len(eq) > 0 else 0.0
    avg_turnover = float(turnovers.mean())
    tc_per_obs = 2 * (tc_bps / 1e4) * avg_turnover
    net_pnl = pnl - tc_per_obs
    net_mean = float(net_pnl.mean())
    net_vol = float(net_pnl.std(ddof=1)) if len(net_pnl) > 1 else 0.0
    net_ir = (net_mean * factor_ret) / (net_vol * factor_vol) if net_vol > 1e-12 else 0.0
    return dict(ann_ret=ann_ret, ann_vol=ann_vol, ir=ir, max_dd=max_dd,
                turnover=avg_turnover, net_ir=net_ir)


def quintile_returns(yhat, y_true, mask, n_q=5):
    T = yhat.shape[0]
    rets = [[] for _ in range(n_q)]
    for t in range(T):
        m = mask[t]
        if int(m.sum()) < n_q * 2:
            continue
        active_idx = np.flatnonzero(m)
        order = np.argsort(yhat[t, active_idx])
        for q, ch in enumerate(np.array_split(active_idx[order], n_q)):
            if len(ch) > 0:
                rets[q].append(y_true[t, ch].mean())
    return [float(np.mean(r)) if r else np.nan for r in rets]


def main():
    # Part 1: long-short portfolio table (per fold, then 5-fold mean)
    rows = []
    for fold in FOLDS:
        for model_name, model_dir in MODELS.items():
            res = load_avg_predictions(model_dir, fold)
            if res is None:
                continue
            yhat, y_true, mask = res
            for K in (10, 25):
                pnl, turn = long_short_pnl(yhat, y_true, mask, K=K)
                m = portfolio_metrics(pnl, turn)
                rows.append({"fold": fold, "model": model_name, "K": K,
                             "n_days": len(pnl), **m})
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "portfolio_metrics_universal.csv", index=False)

    k25 = df[df.K == 25]
    print("=== Per-fold IR (K=25) + 5-fold-mean AR/IR/NetIR/DD/Turn ===")
    hdr = "Model".ljust(13) + " ".join(f"F{f}IR".rjust(7) for f in FOLDS)
    hdr += "   5F_AR  5F_IR 5F_NetIR  5F_DD 5F_Turn"
    print(hdr)
    for model in MODELS:
        sub = k25[k25.model == model]
        if sub.empty:
            continue
        cells = []
        for f in FOLDS:
            r = sub[sub.fold == f]
            cells.append(f"{r['ir'].mean():+.2f}".rjust(7) if not r.empty else "  --  ")
        ar = sub["ann_ret"].mean()
        ir = sub["ir"].mean()
        nir = sub["net_ir"].mean()
        dd = sub["max_dd"].mean()
        tn = sub["turnover"].mean()
        print(f"{model:<13}" + " ".join(cells) +
              f"  {ar:+6.1%} {ir:+6.2f} {nir:+7.2f} {dd:+6.1%} {tn:6.1%}")

    # Part 2: quintile returns + figure
    qrows = []
    for fold in FOLDS:
        for model_name, model_dir in MODELS.items():
            res = load_avg_predictions(model_dir, fold)
            if res is None:
                continue
            yhat, y_true, mask = res
            for q, r in enumerate(quintile_returns(yhat, y_true, mask, 5)):
                qrows.append({"fold": fold, "model": model_name, "quintile": q + 1, "mean_ret": r})
    qdf = pd.DataFrame(qrows)
    qdf.to_csv(OUT_DIR / "quintile_returns_universal.csv", index=False)

    fig, axes = plt.subplots(1, 5, figsize=(20, 3.4), sharey=True)
    fold_titles = {1: "F1 (2020 COVID)", 2: "F2 (2021-22 rotation)",
                   3: "F3 (2022-23 post-shock)", 4: "F4 (2024 AI rally)",
                   5: "F5 (2025 Fed-cut)"}
    bar_width = 0.16
    colors = {"RAG-STAR": "#1f3b6b", "MASTER": "#9c4221", "FactorVAE": "#2e7d32",
              "iTransformer": "#7c3aed", "StockMixer": "#666666"}
    for ax, fold in zip(axes, FOLDS):
        sub = qdf[qdf.fold == fold]
        models = list(MODELS.keys())
        x = np.arange(5)
        for i, model in enumerate(models):
            mvals = sub[sub.model == model].sort_values("quintile")["mean_ret"].values
            if len(mvals) == 5:
                offset = (i - len(models) / 2 + 0.5) * bar_width
                ax.bar(x + offset, mvals * 100, bar_width, label=model,
                       color=colors.get(model, "#888"))
        ax.axhline(0, color="black", lw=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(["Q1", "Q2", "Q3", "Q4", "Q5"])
        ax.set_title(fold_titles[fold], fontsize=9)
        ax.set_ylabel("Mean 5-d fwd return (\\%)" if fold == 1 else "")
        if fold == 1:
            ax.legend(loc="lower left", fontsize=7, ncol=1)
        ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    for out_dir in [Path("drafts/universal_paper_aaai/figures"),
                    Path("drafts/universal_paper_kdd/figures")]:
        out_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_dir / "quintile_portfolios.pdf", bbox_inches="tight")
    plt.close()
    print("\nSaved quintile_portfolios.pdf to 2 dirs")


if __name__ == "__main__":
    main()
