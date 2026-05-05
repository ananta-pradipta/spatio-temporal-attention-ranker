"""Diagnostic 4: Graph-edge staleness on fold 2.

For each edge (i, j) in the mechanistic graph built on train data,
compute the correlation between log_returns of tickers i and j in:
  - train window
  - test window

Identify edges where the correlation changes substantially or flips
sign. A large fraction of "problematic edges" supports the hypothesis
that the STATIC graph itself becomes mis-calibrated during fold 2.

Output: docs/fold2_edge_staleness.md + CSV of per-edge diagnostics.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.mtgn.graph.edges import EdgeBuildConfig, build_mechanistic_edges
from src.mtgn.training.panel_enriched import (
    EnrichedPanelConfig, FEATURE_COLS, build_enriched_panel, panel_to_tensors,
)
from src.mtgn.training.train_baselines import walk_forward_fold

OUT_MD = Path("docs/fold2_edge_staleness.md")
LOG_RET_INDEX = 0


def pairwise_corr(returns: np.ndarray, mask: np.ndarray, i: int, j: int,
                  min_overlap: int = 30) -> float:
    m = mask[:, i] & mask[:, j]
    if m.sum() < min_overlap:
        return np.nan
    a = returns[m, i]; b = returns[m, j]
    if a.std() < 1e-12 or b.std() < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def main():
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)

    print("Building panel...")
    cfg = EnrichedPanelConfig(start_date="2015-01-01", end_date="2022-12-31",
                              horizon_days=5, max_tickers=100)
    panel, tickers, dates = build_enriched_panel(cfg)
    tensors = panel_to_tensors(panel, tickers, dates)
    returns = tensors["x"][:, :, LOG_RET_INDEX]
    mask = tensors["mask"]
    N = len(tickers)

    tr2, va2, te2 = walk_forward_fold(dates, 2, 5)

    # Build mechanistic graph on fold-2 train window
    edge_end = str(pd.Timestamp(dates[tr2.stop - 1]).date())
    ei, ew = build_mechanistic_edges(
        tickers, EdgeBuildConfig(train_start="2015-01-01", train_end=edge_end),
        require_nonempty=False,
    )
    print(f"Mechanistic edges built: {ei.shape[1]} edges")

    # For each edge, compute train vs test correlation
    rows = []
    seen = set()
    for k in range(ei.shape[1]):
        u, v = int(ei[0, k]), int(ei[1, k])
        pair = (min(u, v), max(u, v))
        if pair in seen:
            continue
        seen.add(pair)
        corr_train = pairwise_corr(returns, mask, u, v, min_overlap=100)
        corr_test = pairwise_corr(returns[te2], mask[te2], u, v, min_overlap=30)
        if np.isnan(corr_train) or np.isnan(corr_test):
            continue
        delta = corr_test - corr_train
        sign_flip = (corr_train * corr_test) < -0.01
        rows.append({
            "u": tickers[u], "v": tickers[v],
            "corr_train": corr_train, "corr_test": corr_test,
            "delta": delta, "abs_delta": abs(delta),
            "sign_flip": bool(sign_flip),
            "weight": float(ew[k]) if k < len(ew) else 1.0,
        })
    df = pd.DataFrame(rows)
    df = df.sort_values("abs_delta", ascending=False)

    # Summary stats
    mean_train = df["corr_train"].mean()
    mean_test = df["corr_test"].mean()
    mean_abs_delta = df["abs_delta"].mean()
    flips = df["sign_flip"].sum()
    flip_frac = df["sign_flip"].mean()
    big_delta = (df["abs_delta"] > 0.3).sum()

    print(f"\nEdges analyzed: {len(df)}")
    print(f"Mean train corr: {mean_train:+.4f}")
    print(f"Mean test corr:  {mean_test:+.4f}")
    print(f"Mean |Δ corr|:   {mean_abs_delta:.4f}")
    print(f"Sign flips: {flips} ({flip_frac:.1%})")
    print(f"|Δ| > 0.3: {big_delta} edges")

    df.to_csv(OUT_MD.parent / "fold2_edge_staleness.csv", index=False)

    # Markdown report
    def md_small(d: pd.DataFrame) -> str:
        lines = ["| u | v | corr_train | corr_test | Δ | sign_flip |",
                 "|---|---|---|---|---|---|"]
        for _, r in d.iterrows():
            lines.append(f"| {r['u']} | {r['v']} | {r['corr_train']:+.3f} | "
                         f"{r['corr_test']:+.3f} | {r['delta']:+.3f} | "
                         f"{'YES' if r['sign_flip'] else ''} |")
        return "\n".join(lines)

    report = f"""# Fold-2 Edge Staleness Diagnostic (Phase 1, D4)

Date: 2026-04-16
Method: for each edge (u, v) in the mechanistic graph built on the
fold-2 train window, compute the correlation between u and v's log
returns in (a) the full train window and (b) the fold-2 test
window. Large |Δ| indicates the edge has changed; sign flip
indicates the relationship INVERTED.

## 1. Summary

- Edges analyzed: {len(df)} (unique pairs, both directions dedupicated)
- Mean train correlation: {mean_train:+.4f}
- Mean fold-2 test correlation: {mean_test:+.4f}
- Mean |Δ correlation|: {mean_abs_delta:.4f}
- Sign flips: {flips} / {len(df)} ({flip_frac:.1%})
- |Δ| > 0.3: {big_delta} edges

If train correlations were ~0.2 and test correlations drift to
~0.3-0.4 (uniform upward shift due to drawdown correlation rise)
without sign flips, the graph STRUCTURE is stable but the LEVEL
changes. Good for models that normalize correlation strength.

If sign flips are >10% of edges, the graph encodes relationships
that no longer hold during fold 2 — supports pruning.

## 2. 20 most shifted edges (largest |Δ|)

{md_small(df.head(20))}

## 3. 20 sign-flipped edges

{md_small(df[df['sign_flip']].head(20))}

## 4. Implication

A large fraction of edges keeping their sign = graph structure is
mostly stable. A large sign-flip fraction = graph structure is
unreliable during fold 2 and pruning could help.

If a small number of sign-flipping edges cluster around specific
tickers (e.g., the ABSI/AKBA draggers from D1), **targeted edge
pruning** is a concrete, low-risk fix: rebuild the mechanistic
graph minus those edges and retrain REM 3B. This is a different
fix than "rebuild the graph monthly" (iter 3D adaptive-graph, which
did NOT help); this is a train-time surgical removal of unstable
edges.

## 5. Files

- `docs/fold2_edge_staleness.csv`
"""
    OUT_MD.write_text(report)
    print(f"\nwrote {OUT_MD}")


if __name__ == "__main__":
    main()
