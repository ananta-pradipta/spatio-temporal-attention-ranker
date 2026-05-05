# v2: Regime-Routed Ranker (RRR)

Started 2026-05-01. QE deadline 2026-05-31.

## Architecture (Option B from restart decision)

Backbone: STAR-style spatio-temporal attention over top-N=8 mechanistic graph neighbors x W=20 day window.

Head: regime classifier (k-means on training-window risk-feature trajectories, K regimes) plus K small per-regime expert MLP heads. Final prediction is a soft mixture: $\hat{y}_i = \sum_{k=1}^{K} \alpha_k(s_t) \cdot \text{Head}_k(z_i)$, where $s_t$ is the day-level risk signature and $\alpha_k$ are softmax routing weights.

## Differences from R-STAR (archived in `archive/v1_models/`)

R-STAR addressed regime shift via a loss-function change (Huber + inverse-volatility weighting on a fixed architecture). RRR addresses the same failure architecturally: route the prediction through a per-regime expert head rather than reweighting the loss. The empirical motivation (fold-2 diagnostic suite) is shared.

## Non-negotiables (inherited from v1)

- Same 84-ticker biotech panel, 22 features, 2015-2022 (see `docs/dataset_catalogue.md`)
- Same walk-forward folds with 5-day embargo
- Same 4-check leakage audit
- Same falsification bars: fold-1 IC >= +0.050, fold-2 IC >= +0.010
- Same baselines for comparison (Ridge, LSTM, GCN, GAT, StockMixer, MASTER)
- 5 seeds (42-46), paired t-test for significance claims

## Things v1 already proved DO NOT work (do not re-attempt)

- Cross-attention over noisy retrieved keys (untrainable from scratch)
- FiLM-based risk conditioning (hurt under leakage-free eval)
- Auxiliary volatility supervision (caused leakage)
- Per-ticker gating (no improvement)
- RankNet pairwise loss (no improvement over cross-sectional MSE)
- TGN memory module from `torch_geometric.TGNMemory`

## Layout

```
src/v2/
  data/        # panel loading, regime feature builder
  model/       # RRR architecture
  training/    # train_rrr.py, evaluate_rrr.py
```
