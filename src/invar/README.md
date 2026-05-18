# InVAR

INverted VARiate Attention for Spatio-Temporal Equity Ranking on the
S&P 500.

Two-axis inverted Transformer: variate-token attention over tickers
(spatial axis) plus a parallel regime-window-token axis (temporal axis)
with cross-attention, gated combination, and position-wise FFN.

## Layout

```
src/invar/
  data/        InvarDataset reusing LATTICE panel/macro/cohorts/scalers
  model/       Invar end-to-end module + 3 regime-axis variants
  training/    train.py + loss.py
  evaluation/  metrics.py + statistics.py
  experiments/ smoke_test, run_fold1, headline runner, ablation sweeps,
               biotech case study
```

## Data dependencies

Reuses LATTICE artifacts on disk:
- `data/lattice/processed/panel_features.parquet`
- `data/lattice/processed/macro_state.parquet`
- `data/lattice/processed/cohorts.parquet`
- `experiments/lattice/fold{1,2,3}/scalers.pkl`

Does NOT import from `src/lattice/`, `src/v2/`, `src/mtgn/`, or `src/pilot/`.

## Phases

Phase 1: skeleton + smoke test. Runs forward on 5 random Fold 1 days,
asserts no NaNs, saves tiny ckpt, parameter count under 5M.

Phase 2: training loop + Fold 1 sanity check (test IC at least +0.020).

Phase 3: headline run + 7 external baselines on F1 + F2 (F3 separately
gated by user).

Phase 4: ablations + biotech case study.

See `docs/invar_design.md`.
