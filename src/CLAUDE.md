# Research Code

## Layout

- `src/mtgn/` — Memorizing Temporal Graph Networks (Phase 1, qualifying exam scope). Active development.
- `src/pilot/` — Pilot v2 reference code (GAT, LSTM, MLP ranking baselines on 38 biotech stocks with 5 price features). Kept as reference and as future baseline comparison; not actively developed.

## Running experiments

MTGN (Phase 1):
```
python -m src.mtgn.training.train --config configs/mtgn/phase1.yaml
```

Pilot v2 (reference only):
```
python -m src.pilot.train.train --config src/pilot/configs/pilot_v2_gat.yaml
```

## Conventions
- Modular: data loading, graph construction, memory, episodic store, attention, model, training, evaluation in separate modules under `src/mtgn/`.
- Type hints on all public functions; Google-style docstrings.
- YAML configs for experiments. No hardcoded hyperparameters.
- Reproducibility: random seeds, logged hyperparams, versioned data splits.
- Checkpoints and logs go under `experiments/` and `logs/`, not `src/`.

## MTGN module responsibilities
- `data/` — universe construction, price download, StockTwits subset download, sentiment extraction, VIX/VXN/VVIX fetch, train/val/test temporal split
- `graph/` — continuous-time event stream construction from prices + tweets + catalysts
- `memory/` — TGN memory (GRU update, memory detach between batches, last_update tracking)
- `store/` — salience-gated episodic store with FAISS HNSW index and write policy (four triggers)
- `attention/` — spatial attention over temporal neighbors (TGAT-style) and episodic temporal attention over retrieved store entries
- `model/` — MTGN forward pass, ranking head, quantile-loss risk-aware head
- `training/` — training loop respecting TGN leakage discipline, joint ListNet + pinball loss
- `evaluation/` — IC, RankIC, NDCG@k, long-short portfolio metrics, coverage, slice decompositions (all / catalyst-window / calm)
