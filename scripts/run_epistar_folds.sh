#!/usr/bin/env bash
# Run epiSTAR across all (fold, seed) pairs.
# Usage: bash scripts/run_epistar_folds.sh
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="configs/epistar.yaml"
FOLDS=(1 2 3)
SEEDS=(42 43 44 45 46)

mkdir -p results/epistar logs/epistar
for fold in "${FOLDS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    out="logs/epistar/fold${fold}_seed${seed}.log"
    echo "[run_epistar_folds] fold=${fold} seed=${seed} -> ${out}"
    python -m src.v2.training.train_epistar --config "${CONFIG}" --fold "${fold}" --seed "${seed}" 2>&1 | tee "${out}"
  done
done
