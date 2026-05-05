#!/usr/bin/env bash
# Run DyReg-STAR across all (fold, seed) pairs.
# Usage: bash scripts/run_dyreg_star_folds.sh
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="configs/dyreg_star.yaml"
FOLDS=(1 2 3)
SEEDS=(42 43 44 45 46)

mkdir -p results/dyreg_star logs/dyreg_star
for fold in "${FOLDS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    out="logs/dyreg_star/fold${fold}_seed${seed}.log"
    echo "[run_dyreg_star_folds] fold=${fold} seed=${seed} -> ${out}"
    python -m src.v2.training.train_dyreg_star --config "${CONFIG}" --fold "${fold}" --seed "${seed}" 2>&1 | tee "${out}"
  done
done
