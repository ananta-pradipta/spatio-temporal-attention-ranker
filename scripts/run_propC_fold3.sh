#!/bin/bash
# Authorized single touch of fold 3 for propC lambda=1.0 (strongest F2 variant).
set -e
cd /home/apradipta/phd-research
mkdir -p logs/propC_fold3
for S in 42 43 44 45 46; do
  OUT="results/investigation/regime_memory/propC_l1p0_fold3_seed${S}.json"
  LOG="logs/propC_fold3/seed${S}.log"
  if [ -f "$OUT" ]; then
    echo "[$(date +%H:%M:%S)] skip seed=$S (exists)"
    continue
  fi
  echo "[$(date +%H:%M:%S)] propC lambda=1.0 fold=3 seed=$S"
  python3 -u -m src.investigation.regime_memory.train \
    --fold 3 --seed "$S" --K 4 --no-regime-token \
    --memory-tokens 0 --single-graph \
    --num-prototypes 16 --sparsity-weight 0.01 \
    --huber-delta 1.0 --vol-weight \
    --dann-lambda-max 1.0 --dann-hidden 64 \
    --output "$OUT" > "$LOG" 2>&1
done
echo "[$(date +%H:%M:%S)] DONE"
