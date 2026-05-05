#!/bin/bash
# Vol-only ablation: REM iter 3B architecture + inverse-volatility weighting only (no Huber).
# 3 folds x 5 seeds = 15 runs.
set -e
cd /home/apradipta/phd-research
mkdir -p logs/iter10b_volonly
for F in 1 2 3; do
  for S in 42 43 44 45 46; do
    OUT="results/investigation/regime_memory/iter10b_volonly_fold${F}_seed${S}.json"
    LOG="logs/iter10b_volonly/fold${F}_seed${S}.log"
    echo "[$(date +%H:%M:%S)] fold=$F seed=$S"
    python3 -m src.investigation.regime_memory.train \
      --fold "$F" --seed "$S" --K 4 --no-regime-token \
      --memory-tokens 0 --single-graph \
      --num-prototypes 16 --sparsity-weight 0.01 \
      --vol-weight \
      --output "$OUT" > "$LOG" 2>&1
  done
done
echo "[$(date +%H:%M:%S)] DONE"
