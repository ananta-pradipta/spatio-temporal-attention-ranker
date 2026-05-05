#!/bin/bash
# D2 (target-vol-norm) + R-STAR on 244-ticker universe, 2 folds x 5 seeds.
set -e
cd /home/apradipta/phd-research
mkdir -p logs/D2_244
for F in 1 2; do
  for S in 42 43 44 45 46; do
    OUT="results/investigation/regime_memory/D2_244_fold${F}_seed${S}.json"
    LOG="logs/D2_244/fold${F}_seed${S}.log"
    if [ -f "$OUT" ]; then
      echo "[$(date +%H:%M:%S)] skip fold=$F seed=$S (exists)"
      continue
    fi
    echo "[$(date +%H:%M:%S)] D2 244 fold=$F seed=$S"
    python3 -u -m src.investigation.regime_memory.train \
      --fold "$F" --seed "$S" --K 4 --no-regime-token \
      --memory-tokens 0 --single-graph \
      --num-prototypes 16 --sparsity-weight 0.01 \
      --huber-delta 1.0 --vol-weight \
      --max-tickers 244 \
      --target-vol-norm --target-vol-window 60 \
      --output "$OUT" > "$LOG" 2>&1
  done
done
echo "[$(date +%H:%M:%S)] DONE"
