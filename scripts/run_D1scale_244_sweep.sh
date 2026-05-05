#!/bin/bash
# D1 SetSTAR at 2x capacity (hidden=256) + longer training (25 epochs, patience 8).
set -e
cd /home/apradipta/phd-research
mkdir -p logs/D1scale_244
for F in 1 2; do
  for S in 42 43 44 45 46; do
    OUT="results/investigation/regime_memory/D1scale_244_fold${F}_seed${S}.json"
    LOG="logs/D1scale_244/fold${F}_seed${S}.log"
    if [ -f "$OUT" ]; then
      echo "[$(date +%H:%M:%S)] skip fold=$F seed=$S"
      continue
    fi
    echo "[$(date +%H:%M:%S)] D1scale 244 fold=$F seed=$S"
    python3 -u -m src.investigation.regime_memory.train \
      --fold "$F" --seed "$S" --K 4 --no-regime-token \
      --memory-tokens 0 --single-graph \
      --num-prototypes 16 --sparsity-weight 0.01 \
      --huber-delta 1.0 --vol-weight \
      --max-tickers 244 \
      --use-set-model --set-temporal-encoder gru \
      --hidden-dim 256 --epochs 25 --patience 8 \
      --output "$OUT" > "$LOG" 2>&1
  done
done
echo "[$(date +%H:%M:%S)] DONE"
