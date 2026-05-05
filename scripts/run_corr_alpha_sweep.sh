#!/bin/bash
# Proposal A: correlation-aware attention alpha sweep.
# REM iter 3B architecture + robust loss + corr-bias-alpha.
# Folds 1 and 2 only (fold 3 reserved). 5 seeds each.
set -e
cd /home/apradipta/phd-research
mkdir -p logs/corr_alpha_sweep
for ALPHA in 0.25 0.5 1.0; do
  TAG=$(echo "$ALPHA" | tr . p)
  for F in 1 2; do
    for S in 42 43 44 45 46; do
      OUT="results/investigation/regime_memory/corrA_a${TAG}_fold${F}_seed${S}.json"
      LOG="logs/corr_alpha_sweep/a${TAG}_fold${F}_seed${S}.log"
      if [ -f "$OUT" ]; then
        echo "[$(date +%H:%M:%S)] skip (exists) alpha=$ALPHA fold=$F seed=$S"
        continue
      fi
      echo "[$(date +%H:%M:%S)] alpha=$ALPHA fold=$F seed=$S"
      python3 -u -m src.investigation.regime_memory.train \
        --fold "$F" --seed "$S" --K 4 --no-regime-token \
        --memory-tokens 0 --single-graph \
        --num-prototypes 16 --sparsity-weight 0.01 \
        --huber-delta 1.0 --vol-weight \
        --corr-bias-alpha "$ALPHA" \
        --output "$OUT" > "$LOG" 2>&1
    done
  done
done
echo "[$(date +%H:%M:%S)] DONE"
