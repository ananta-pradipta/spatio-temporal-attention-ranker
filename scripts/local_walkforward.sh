#!/usr/bin/env bash
# Walk-forward CV sweep, run locally (Wulver is in scheduled maintenance
# 2026-04-14..2026-04-16). Identical semantics to scripts/wulver/CS785-WalkForward.sbatch.
set -euo pipefail

MODELS=(ridge lstm gcn gat rgcn tgcn)
FOLDS=(1 2 3)
SEEDS=(11 22 33 42 55)

OUT_DIR="results/walkforward"
mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/run.log"
: > "$LOG"

echo "=== local walk-forward start $(date -Iseconds) ===" | tee -a "$LOG"

for MODEL in "${MODELS[@]}"; do
    for FOLD in "${FOLDS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            OUT="$OUT_DIR/${MODEL}_fold${FOLD}_seed${SEED}.json"
            if [[ -s "$OUT" ]]; then
                echo "skip existing: $OUT" | tee -a "$LOG"
                continue
            fi
            echo "--- $MODEL fold=$FOLD seed=$SEED  $(date -Iseconds) ---" | tee -a "$LOG"
            python3 -m src.mtgn.training.train_baselines \
                --model "$MODEL" --max-tickers 300 \
                --start 2018-01-01 --end 2022-12-31 \
                --horizon-days 5 --epochs 30 --seed "$SEED" \
                --fold "$FOLD" \
                --output "$OUT" 2>&1 | tail -4 | tee -a "$LOG"
        done
    done
done

echo "=== local walk-forward done $(date -Iseconds) ===" | tee -a "$LOG"
