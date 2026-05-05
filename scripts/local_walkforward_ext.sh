#!/usr/bin/env bash
# Walk-forward CV sweep on the EXTENDED 2015-2022 panel.
# Writes to a separate output directory so it doesn't clobber the
# 2019-2022 sweep in results/walkforward/.
set -euo pipefail

MODELS=(ridge lstm gcn gat rgcn tgcn)
FOLDS=(1 2 3)
SEEDS=(11 22 33 42 55)

OUT_DIR="results/walkforward_2015"
mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/run.log"
: > "$LOG"

echo "=== local walk-forward (2015-2022) start $(date -Iseconds) ===" | tee -a "$LOG"

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
                --start 2015-01-01 --end 2022-12-31 \
                --horizon-days 5 --epochs 30 --seed "$SEED" \
                --fold "$FOLD" \
                --output "$OUT" 2>&1 | tail -4 | tee -a "$LOG"
        done
    done
done

echo "=== local walk-forward (2015-2022) done $(date -Iseconds) ===" | tee -a "$LOG"
