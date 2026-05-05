#!/usr/bin/env bash
# Matched 100-ticker baselines on fold 1 seed 42, for apples-to-apples
# comparison against Combined v2 (MTGN test IC +0.0085).
# Per mtgn-diagnosis-followup-round3.md Priority 1.
set -euo pipefail

MODELS=(ridge lstm gcn gat)
OUT_DIR="results/baselines_n100"
mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/run.log"
: > "$LOG"

echo "=== matched baselines n=100 fold=1 seed=42 start $(date -Iseconds) ===" | tee -a "$LOG"

for MODEL in "${MODELS[@]}"; do
    OUT="$OUT_DIR/${MODEL}_fold1_seed42.json"
    if [[ -s "$OUT" ]]; then
        echo "skip existing: $OUT" | tee -a "$LOG"
        continue
    fi
    echo "--- $MODEL $(date -Iseconds) ---" | tee -a "$LOG"
    python3 -m src.mtgn.training.train_baselines \
        --model "$MODEL" --max-tickers 100 \
        --start 2015-01-01 --end 2022-12-31 \
        --horizon-days 5 --epochs 30 --seed 42 \
        --fold 1 \
        --output "$OUT" 2>&1 | tail -3 | tee -a "$LOG"
done

echo "=== done $(date -Iseconds) ===" | tee -a "$LOG"
