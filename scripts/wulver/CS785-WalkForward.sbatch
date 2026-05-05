#!/bin/bash -l
#SBATCH --job-name=CS785-WalkForward
#SBATCH --partition=course
#SBATCH --account=2026-spring-cs-785-hz79-adp232
#SBATCH --qos=course
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

set -euo pipefail

# Walk-forward CV across 3 folds for every baseline model.
# Fold 1: train 2019..2020-H1  val 2020-H2  test 2021-H1  (COVID+vaccine recovery)
# Fold 2: train 2019..2020     val 2021-H1  test 2021-H2..2022-H1  (bull->bear)
# Fold 3: train 2019..2021     val 2022-H1  test 2022-H2  (deep bear)
# 5-day embargo at every boundary to prevent label-window leakage.

MODELS=(ridge lstm gcn gat rgcn tgcn)
FOLDS=(1 2 3)
SEEDS=(11 22 33 42 55)

echo "=== CS785-WalkForward  host=$(hostname)  start=$(date -Is) ==="
REPO="${REPO:-$HOME/phd-research}"
VENV="${VENV:-$HOME/phd-research-env}"
cd "$REPO"
module load bright 2>/dev/null || true
module load python3 2>/dev/null || true
source "$VENV/bin/activate"
python3 -c "import torch, torch_geometric; print('deps ok')"

OUT_DIR="results/walkforward"
mkdir -p "$OUT_DIR"

for MODEL in "${MODELS[@]}"; do
    for FOLD in "${FOLDS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            OUT="$OUT_DIR/${MODEL}_fold${FOLD}_seed${SEED}.json"
            echo ""; echo "--- model=$MODEL fold=$FOLD seed=$SEED $(date -Is) ---"
            python3 -m src.mtgn.training.train_baselines \
                --model "$MODEL" --max-tickers 300 \
                --start 2018-01-01 --end 2022-12-31 \
                --horizon-days 5 --epochs 30 --seed "$SEED" \
                --fold "$FOLD" \
                --output "$OUT"
        done
    done
done

echo ""; echo "=== CS785-WalkForward done $(date -Is) ==="
ls -la "$OUT_DIR" | head -30
