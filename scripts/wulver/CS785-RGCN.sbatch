#!/bin/bash -l
#SBATCH --job-name=CS785-RGCN
#SBATCH --partition=course
#SBATCH --account=2026-spring-cs-785-hz79-adp232
#SBATCH --qos=course
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

set -euo pipefail

# Heterogeneous RGCN baseline: 3 edge types (trial co-participation,
# StockTwits co-mention, sector) over the enriched 22-feature panel.
# Same panel / split / loss / metrics as Ridge/LSTM/GCN/GAT for direct
# comparison of the 'heterogeneous edges' effect in isolation.

MODELS=(rgcn)
SEEDS=(11 22 33 42 55)

echo "=== CS785-RGCN  host=$(hostname)  start=$(date -Is) ==="
REPO="${REPO:-$HOME/phd-research}"
VENV="${VENV:-$HOME/phd-research-env}"
cd "$REPO"
module load bright 2>/dev/null || true
module load python3 2>/dev/null || true
source "$VENV/bin/activate"
python3 -c "import torch, torch_geometric; print('deps ok')"

OUT_DIR="results/baselines"
mkdir -p "$OUT_DIR"

for MODEL in "${MODELS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        OUT="$OUT_DIR/${MODEL}_seed${SEED}.json"
        echo ""; echo "--- model=$MODEL seed=$SEED $(date -Is) ---"
        python3 -m src.mtgn.training.train_baselines \
            --model "$MODEL" --max-tickers 300 \
            --start 2018-01-01 --end 2022-12-31 \
            --horizon-days 5 --epochs 30 --seed "$SEED" \
            --output "$OUT"
    done
done

echo ""; echo "=== CS785-RGCN done $(date -Is) ==="
ls -la "$OUT_DIR"
