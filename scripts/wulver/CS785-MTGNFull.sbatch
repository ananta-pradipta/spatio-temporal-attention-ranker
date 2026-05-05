#!/bin/bash -l
#SBATCH --job-name=CS785-MTGNFull
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

# Single-job sweep: all mode x seed combinations run sequentially under one
# Slurm job id (per user directive 2026-04-13). Keeps queue tidy and logs
# in one place. Parallelism within the job can be added with xargs -P if
# node resources permit; start sequential for clarity.

MODES=(none self_only cross_entity)
SEEDS=(11 22 33 42 55)
EPOCHS=15
PATIENCE=4
MAX_TICKERS=300
START=2020-01-01
END=2022-12-31
HORIZON=5
K=32

echo "=== CS785-MTGNFull  host=$(hostname)  start=$(date -Is) ==="

REPO="${REPO:-$HOME/phd-research}"
VENV="${VENV:-$HOME/phd-research-env}"
cd "$REPO"

module load bright 2>/dev/null || true
module load python3 2>/dev/null || true
# shellcheck disable=SC1091
source "$VENV/bin/activate"

python3 -c "import torch, torch_geometric, faiss; print('deps ok')"

OUT_DIR="results/full_mtgn"
mkdir -p "$OUT_DIR"

for MODE in "${MODES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        OUT_PATH="$OUT_DIR/mtgn_full_${MODE}_seed${SEED}.json"
        echo ""
        echo "--- combo: mode=$MODE  seed=$SEED  $(date -Is) ---"
        python3 -m src.mtgn.training.train_mtgn_full \
            --max-tickers "$MAX_TICKERS" --start "$START" --end "$END" \
            --epochs "$EPOCHS" --horizon-days "$HORIZON" --patience "$PATIENCE" \
            --retrieval-mode "$MODE" --retrieval-k "$K" \
            --seed "$SEED" \
            --output "$OUT_PATH"
        echo "--- wrote $OUT_PATH ---"
    done
done

echo ""
echo "=== CS785-MTGNFull  all combos complete  end=$(date -Is) ==="
ls -la "$OUT_DIR"
