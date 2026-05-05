#!/bin/bash -l
#SBATCH --job-name=CS785-MTGNTwoStage
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

# Single-job sweep (per memory: no Slurm arrays for sweeps).
# Two-stage training ablation: 2 retrieval modes x 5 seeds, with
# 5-epoch backbone warmup then 10 epochs of head-only training.
# Compares cleanly against the single-stage MTGNFull sweep already
# committed (results/full_mtgn/).

MODES=(self_only cross_entity)
SEEDS=(11 22 33 42 55)
EPOCHS=15
WARMUP=5
PATIENCE=4
MAX_TICKERS=300
START=2020-01-01
END=2022-12-31
HORIZON=5
K=32

echo "=== CS785-MTGNTwoStage  host=$(hostname)  start=$(date -Is) ==="

REPO="${REPO:-$HOME/phd-research}"
VENV="${VENV:-$HOME/phd-research-env}"
cd "$REPO"

module load bright 2>/dev/null || true
module load python3 2>/dev/null || true
# shellcheck disable=SC1091
source "$VENV/bin/activate"

python3 -c "import torch, torch_geometric, faiss; print('deps ok')"

OUT_DIR="results/twostage_mtgn"
mkdir -p "$OUT_DIR"

for MODE in "${MODES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        OUT_PATH="$OUT_DIR/mtgn_full_${MODE}_seed${SEED}_warmup${WARMUP}.json"
        echo ""
        echo "--- combo: mode=$MODE  seed=$SEED  warmup=$WARMUP  $(date -Is) ---"
        python3 -m src.mtgn.training.train_mtgn_full \
            --max-tickers "$MAX_TICKERS" --start "$START" --end "$END" \
            --epochs "$EPOCHS" --horizon-days "$HORIZON" --patience "$PATIENCE" \
            --backbone-warmup-epochs "$WARMUP" \
            --retrieval-mode "$MODE" --retrieval-k "$K" \
            --seed "$SEED" \
            --output "$OUT_PATH"
    done
done

echo ""
echo "=== CS785-MTGNTwoStage  done  end=$(date -Is) ==="
ls -la "$OUT_DIR"
