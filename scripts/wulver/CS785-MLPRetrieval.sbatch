#!/bin/bash -l
#SBATCH --job-name=CS785-MLPRetrieval
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

# Single-job sweep: 3 retrieval modes x 3 seeds = 9 runs.
# MLP backbone (no graph, no TGN memory) + episodic retrieval + heads.
# Tests whether retrieval alone adds value over the pure-MLP baseline.

MODES=(none self_only cross_entity)
SEEDS=(11 22 33)

echo "=== CS785-MLPRetrieval  host=$(hostname)  start=$(date -Is) ==="
REPO="${REPO:-$HOME/phd-research}"
VENV="${VENV:-$HOME/phd-research-env}"
cd "$REPO"
module load bright 2>/dev/null || true
module load python3 2>/dev/null || true
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python3 -c "import torch, faiss; print('deps ok')"

OUT_DIR="results/mlp_retrieval"
mkdir -p "$OUT_DIR"

for MODE in "${MODES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        OUT="$OUT_DIR/mlp_retr_${MODE}_seed${SEED}.json"
        echo ""; echo "--- mode=$MODE seed=$SEED $(date -Is) ---"
        python3 -m src.mtgn.training.train_mlp_retrieval \
            --max-tickers 300 --retrieval-mode "$MODE" --retrieval-k 32 \
            --epochs 15 --patience 4 --seed "$SEED" \
            --output "$OUT"
    done
done

echo ""
echo "=== CS785-MLPRetrieval done $(date -Is) ==="
ls -la "$OUT_DIR"
