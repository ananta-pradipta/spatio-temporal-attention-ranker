#!/bin/bash -l
#SBATCH --job-name=CS785-MTGNSweep
#SBATCH --partition=course
#SBATCH --account=2026-spring-cs-785-hz79-adp232
#SBATCH --qos=course
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=%x.%A_%a.out
#SBATCH --error=%x.%A_%a.err
#SBATCH --array=0-14

set -euo pipefail

# 15 runs = 3 modes x 5 seeds.
MODES=(none self_only cross_entity)
SEEDS=(11 22 33 42 55)

IDX=${SLURM_ARRAY_TASK_ID}
MODE_IDX=$(( IDX / ${#SEEDS[@]} ))
SEED_IDX=$(( IDX % ${#SEEDS[@]} ))
MODE=${MODES[$MODE_IDX]}
SEED=${SEEDS[$SEED_IDX]}

echo "=== CS785-MTGNSweep  idx=$IDX  mode=$MODE  seed=$SEED ==="
echo "host=$(hostname)  start=$(date -Is)"

REPO="${REPO:-$HOME/phd-research}"
VENV="${VENV:-$HOME/phd-research-env}"

cd "$REPO"

module load bright 2>/dev/null || true
module load python3 2>/dev/null || true

if [ ! -d "$VENV" ]; then
    python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install --upgrade pip --quiet
# CPU-only PyTorch + PyG on the course (non-GPU) partition.
if ! python3 -c "import torch" 2>/dev/null; then
    pip install --quiet --index-url https://download.pytorch.org/whl/cpu torch
fi
if ! python3 -c "import torch_geometric" 2>/dev/null; then
    pip install --quiet torch-geometric
fi
pip install --quiet -r requirements.txt

python3 -c "import torch, torch_geometric, faiss; print('deps ok')"

OUT_DIR="results/sweep_mtgn"
mkdir -p "$OUT_DIR"
OUT_PATH="$OUT_DIR/mtgn_${MODE}_seed${SEED}.json"

python3 -m src.mtgn.training.train_mtgn \
    --max-tickers 300 --start 2020-01-01 --end 2022-12-31 \
    --epochs 15 --horizon-days 5 --patience 4 \
    --retrieval-mode "$MODE" --retrieval-k 32 \
    --seed "$SEED" \
    --output "$OUT_PATH"

echo "wrote $OUT_PATH"
echo "end=$(date -Is)"
