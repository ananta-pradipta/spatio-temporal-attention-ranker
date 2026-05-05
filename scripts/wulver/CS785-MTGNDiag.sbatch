#!/bin/bash -l
#SBATCH --job-name=CS785-MTGNDiag
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
#SBATCH --array=0-5

set -euo pipefail

# Diagnostic sweep: 2 modes x 3 seeds = 6 runs, saves per-day test
# predictions so we can slice by catalyst-window vs calm-period.
MODES=(none cross_entity)
SEEDS=(11 33 55)

IDX=${SLURM_ARRAY_TASK_ID}
MODE_IDX=$(( IDX / ${#SEEDS[@]} ))
SEED_IDX=$(( IDX % ${#SEEDS[@]} ))
MODE=${MODES[$MODE_IDX]}
SEED=${SEEDS[$SEED_IDX]}

echo "=== CS785-MTGNDiag  idx=$IDX  mode=$MODE  seed=$SEED ==="
echo "host=$(hostname)  start=$(date -Is)"

REPO="${REPO:-$HOME/phd-research}"
VENV="${VENV:-$HOME/phd-research-env}"
cd "$REPO"

module load bright 2>/dev/null || true
module load python3 2>/dev/null || true
# shellcheck disable=SC1091
source "$VENV/bin/activate"

python3 -c "import torch, torch_geometric, faiss; print('deps ok')"

OUT_DIR="results/diag_mtgn"
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
