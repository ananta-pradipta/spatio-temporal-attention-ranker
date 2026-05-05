#!/bin/bash -l
#SBATCH --job-name=model-eval-5
#SBATCH --partition=gpu
#SBATCH --account=hz79
#SBATCH --qos=standard
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --mem=24G
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

echo "=== RSR v2 sweep: 3 folds x 5 seeds = 15 runs ==="
echo "host=$(hostname)  start=$(date -Is)"
nvidia-smi | head -20 || true

REPO="${REPO:-$HOME/phd-research}"
VENV="${VENV:-$HOME/phd-research-gpu-env}"
cd "$REPO"

module load bright 2>/dev/null || true
module load python3 2>/dev/null || true
source "$VENV/bin/activate"
python3 -c "import torch; print('torch', torch.__version__, 'cuda:', torch.cuda.is_available())"

# Build the per-fold static relation graphs (idempotent; fast).
# Each fold's graph is constructed using only data available at the fold's
# train-end date (no leakage of post-train ticker classifications).
declare -A FOLD_TRAIN_END=( [1]=2018-12-21 [2]=2020-12-23 [3]=2021-12-23 )
for FOLD in 1 2 3; do
  GRAPH_PATH="data/processed/rsr_relation_graph_fold${FOLD}.pt"
  if [ ! -f "$GRAPH_PATH" ]; then
    echo "[$(date +%H:%M:%S)] building rsr relation graph fold=$FOLD"
    python3 -m src.baselines.build_rsr_relation_graph \
      --fold "$FOLD" --train-end-date "${FOLD_TRAIN_END[$FOLD]}"
  fi
done

LOG_BASE="logs/rsr_v2"
mkdir -p "$LOG_BASE"

for FOLD in 1 2 3; do
  for S in 42 43 44 45 46; do
    OUT="results/baselines_244/rsr_v2/fold${FOLD}_seed${S}.json"
    LF="$LOG_BASE/fold${FOLD}_seed${S}.log"
    if [ -f "$OUT" ]; then
      echo "[$(date +%H:%M:%S)] skip $OUT"
      continue
    fi
    echo "[$(date +%H:%M:%S)] rsr fold=$FOLD seed=$S"
    python3 -u -m src.baselines.train_rsr_v2 --fold "$FOLD" --seed "$S" \
      > "$LF" 2>&1
  done
done

echo "[$(date +%H:%M:%S)] DONE"
echo "end=$(date -Is)"
