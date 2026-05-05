#!/bin/bash -l
#SBATCH --job-name=model-8-train-all
#SBATCH --partition=gpu
#SBATCH --account=hz79
#SBATCH --qos=standard
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100_10g:1
#SBATCH --mem=24G
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

echo "=== model-8-train-all (epiSTAR-SBP v1: cohort-aug retrieval + dual-pool + IRF + V-REx + alpha-prior) ==="
echo "host=$(hostname)  start=$(date -Is)"
nvidia-smi | head -20 || true

REPO="${REPO:-$HOME/phd-research}"
VENV="${VENV:-$HOME/phd-research-gpu-env}"
cd "$REPO"

module load bright 2>/dev/null || true
module load python3 2>/dev/null || true
if [ ! -d "$VENV" ]; then python3 -m venv "$VENV"; fi
source "$VENV/bin/activate"
pip install --upgrade pip --quiet
if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  pip install --quiet --index-url https://download.pytorch.org/whl/cu121 torch
fi
pip install --quiet -r requirements.txt
python3 -c "import torch; print('torch', torch.__version__, 'cuda:', torch.cuda.is_available())"

OUT_DIR="results/epistar_sbp"
LOG_DIR="logs/epistar_sbp"
mkdir -p "$OUT_DIR" "$LOG_DIR"

CONFIG="configs/epistar_sbp.yaml"
FOLDS=(1 2 3)
SEEDS=(42 43 44 45 46)

for FOLD in "${FOLDS[@]}"; do
  for S in "${SEEDS[@]}"; do
    OUT="$OUT_DIR/fold${FOLD}_seed${S}.json"
    LOG="$LOG_DIR/fold${FOLD}_seed${S}.log"
    if [ -f "$OUT" ]; then
      echo "[$(date +%H:%M:%S)] skip existing fold=$FOLD seed=$S"
      continue
    fi
    echo "[$(date +%H:%M:%S)] epiSTAR-SBP fold=$FOLD seed=$S -> $OUT"
    python3 -u -m src.v2.training.train_epistar_sbp \
      --config "$CONFIG" --fold "$FOLD" --seed "$S" \
      > "$LOG" 2>&1
  done
done

echo "[$(date +%H:%M:%S)] DONE"
echo "end=$(date -Is)"
