#!/bin/bash -l
#SBATCH --job-name=universal-val-f1
#SBATCH --partition=gpu
#SBATCH --account=hz79
#SBATCH --qos=standard
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --mem=32G
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

echo "=== Universal validation: RAG-STAR on S&P 500, fold 1, 5 seeds + corr-only ablation ==="
echo "host=$(hostname)  start=$(date -Is)"
nvidia-smi | head -20 || true

REPO="${REPO:-$HOME/phd-research}"
VENV="${VENV:-$HOME/phd-research-gpu-env}"
cd "$REPO"

module load bright 2>/dev/null || true
module load python3 2>/dev/null || true
source "$VENV/bin/activate"
python3 -c "import torch; print('torch', torch.__version__, 'cuda:', torch.cuda.is_available())"

LOG_BASE="logs/universal_validation"
mkdir -p "$LOG_BASE"

VARIANTS=(
  full
  corr_only
)
FOLD=1

for V in "${VARIANTS[@]}"; do
  for S in 42 43 44 45 46; do
    OUT="results/universal_validation_${V}/fold${FOLD}_seed${S}.json"
    LF="$LOG_BASE/${V}_fold${FOLD}_seed${S}.log"
    if [ -f "$OUT" ]; then
      echo "[$(date +%H:%M:%S)] skip $OUT"
      continue
    fi
    if [ "$V" = "full" ]; then
      CFG="configs/universal_validation_full.yaml"
    else
      CFG="configs/universal_validation_corr_only.yaml"
    fi
    echo "[$(date +%H:%M:%S)] variant=$V fold=$FOLD seed=$S"
    python3 -u -m src.v2.training.train_dow_epistar \
      --config "$CFG" --fold "$FOLD" --seed "$S" > "$LF" 2>&1
  done
done

echo "[$(date +%H:%M:%S)] DONE"
echo "end=$(date -Is)"
