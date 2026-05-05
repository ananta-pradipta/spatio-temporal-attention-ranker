#!/bin/bash -l
#SBATCH --job-name=csid_v2_fix1
#SBATCH --partition=gpu
#SBATCH --account=hz79
#SBATCH --qos=standard
#SBATCH --time=04:00:00
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

echo "=== CSID v2 Fix 1: gate_init_bias -3.0 -> -1.5 (sigmoid 0.18 init) ==="
echo "host=$(hostname)  start=$(date -Is)"
nvidia-smi | head -20 || true

REPO="${REPO:-$HOME/phd-research}"
VENV="${VENV:-$HOME/phd-research-gpu-env}"
cd "$REPO"

module load bright 2>/dev/null || true
module load python3 2>/dev/null || true
source "$VENV/bin/activate"
python3 -c "import torch; print('torch', torch.__version__, 'cuda:', torch.cuda.is_available())"

CONFIG="configs/csid_v2_fix1.yaml"
OUT="results/csid_v2_fix1"
LOG="logs/csid_v2_fix1"
mkdir -p "$OUT" "$LOG"

for FOLD in 1 2 3; do
  for S in 42 43 44 45 46; do
    OF="$OUT/fold${FOLD}_seed${S}.json"
    LF="$LOG/fold${FOLD}_seed${S}.log"
    if [ -f "$OF" ]; then echo "[$(date +%H:%M:%S)] skip $OF"; continue; fi
    echo "[$(date +%H:%M:%S)] CSID-v2-Fix1 fold=$FOLD seed=$S"
    python3 -u -m src.v2.training.train_csid --config "$CONFIG" --fold "$FOLD" --seed "$S" \
      > "$LF" 2>&1
  done
done

echo "[$(date +%H:%M:%S)] DONE"
echo "end=$(date -Is)"
