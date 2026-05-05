#!/bin/bash -l
#SBATCH --job-name=model-eval-4
#SBATCH --partition=gpu
#SBATCH --account=hz79
#SBATCH --qos=standard
#SBATCH --time=05:00:00
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

echo "=== MERA v2 sweep: 3 folds x 5 seeds = 15 runs ==="
echo "host=$(hostname)  start=$(date -Is)"
nvidia-smi | head -20 || true

REPO="${REPO:-$HOME/phd-research}"
VENV="${VENV:-$HOME/phd-research-gpu-env}"
cd "$REPO"

module load bright 2>/dev/null || true
module load python3 2>/dev/null || true
source "$VENV/bin/activate"
python3 -c "import torch; print('torch', torch.__version__, 'cuda:', torch.cuda.is_available())"

LOG_BASE="logs/mera_v2"
mkdir -p "$LOG_BASE"

# Anti-overfit hyperparameters for our 244-ticker / ~990-day panel
# (paper used CSI 300/500/1000 with much more data). See
# src/baselines/train_mera_v2.py defaults; these env vars override.
PHASE1_EPOCHS="${PHASE1_EPOCHS:-8}"
PHASE2_LR="${PHASE2_LR:-3e-5}"
PHASE2_PATIENCE="${PHASE2_PATIENCE:-1}"

for FOLD in 1 2 3; do
  for S in 42 43 44 45 46; do
    OUT="results/baselines_244/mera_v2/fold${FOLD}_seed${S}.json"
    LF="$LOG_BASE/fold${FOLD}_seed${S}.log"
    if [ -f "$OUT" ]; then
      echo "[$(date +%H:%M:%S)] skip $OUT"
      continue
    fi
    echo "[$(date +%H:%M:%S)] mera fold=$FOLD seed=$S " \
         "(phase1_epochs=$PHASE1_EPOCHS phase2_lr=$PHASE2_LR phase2_patience=$PHASE2_PATIENCE)"
    python3 -u -m src.baselines.train_mera_v2 --fold "$FOLD" --seed "$S" \
      --phase1_epochs "$PHASE1_EPOCHS" \
      --phase2_lr "$PHASE2_LR" \
      --phase2_patience "$PHASE2_PATIENCE" \
      > "$LF" 2>&1
  done
done

echo "[$(date +%H:%M:%S)] DONE"
echo "end=$(date -Is)"
