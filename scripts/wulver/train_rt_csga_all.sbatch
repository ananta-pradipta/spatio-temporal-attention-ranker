#!/bin/bash -l
#SBATCH --job-name=rt-csga-all
#SBATCH --partition=gpu
#SBATCH --account=hz79
#SBATCH --qos=standard
#SBATCH --time=06:00:00
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

echo "=== RT-CSGA: regime-tempered cross-sectional graph attention ==="
echo "host=$(hostname)  start=$(date -Is)"
nvidia-smi | head -20 || true

REPO="${REPO:-$HOME/phd-research}"
VENV="${VENV:-$HOME/phd-research-gpu-env}"
cd "$REPO"

module load bright 2>/dev/null || true
module load python3 2>/dev/null || true
source "$VENV/bin/activate"
python3 -c "import torch; print('torch', torch.__version__, 'cuda:', torch.cuda.is_available())"

# Phase 1: headline RT-CSGA+Aux on all 3 folds x 5 seeds.
HEADLINE_CONFIG="configs/rt_csga.yaml"
HEADLINE_OUT="results/rt_csga"
HEADLINE_LOG="logs/rt_csga/headline"
mkdir -p "$HEADLINE_OUT" "$HEADLINE_LOG"
for FOLD in 1 2 3; do
  for S in 42 43 44 45 46; do
    OF="$HEADLINE_OUT/fold${FOLD}_seed${S}.json"
    LF="$HEADLINE_LOG/fold${FOLD}_seed${S}.log"
    if [ -f "$OF" ]; then echo "[$(date +%H:%M:%S)] skip $OF"; continue; fi
    echo "[$(date +%H:%M:%S)] HEADLINE fold=$FOLD seed=$S"
    python3 -u -m src.v2.training.train_rt_csga --config "$HEADLINE_CONFIG" --fold "$FOLD" --seed "$S" \
      > "$LF" 2>&1
  done
done

# Phase 2: 4 ablations on fold 3 only x 5 seeds.
for ABL in no_aux fixed_tau random_edges shuffled_regime; do
  CFG="configs/rt_csga_${ABL}.yaml"
  OUT="results/rt_csga_${ABL}"
  LOG="logs/rt_csga/${ABL}"
  mkdir -p "$OUT" "$LOG"
  for S in 42 43 44 45 46; do
    OF="$OUT/fold3_seed${S}.json"
    LF="$LOG/fold3_seed${S}.log"
    if [ -f "$OF" ]; then echo "[$(date +%H:%M:%S)] skip $OF"; continue; fi
    echo "[$(date +%H:%M:%S)] ABLATION $ABL fold=3 seed=$S"
    python3 -u -m src.v2.training.train_rt_csga --config "$CFG" --fold 3 --seed "$S" \
      > "$LF" 2>&1
  done
done

echo "[$(date +%H:%M:%S)] DONE"
echo "end=$(date -Is)"
