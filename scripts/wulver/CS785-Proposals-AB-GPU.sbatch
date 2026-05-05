#!/bin/bash -l
#SBATCH --job-name=CS785-Proposals-AB-GPU
#SBATCH --partition=course_gpu
#SBATCH --account=2026-spring-cs-785-hz79-adp232
#SBATCH --qos=course
#SBATCH --time=04:00:00
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

echo "=== CS785-Proposals-AB-GPU ==="
echo "host=$(hostname)  start=$(date -Is)"
nvidia-smi | head -20 || true

REPO="${REPO:-$HOME/phd-research}"
VENV="${VENV:-$HOME/phd-research-gpu-env}"

cd "$REPO"

module load bright 2>/dev/null || true
module load python3 2>/dev/null || true

if [ ! -d "$VENV" ]; then
    python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install --upgrade pip --quiet
if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    pip install --quiet --index-url https://download.pytorch.org/whl/cu121 torch
fi
pip install --quiet -r requirements.txt

python3 -c "import torch; print('torch', torch.__version__, 'cuda available:', torch.cuda.is_available(), 'device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"

OUT_DIR="results/investigation/regime_memory"
LOG_DIR="logs/proposals_AB_gpu"
mkdir -p "$OUT_DIR" "$LOG_DIR"

FOLDS=(1 2)
SEEDS=(42 43 44 45 46)

for F in "${FOLDS[@]}"; do
  for S in "${SEEDS[@]}"; do
    # Proposal B (alone)
    OUT="$OUT_DIR/propB_g24_fold${F}_seed${S}.json"
    LOG="$LOG_DIR/propB_g24_fold${F}_seed${S}.log"
    if [ ! -f "$OUT" ]; then
      echo "[$(date +%H:%M:%S)] propB fold=$F seed=$S"
      python3 -u -m src.investigation.regime_memory.train \
        --fold "$F" --seed "$S" --K 4 --no-regime-token \
        --memory-tokens 0 --single-graph \
        --num-prototypes 16 --sparsity-weight 0.01 \
        --huber-delta 1.0 --vol-weight \
        --group-relative-loss --loss-group-size 24 \
        --output "$OUT" > "$LOG" 2>&1
    fi

    OUT="$OUT_DIR/propAgate_a0p5_h8_fold${F}_seed${S}.json"
    LOG="$LOG_DIR/propAgate_a0p5_h8_fold${F}_seed${S}.log"
    if [ ! -f "$OUT" ]; then
      echo "[$(date +%H:%M:%S)] propAgate fold=$F seed=$S"
      python3 -u -m src.investigation.regime_memory.train \
        --fold "$F" --seed "$S" --K 4 --no-regime-token \
        --memory-tokens 0 --single-graph \
        --num-prototypes 16 --sparsity-weight 0.01 \
        --huber-delta 1.0 --vol-weight \
        --corr-bias-alpha 0.5 --corr-bias-gate-hidden 8 \
        --output "$OUT" > "$LOG" 2>&1
    fi

    OUT="$OUT_DIR/propAB_fold${F}_seed${S}.json"
    LOG="$LOG_DIR/propAB_fold${F}_seed${S}.log"
    if [ ! -f "$OUT" ]; then
      echo "[$(date +%H:%M:%S)] propAB fold=$F seed=$S"
      python3 -u -m src.investigation.regime_memory.train \
        --fold "$F" --seed "$S" --K 4 --no-regime-token \
        --memory-tokens 0 --single-graph \
        --num-prototypes 16 --sparsity-weight 0.01 \
        --huber-delta 1.0 --vol-weight \
        --corr-bias-alpha 0.5 --corr-bias-gate-hidden 8 \
        --group-relative-loss --loss-group-size 24 \
        --output "$OUT" > "$LOG" 2>&1
    fi
  done
done

echo "[$(date +%H:%M:%S)] DONE"
echo "end=$(date -Is)"
