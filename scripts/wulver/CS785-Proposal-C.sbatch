#!/bin/bash -l
#SBATCH --job-name=CS785-Proposal-C
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

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

echo "=== CS785-Proposal-C ==="
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
source "$VENV/bin/activate"
pip install --upgrade pip --quiet
if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    pip install --quiet --index-url https://download.pytorch.org/whl/cu121 torch
fi
pip install --quiet -r requirements.txt

python3 -c "import torch; print('torch', torch.__version__, 'cuda:', torch.cuda.is_available())"

OUT_DIR="results/investigation/regime_memory"
LOG_DIR="logs/proposal_C_gpu"
mkdir -p "$OUT_DIR" "$LOG_DIR"

# Two lambda_max values for DANN: 0.3 (mild) and 1.0 (strong).
# 2 folds x 5 seeds x 2 lambdas = 20 runs.

FOLDS=(1 2)
SEEDS=(42 43 44 45 46)

for LAM in 0.3 1.0; do
  LAM_TAG=$(echo "$LAM" | tr . p)
  for F in "${FOLDS[@]}"; do
    for S in "${SEEDS[@]}"; do
      OUT="$OUT_DIR/propC_l${LAM_TAG}_fold${F}_seed${S}.json"
      LOG="$LOG_DIR/propC_l${LAM_TAG}_fold${F}_seed${S}.log"
      if [ -f "$OUT" ]; then continue; fi
      echo "[$(date +%H:%M:%S)] propC lambda=$LAM fold=$F seed=$S"
      python3 -u -m src.investigation.regime_memory.train \
        --fold "$F" --seed "$S" --K 4 --no-regime-token \
        --memory-tokens 0 --single-graph \
        --num-prototypes 16 --sparsity-weight 0.01 \
        --huber-delta 1.0 --vol-weight \
        --dann-lambda-max "$LAM" --dann-hidden 64 \
        --output "$OUT" > "$LOG" 2>&1
    done
  done
done

echo "[$(date +%H:%M:%S)] DONE"
echo "end=$(date -Is)"
