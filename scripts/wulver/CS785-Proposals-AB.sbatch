#!/bin/bash -l
#SBATCH --job-name=CS785-Proposals-AB
#SBATCH --partition=course
#SBATCH --account=2026-spring-cs-785-hz79-adp232
#SBATCH --qos=course
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

set -euo pipefail

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

echo "=== CS785-Proposals-AB ==="
echo "host=$(hostname)  start=$(date -Is)  threads=${OMP_NUM_THREADS}"

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
if ! python3 -c "import torch" 2>/dev/null; then
    pip install --quiet --index-url https://download.pytorch.org/whl/cpu torch
fi
pip install --quiet -r requirements.txt

python3 -c "import torch; print('torch ok', torch.__version__)"

OUT_DIR="results/investigation/regime_memory"
LOG_DIR="logs/proposals_AB"
mkdir -p "$OUT_DIR" "$LOG_DIR"

# Variant B: group-relative loss with top-24 correlation peers.
# Variant A-gated: correlation bias alpha=0.5 with learnable 8-hidden-unit gate.
# Variant A+B: both stacked.
# 2 folds x 5 seeds per variant = 10 runs per variant.

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

    # Proposal A-gated (alone)
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

    # Proposal A + B combined
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
