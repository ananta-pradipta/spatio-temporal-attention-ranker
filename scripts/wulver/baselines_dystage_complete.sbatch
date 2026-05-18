#!/bin/bash -l
#SBATCH --job-name=model-eval
#SBATCH --partition=gpu
#SBATCH --account=hz79
#SBATCH --qos=standard
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=96G
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
# DySTAGE at 600 tickers is GPU-memory heavy; reduce fragmentation OOMs.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO="${REPO:-$HOME/phd-research}"
VENV="${VENV:-$HOME/phd-research-gpu-env}"
cd "$REPO"
module load bright 2>/dev/null || true
module load python3 2>/dev/null || true
source "$VENV/bin/activate"

mkdir -p logs/baselines_universal_two_regime_val \
         results/baselines_universal_two_regime_val/dystage

echo "host=$(hostname) start=$(date -Is)"
nvidia-smi | head -8 || true

# Completes DySTAGE on the universal panel + two-regime val so it can be
# added as a sixth baseline. IDENTICAL flags to the canonical Tier-1 runner
# (baselines_universal_two_regime_val_sweep.sbatch): same panel_kind,
# two_regime_val, folds, seeds, output dir, eval harness as FactorVAE /
# MASTER / the headline RAG-STAR. Skip-if-done leaves the existing 10 cells
# untouched and fills the missing 15 (folds 3-5 x seeds 42-46).
# 1 baseline x 5 folds x 5 seeds = 25 runs (10 skipped, ~15 to run).

run_baseline() {
  local NAME="$1"
  local FOLD="$2"; local SEED="$3"
  local OUT_DIR="results/baselines_universal_two_regime_val/${NAME}"
  local OUT="${OUT_DIR}/fold${FOLD}_seed${SEED}.json"
  if [ -f "$OUT" ] && [ "$(grep -c '"epoch"' "$OUT")" -ge 1 ]; then
    echo "[$(date +%H:%M:%S)] skip $OUT"
    return
  fi
  local LF="logs/baselines_universal_two_regime_val/${NAME}_fold${FOLD}_seed${SEED}.log"
  echo "[$(date +%H:%M:%S)] ${NAME} fold=$FOLD seed=$SEED"
  PYTHONPATH=$PWD python3 -u -m "src.baselines.train_${NAME}_v2" \
    --fold "$FOLD" --seed "$SEED" \
    --panel_kind lattice_native \
    --two_regime_val \
    --output_dir "$OUT_DIR" \
    > "$LF" 2>&1
}

for FOLD in 1 2 3 4 5; do
  for S in 42 43 44 45 46; do
    run_baseline dystage "$FOLD" "$S"
  done
done

echo "DONE $(date -Is)"
