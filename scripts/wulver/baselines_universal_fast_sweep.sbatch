#!/bin/bash -l
#SBATCH --job-name=model-eval-fast
#SBATCH --partition=gpu
#SBATCH --account=hz79
#SBATCH --qos=standard
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --mem=48G
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

REPO="${REPO:-$HOME/phd-research}"
VENV="${VENV:-$HOME/phd-research-gpu-env}"
cd "$REPO"
source "$VENV/bin/activate"

mkdir -p logs/baselines_universal_two_regime_val \
         results/baselines_universal_two_regime_val/factorvae \
         results/baselines_universal_two_regime_val/master

echo "host=$(hostname) start=$(date -Is)"
nvidia-smi | head -8 || true

# Fast baselines (FactorVAE, MASTER) on universal panel + two-regime val.
# DyStAGE excluded because it's blocked on a one-time graph-cache rebuild
# in the sibling job 1029709 (~2-3h). This sbatch runs the other two
# baselines in parallel so we get F1-F5 results for FactorVAE and MASTER
# while DyStAGE is busy.
#
# 2 baselines x 5 folds x 5 seeds = 50 runs. Skip-if-done so it cooperates
# with 1029709's writes.

run_baseline() {
  local NAME="$1"  # factorvae|master
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

# Order: cycle MASTER + FACTORVAE per (fold, seed) so completion is balanced.
for FOLD in 1 2 3 4 5; do
  for S in 42 43 44 45 46; do
    for NAME in factorvae master; do
      run_baseline "$NAME" "$FOLD" "$S"
    done
  done
done

echo "DONE $(date -Is)"
