#!/bin/bash -l
#SBATCH --job-name=model-eval-baselines
#SBATCH --partition=gpu
#SBATCH --account=hz79
#SBATCH --qos=standard
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --mem=48G
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

REPO="${REPO:-$HOME/phd-research}"
VENV="${VENV:-$HOME/phd-research-gpu-env}"
cd "$REPO"
module load bright 2>/dev/null || true
module load python3 2>/dev/null || true
source "$VENV/bin/activate"

mkdir -p logs/baselines_universal_two_regime_val \
         results/baselines_universal_two_regime_val/factorvae \
         results/baselines_universal_two_regime_val/master \
         results/baselines_universal_two_regime_val/dystage

echo "host=$(hostname) start=$(date -Is)"
nvidia-smi | head -8 || true

# Tier 1 baselines on the universal panel + two-regime val protocol.
# Same panel (lattice_native 26-col), same fold protocol (val=2017H2+2018H2),
# same eval (cs_mse_loss, per-day IC, ndcg@k, cohort IC) as the headline
# RAG-STAR sweep at results/rag_star_universe_v2_two_regime_val/.
#
# 3 baselines x 5 folds x 5 seeds = 75 runs. Walltime 24h sized for ~15min/run avg.

run_baseline() {
  local NAME="$1"  # factorvae|master|dystage
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

# Order: cycle baselines per (fold, seed) so partial completion has all 3 covered.
for FOLD in 1 2 3 4 5; do
  for S in 42 43 44 45 46; do
    for NAME in factorvae master dystage; do
      run_baseline "$NAME" "$FOLD" "$S"
    done
  done
done

echo "DONE $(date -Is)"
