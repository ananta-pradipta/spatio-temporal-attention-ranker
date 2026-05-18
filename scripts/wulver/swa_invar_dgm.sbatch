#!/bin/bash -l
#SBATCH --job-name=model-eval
#SBATCH --partition=gpu
#SBATCH --account=hz79
#SBATCH --qos=standard
#SBATCH --time=12:00:00
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

mkdir -p logs/swa_invar_dgm \
         results/swa_invar_dgm

echo "host=$(hostname) start=$(date -Is)"
nvidia-smi | head -8 || true

# SWA-InVAR-DGM EXPERIMENT-ONLY variant on the universal panel +
# two-regime val protocol. SWA-InVAR core + RAG-STAR day-memory +
# RAG-STAR macro-gated rate-sensitivity graph-bias. Same panel
# (lattice_native 26-col), same fold protocol (val=2017H2+2018H2),
# same eval (cs_mse_loss, per-day IC, ndcg@k, cohort IC) as the
# headline RAG-STAR / SWA-InVAR sweeps. NO predictions npz is written
# (home storage is nearly full).
#
# 5 folds x 5 seeds = 25 runs. Walltime 12h. skip-if-done = json exists
# and contains "epoch".

run() {
  local FOLD="$1"; local SEED="$2"
  local OUT_DIR="results/swa_invar_dgm"
  local OUT="${OUT_DIR}/fold${FOLD}_seed${SEED}.json"
  if [ -f "$OUT" ] && [ "$(grep -c '"epoch"' "$OUT")" -ge 1 ]; then
    echo "[$(date +%H:%M:%S)] skip $OUT"
    return
  fi
  local LF="logs/swa_invar_dgm/swa_invar_dgm_fold${FOLD}_seed${SEED}.log"
  echo "[$(date +%H:%M:%S)] swa_invar_dgm fold=$FOLD seed=$SEED"
  PYTHONPATH=$PWD python3 -u -m src.baselines.train_swa_invar_dgm_v2 \
    --fold "$FOLD" --seed "$SEED" \
    --panel_kind lattice_native \
    --two_regime_val \
    --output_dir "$OUT_DIR" \
    > "$LF" 2>&1
}

for FOLD in 1 2 3 4 5; do
  for S in 42 43 44 45 46; do
    run "$FOLD" "$S"
  done
done

echo "DONE $(date -Is)"
