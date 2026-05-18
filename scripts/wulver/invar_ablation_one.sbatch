#!/bin/bash -l
#SBATCH --job-name=model-eval
#SBATCH --partition=gpu
#SBATCH --account=hz79
#SBATCH --qos=standard
#SBATCH --time=04:00:00
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

# One InVAR ablation config per job (parallelism across configs, user
# request 2026-05-16). Each job is still a single sbatch with an internal
# fold x seed loop (not a Slurm array). ABL_TAG names the config and the
# output subdir; ABL_FLAGS holds the single ablation flag (empty for full).
ABL_TAG="${ABL_TAG:?ABL_TAG must be set via --export}"
ABL_FLAGS="${ABL_FLAGS:-}"
OUT_DIR="results/invar_ablation/${ABL_TAG}"
LOG_DIR="logs/invar_ablation/${ABL_TAG}"
mkdir -p "$OUT_DIR" "$LOG_DIR"

echo "host=$(hostname) start=$(date -Is) tag=${ABL_TAG} flags=[${ABL_FLAGS}]"
nvidia-smi | head -8 || true

run() {
  local FOLD="$1"; local SEED="$2"
  local OUT="${OUT_DIR}/fold${FOLD}_seed${SEED}.json"
  if [ -f "$OUT" ] && [ "$(grep -c '"epoch"' "$OUT")" -ge 1 ]; then
    echo "[$(date +%H:%M:%S)] skip $OUT"
    return
  fi
  local LF="${LOG_DIR}/fold${FOLD}_seed${SEED}.log"
  echo "[$(date +%H:%M:%S)] ${ABL_TAG} fold=$FOLD seed=$SEED"
  PYTHONPATH=$PWD python3 -u -m src.baselines.train_invar_stx_v2 \
    --fold "$FOLD" --seed "$SEED" \
    --panel_kind lattice_native \
    --two_regime_val \
    --ablation_tag "$ABL_TAG" \
    ${ABL_FLAGS} \
    --output_dir "$OUT_DIR" \
    > "$LF" 2>&1
}

for FOLD in 1 2 3 4 5; do
  for S in 42 43 44 45 46; do
    run "$FOLD" "$S"
  done
done

echo "DONE ${ABL_TAG} $(date -Is)"
