#!/bin/bash -l
#SBATCH --job-name=model-eval
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

# InVAR component-ablation sweep (EXPERIMENT-ONLY; NOT a paper
# baseline). One sbatch, single job, small GPU slice (a100_20g) ON
# PURPOSE so the paper-critical jobs keep the big/40g GPUs. Universal
# panel (lattice_native 26-col) + two-regime val protocol, identical
# fold/eval to the headline RAG-STAR / SWA-InVAR sweeps. Each tag
# disables EXACTLY ONE component; A0_full is the unmodified InVAR and
# is run FIRST so we can verify it reproduces the 5-fold mean +0.0211
# early. NO predictions npz (home storage nearly full). skip-if-done =
# json exists and contains "epoch".
#
# 10 configs x 5 folds x 5 seeds = 250 runs. Walltime 24h.

# Config tag -> ablation flag string. A0_full="" is the full model.
TAGS=(A0_full A1_no_temporal_encoder A2_no_spatial A3_no_day_memory \
      A4_no_macro_film A5_no_macro_head A6_no_retrieval_bank A7_no_swa \
      A8_random_retrieval A9_shuffle_macro)
declare -A FLAGS
FLAGS[A0_full]=""
FLAGS[A1_no_temporal_encoder]="--abl_no_temporal_encoder"
FLAGS[A2_no_spatial]="--abl_no_spatial"
FLAGS[A3_no_day_memory]="--abl_no_day_memory"
FLAGS[A4_no_macro_film]="--abl_no_macro_film"
FLAGS[A5_no_macro_head]="--abl_no_macro_head"
FLAGS[A6_no_retrieval_bank]="--abl_no_retrieval_bank"
FLAGS[A7_no_swa]="--abl_no_swa"
FLAGS[A8_random_retrieval]="--abl_random_retrieval"
FLAGS[A9_shuffle_macro]="--abl_shuffle_macro"

for TAG in "${TAGS[@]}"; do
  mkdir -p "logs/invar_ablation/${TAG}" "results/invar_ablation/${TAG}"
done

echo "host=$(hostname) start=$(date -Is)"
nvidia-smi | head -8 || true

run() {
  local TAG="$1"; local FOLD="$2"; local SEED="$3"
  local OUT_DIR="results/invar_ablation/${TAG}"
  local OUT="${OUT_DIR}/fold${FOLD}_seed${SEED}.json"
  if [ -f "$OUT" ] && [ "$(grep -c '"epoch"' "$OUT")" -ge 1 ]; then
    echo "[$(date +%H:%M:%S)] skip $OUT"
    return
  fi
  local LF="logs/invar_ablation/${TAG}/invar_abl_${TAG}_fold${FOLD}_seed${SEED}.log"
  echo "[$(date +%H:%M:%S)] invar_abl tag=$TAG fold=$FOLD seed=$SEED"
  PYTHONPATH=$PWD python3 -u -m src.baselines.train_invar_stx_v2 \
    --fold "$FOLD" --seed "$SEED" \
    --panel_kind lattice_native \
    --two_regime_val \
    ${FLAGS[$TAG]} \
    --output_dir "$OUT_DIR" \
    > "$LF" 2>&1
}

# Config-major so A0_full (the unmodified InVAR) finishes first.
for TAG in "${TAGS[@]}"; do
  for FOLD in 1 2 3 4 5; do
    for S in 42 43 44 45 46; do
      run "$TAG" "$FOLD" "$S"
    done
  done
done

echo "DONE $(date -Is)"
