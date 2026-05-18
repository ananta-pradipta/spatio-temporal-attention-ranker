#!/bin/bash -l
#SBATCH --job-name=model-eval
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
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

REPO="${REPO:-$HOME/phd-research}"
VENV="${VENV:-$HOME/phd-research-gpu-env}"
cd "$REPO"
module load bright 2>/dev/null || true
module load python3 2>/dev/null || true
source "$VENV/bin/activate"

# F2-targeted macro feature experiment (pre-registered 2026-05-17).
# IDENTICAL to invar_clpretrain.sbatch (active/default InVAR = bankless +
# regime-contrastive pretrain); the ONLY changed variable is the macro
# parquet, which now carries 4 net-new F2 features (curvature_butterfly,
# move_proxy, vix_term_slope, stockbond_corr60) -> 32 cols vs the 28-col
# baseline in results/invar_clpretrain. Separate OUT_DIR so the +0.0225
# headline result is never overwritten. One job per FOLD, internal seed
# loop (not a Slurm array). a100_20g small slice (must not take a big GPU).
FOLD="${FOLD:?FOLD must be set via --export=FOLD=<1..5>}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-10}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-10}"
OUT_DIR="results/invar_clpretrain_f2feat"
LOG_DIR="logs/invar_clpretrain_f2feat"
CKPT="${OUT_DIR}/_ckpt/fold${FOLD}_encoder.pt"
mkdir -p "$OUT_DIR" "$OUT_DIR/_ckpt" "$LOG_DIR"

echo "host=$(hostname) start=$(date -Is) fold=${FOLD}"
nvidia-smi | head -8 || true

# ---- STAGE 1: pretrain ONCE for this fold (seed 42, shared across all
# finetune seeds). Skip if the encoder checkpoint already exists. ----
if [ -f "$CKPT" ]; then
  echo "[$(date +%H:%M:%S)] skip stage-1 (ckpt exists: $CKPT)"
else
  PLF="${LOG_DIR}/fold${FOLD}_pretrain.log"
  echo "[$(date +%H:%M:%S)] stage-1 pretrain fold=$FOLD"
  PYTHONPATH=$PWD python3 -u -m src.baselines.train_invar_clpretrain_v2 \
    --fold "$FOLD" --seed 42 \
    --panel_kind lattice_native \
    --two_regime_val \
    --pretrain_only \
    --pretrain_epochs "$PRETRAIN_EPOCHS" \
    --output_dir "$OUT_DIR" \
    > "$PLF" 2>&1
fi

# ---- STAGE 2: finetune all 5 seeds against the single fold checkpoint.
# Skip-if-done: a JSON with at least one "epoch" entry already exists. ----
run_finetune() {
  local SEED="$1"
  local OUT="${OUT_DIR}/fold${FOLD}_seed${SEED}.json"
  if [ -f "$OUT" ] && [ "$(grep -c '"epoch"' "$OUT")" -ge 1 ]; then
    echo "[$(date +%H:%M:%S)] skip $OUT"
    return
  fi
  local LF="${LOG_DIR}/fold${FOLD}_seed${SEED}.log"
  echo "[$(date +%H:%M:%S)] stage-2 finetune fold=$FOLD seed=$SEED"
  PYTHONPATH=$PWD python3 -u -m src.baselines.train_invar_clpretrain_v2 \
    --fold "$FOLD" --seed "$SEED" \
    --panel_kind lattice_native \
    --two_regime_val \
    --skip_pretrain \
    --finetune_epochs "$FINETUNE_EPOCHS" \
    --output_dir "$OUT_DIR" \
    > "$LF" 2>&1
}

for S in 42 43 44 45 46; do
  run_finetune "$S"
done

echo "DONE fold=${FOLD} $(date -Is)"
