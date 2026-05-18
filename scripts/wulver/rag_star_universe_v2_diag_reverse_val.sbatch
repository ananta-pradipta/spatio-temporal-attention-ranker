#!/bin/bash -l
#SBATCH --job-name=model-eval-rev
#SBATCH --partition=gpu
#SBATCH --account=hz79
#SBATCH --qos=standard
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --mem=32G
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

mkdir -p logs/rag_star_universe_v2_diag_rev results/rag_star_universe_v2_diag_rev

echo "host=$(hostname) start=$(date -Is)"
nvidia-smi | head -8 || true

# Diagnostic: reverse-time validation. Val is fixed at calendar 2017 across
# all folds; train is two-segment (pre-2017 + post-2017 up to fd.train_end);
# test is identical to the canonical sweep. Same config, same panel, same
# seed as canonical — only --reverse_time_val differs.
#
# Hypothesis (concrete): F2 baseline test_ic = -0.0265 because val (H1 2021)
# is regime-correlated with the regime that flips during test (Fed pivot Nov
# 2021 → growth crash through 2022). With val at 2017 (calm-bull, low VIX,
# no major regime break), the val/test regime correlation drops by
# construction. If the diagnosis is right, F2 test_ic should move toward
# zero or flip positive. F3 is the inverse case (baseline -0.06 val_ic that
# helped test_ic) — so under reverse val, F3 may either improve, stay flat,
# or DROP, depending on whether the original anti-correlated val signal was
# actually selecting a useful checkpoint.

run_revval() {
  local FOLD="$1"; local SEED="$2"
  local TAG="fold${FOLD}_seed${SEED}_revval"
  local OUT_DIR="results/rag_star_universe_v2_diag_rev/${TAG}"
  local LF="logs/rag_star_universe_v2_diag_rev/${TAG}.log"
  echo "[$(date +%H:%M:%S)] diag-rev fold=$FOLD seed=$SEED reverse_time_val=true"
  PYTHONPATH=$PWD python3 -u -m src.v2.training.train_dow_epistar \
    --config configs/rag_star_universe_v2.yaml \
    --fold "$FOLD" --seed "$SEED" \
    --reverse_time_val \
    --output_dir "$OUT_DIR" \
    > "$LF" 2>&1
}

run_fixed_baseline() {
  local FOLD="$1"; local SEED="$2"
  local TAG="fold${FOLD}_seed${SEED}_idxfix"
  local OUT_DIR="results/rag_star_universe_v2_diag_rev/${TAG}"
  local LF="logs/rag_star_universe_v2_diag_rev/${TAG}.log"
  echo "[$(date +%H:%M:%S)] idxfix-baseline fold=$FOLD seed=$SEED"
  PYTHONPATH=$PWD python3 -u -m src.v2.training.train_dow_epistar \
    --config configs/rag_star_universe_v2.yaml \
    --fold "$FOLD" --seed "$SEED" \
    --output_dir "$OUT_DIR" \
    > "$LF" 2>&1
}

# Step 1: re-run F2 + F3 baseline with the index fix but NO reverse-val.
# Tells us whether fixing the IPO/duration column indexing alone changes
# the canonical results (orthogonal effect from reverse-val).
run_fixed_baseline 2 42
run_fixed_baseline 3 42

# Step 2: F2 + F3 with index fix AND reverse-val (val=2017).
# Comparing step 2 vs step 1 isolates the reverse-val effect on the
# correctly-configured model.
run_revval 2 42
run_revval 3 42

echo "DONE $(date -Is)"
