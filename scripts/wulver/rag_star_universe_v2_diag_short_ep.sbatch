#!/bin/bash -l
#SBATCH --job-name=model-eval-diag
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

mkdir -p logs/rag_star_universe_v2_diag results/rag_star_universe_v2_diag

echo "host=$(hostname) start=$(date -Is)"
nvidia-smi | head -8 || true

# Diagnostic: test the val-regime overfit hypothesis on F2 (and F5).
# Same config / panel / seed as the canonical sweep, but two knobs vary:
#   --epochs N          cap at N epochs (also disables early-stop)
#   --use_final_state   evaluate last-epoch state, NOT best-val state
# Hypothesis: test IC on F2 is negative because best-val selection picks a
# checkpoint over-aligned to H1 2021 (the val window). If we evaluate the
# state at epoch 1-3 (before val_ic peaks) OR skip best-val selection on a
# longer run, test IC should move closer to zero or flip positive.

run_one() {
  local FOLD="$1"; local SEED="$2"; local EPOCHS="$3"
  local TAG="fold${FOLD}_seed${SEED}_ep${EPOCHS}_finalstate"
  local OUT_DIR="results/rag_star_universe_v2_diag/${TAG}"
  local LF="logs/rag_star_universe_v2_diag/${TAG}.log"
  echo "[$(date +%H:%M:%S)] diag fold=$FOLD seed=$SEED epochs=$EPOCHS use_final_state=true"
  PYTHONPATH=$PWD python3 -u -m src.v2.training.train_dow_epistar \
    --config configs/rag_star_universe_v2.yaml \
    --fold "$FOLD" --seed "$SEED" \
    --epochs "$EPOCHS" \
    --use_final_state \
    --output_dir "$OUT_DIR" \
    > "$LF" 2>&1
}

# F2 sweep: 1, 2, 3, 5 epochs (val_ic peaks at epoch 5 with 0.086, so this
# brackets pre-peak vs at-peak with no best-val selection).
for EP in 1 2 3 5; do
  run_one 2 42 "$EP"
done

# F5 with same diagnostic: epochs=3 only (the val_ic also peaks around
# epoch 5 there, so 3 should sit pre-peak).
run_one 5 42 3

echo "DONE $(date -Is)"
