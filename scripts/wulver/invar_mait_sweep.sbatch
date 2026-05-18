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
#SBATCH --mem=32G
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
cd $HOME/phd-research
module load bright 2>/dev/null || true
module load python3 2>/dev/null || true
source $HOME/phd-research-gpu-env/bin/activate
mkdir -p logs/mait experiments/invar/baselines

echo "host=$(hostname) start=$(date -Is)"
nvidia-smi | head -10 || true

# MAiT Phase 5: full 5-seed x 3-fold sweep = 15 runs.
# Single sbatch with internal loop (per repo convention; see
# scripts/wulver/invar_phase3_baselines_tier1.sbatch). Each run is full
# 30 epochs with early stopping (patience 5). F1 seed 42 with patience
# stopped at epoch 6 in Phase 4 -> roughly 5 min/run. Budget 15 x 20 min
# = 300 min, plus per-fold dataset init time; 8h walltime is safe.
#
# F3 needs --allow-fold-3 in the InVAR codebase. The baseline trainer
# doesn't enforce that gate (only the WAVE trainer does), so no flag here.
#
# Each run skips if its results.json already exists, so a re-launch after
# wall-time-out resumes from the next missing (fold, seed) pair.
for FOLD in 1 2 3; do
  for S in 42 43 44 45 46; do
    OUT="experiments/invar/baselines/mait/fold${FOLD}/seed${S}/results.json"
    if [ -f "$OUT" ]; then
      echo "[$(date +%H:%M:%S)] skip $OUT"
      continue
    fi
    LF="logs/mait/sweep_fold${FOLD}_seed${S}.log"
    echo "[$(date +%H:%M:%S)] mait fold=$FOLD seed=$S"
    PYTHONPATH=$PWD python3 -u -m src.invar.baselines.train_baseline \
      --baseline mait \
      --fold "$FOLD" --seed "$S" --epochs 30 \
      --output-dir experiments/invar/baselines \
      --save-predictions \
      > "$LF" 2>&1
  done
done

echo "DONE $(date -Is)"
