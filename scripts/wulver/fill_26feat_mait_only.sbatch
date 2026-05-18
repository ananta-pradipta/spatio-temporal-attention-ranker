#!/bin/bash -l
#SBATCH --job-name=model-eval
#SBATCH --partition=gpu
#SBATCH --account=hz79
#SBATCH --qos=standard
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --mem=32G
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MAIT_PRESET=v17

cd $HOME/phd-research
module load bright 2>/dev/null || true
module load python3 2>/dev/null || true
source $HOME/phd-research-gpu-env/bin/activate
mkdir -p logs/invar_2025 experiments/invar_2025/baselines

echo "host=$(hostname) start=$(date -Is)"
echo "panel = canonical 26-feature (data/lattice/processed, default)"
echo "MAIT_PRESET=$MAIT_PRESET"
nvidia-smi | head -10 || true

# Finish MAiT-17 F4 + F5 (per user 2026-05-12 00:44 instruction:
# StockMixer F3-F5 deferred). F3 already complete. Per-seed wall
# ~25-30 min; total ~5h max.
for FOLD in 4 5; do
  for S in 42 43 44 45 46; do
    OUT="experiments/invar_2025/baselines/mait/fold${FOLD}/seed${S}/results.json"
    if [ -f "$OUT" ]; then
      echo "[$(date +%H:%M:%S)] skip $OUT"
      continue
    fi
    LF="logs/invar_2025/mait_fold${FOLD}_seed${S}.log"
    echo "[$(date +%H:%M:%S)] baseline=mait fold=$FOLD seed=$S"
    PYTHONPATH=$PWD python3 -u -m src.invar.baselines.train_baseline \
      --baseline mait \
      --fold "$FOLD" --seed "$S" --epochs 30 \
      --output-dir experiments/invar_2025/baselines \
      --save-predictions \
      > "$LF" 2>&1
  done
done

echo "DONE $(date -Is)"
