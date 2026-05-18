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
nvidia-smi | head -10 || true

# Extended-panel MAiT-17 sweep: F3, F4, F5 x 5 seeds = 15 runs at ~5-10 min
# per seed (early stops). Uses MAIT_PRESET=v17 to select the 17-feature
# macro subset (original 2026-05-10 MAiT spec). Output goes to
# experiments/invar_2025/baselines/mait/ (canonical MAiT row alongside iT,
# MASTER, etc.); the 8-feature minimal results stay at
# experiments/invar_2025/baselines_minimal/mait/.
for FOLD in 3 4 5; do
  for S in 42 43 44 45 46; do
    OUT="experiments/invar_2025/baselines/mait/fold${FOLD}/seed${S}/results.json"
    if [ -f "$OUT" ]; then
      echo "[$(date +%H:%M:%S)] skip $OUT"
      continue
    fi
    LF="logs/invar_2025/mait_v17_fold${FOLD}_seed${S}.log"
    echo "[$(date +%H:%M:%S)] mait-v17 fold=$FOLD seed=$S"
    PYTHONPATH=$PWD python3 -u -m src.invar.baselines.train_baseline \
      --baseline mait \
      --fold "$FOLD" --seed "$S" --epochs 30 \
      --output-dir experiments/invar_2025/baselines \
      --save-predictions \
      > "$LF" 2>&1
  done
done

echo "DONE $(date -Is)"
