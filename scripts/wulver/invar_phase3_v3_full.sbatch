#!/bin/bash -l
#SBATCH --job-name=invar-phase-3-v3-full
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
cd $HOME/phd-research
module load bright 2>/dev/null || true
module load python3 2>/dev/null || true
source $HOME/phd-research-gpu-env/bin/activate
mkdir -p logs/invar experiments/invar/headline_v3

echo "host=$(hostname) start=$(date -Is)"
nvidia-smi | head -10 || true

# v3 surgical fixes scaled to F1+F2 x 5 seeds. Skips already-done runs
# (e.g., F1 seed 42 was completed in the verification gate).
for FOLD in 1 2; do
  for S in 42 43 44 45 46; do
    OUT="experiments/invar/headline_v3/fold${FOLD}/seed${S}_designR/results.json"
    if [ -f "$OUT" ]; then
      echo "[$(date +%H:%M:%S)] skip $OUT"
      continue
    fi
    LF="logs/invar/headline_v3_fold${FOLD}_seed${S}.log"
    echo "[$(date +%H:%M:%S)] fold=$FOLD seed=$S"
    PYTHONPATH=$PWD python3 -u -m src.invar.training.train \
      --fold "$FOLD" --seed "$S" --epochs 10 \
      --output-dir "experiments/invar/headline_v3/fold${FOLD}" \
      --regime-axis retrieval \
      --loss-config ranking \
      --bank-size 64 \
      --save-predictions \
      > "$LF" 2>&1
  done
done

echo "DONE $(date -Is)"
