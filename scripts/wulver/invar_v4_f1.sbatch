#!/bin/bash -l
#SBATCH --job-name=invar-v4-f1
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
cd $HOME/phd-research
module load bright 2>/dev/null || true
module load python3 2>/dev/null || true
source $HOME/phd-research-gpu-env/bin/activate
mkdir -p logs/invar experiments/invar/v4/fold1

echo "host=$(hostname) start=$(date -Is)"
nvidia-smi | head -10 || true

# InVAR v4 F1, 5 seeds.
for S in 42 43 44 45 46; do
  OUT="experiments/invar/v4/fold1/seed${S}_designR/results.json"
  if [ -f "$OUT" ]; then
    echo "[$(date +%H:%M:%S)] skip $OUT"
    continue
  fi
  LF="logs/invar/v4_fold1_seed${S}.log"
  PYTHONPATH=$PWD python3 -u -m src.invar.training.train \
    --fold 1 --seed "$S" --epochs 10 \
    --output-dir experiments/invar/v4/fold1 \
    --regime-axis retrieval \
    --loss-config ranking \
    --bank-size 64 \
    --use-market-gate --gate-beta-init 2.0 --gate-l2-weight 1.0e-4 \
    --save-predictions \
    > "$LF" 2>&1
done

echo "DONE $(date -Is)"
