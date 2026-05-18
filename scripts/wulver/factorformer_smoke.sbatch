#!/bin/bash -l
#SBATCH --job-name=model-eval
#SBATCH --partition=gpu
#SBATCH --account=hz79
#SBATCH --qos=standard
#SBATCH --time=01:00:00
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
mkdir -p logs/factorformer experiments/factorformer/fold1

echo "host=$(hostname) start=$(date -Is)"
nvidia-smi | head -10 || true

# Smoke: one fold (F1), one seed (42), full 10 epochs. Verifies the
# variate-as-token encoder + variational factor bottleneck train end to end
# before the full 5-seed x 3-fold sweep.
LF="logs/factorformer/smoke_f1_s42.log"
PYTHONPATH=$PWD python3 -u -m src.factorformer.train \
  --fold 1 --seed 42 --epochs 10 \
  --output-dir experiments/factorformer/fold1 \
  --kl-weight 1.0e-3 \
  --n-factors 8 \
  --save-predictions \
  > "$LF" 2>&1

echo "DONE $(date -Is)"
