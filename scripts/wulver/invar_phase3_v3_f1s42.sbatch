#!/bin/bash -l
#SBATCH --job-name=invar-phase-3-v3
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
mkdir -p logs/invar experiments/invar/headline_v3/fold1

echo "host=$(hostname) start=$(date -Is)"
nvidia-smi | head -10 || true

# v3 surgical fixes: minimal loss (Huber only), bank_size=64, scalar gate,
# zero-init cross_attn out_proj, no K curriculum, no stop-grad warmup.
PYTHONPATH=$PWD python3 -u -m src.invar.training.train \
  --fold 1 --seed 42 --epochs 10 \
  --output-dir experiments/invar/headline_v3/fold1 \
  --regime-axis retrieval \
  --loss-config minimal \
  --bank-size 64 \
  --save-predictions \
  > logs/invar/headline_v3_fold1_seed42.log 2>&1

echo "end=$(date -Is)"
