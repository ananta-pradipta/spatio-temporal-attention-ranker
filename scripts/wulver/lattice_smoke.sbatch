#!/bin/bash -l
#SBATCH --job-name=lattice-smoke
#SBATCH --partition=gpu
#SBATCH --account=hz79
#SBATCH --qos=standard
#SBATCH --time=01:30:00
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
mkdir -p logs/lattice

echo "host=$(hostname) start=$(date -Is)"
nvidia-smi | head -10 || true

python3 -u -m src.lattice.training.train \
  --fold 1 --seed 42 --epochs 3 \
  --output-dir experiments/lattice/smoke \
  > logs/lattice/smoke_fold1_seed42.log 2>&1

echo "end=$(date -Is)"
