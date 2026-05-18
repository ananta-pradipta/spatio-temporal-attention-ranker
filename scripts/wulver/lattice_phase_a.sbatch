#!/bin/bash -l
#SBATCH --job-name=lattice-phase-a
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
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
cd $HOME/phd-research
module load bright 2>/dev/null || true
module load python3 2>/dev/null || true
source $HOME/phd-research-gpu-env/bin/activate
mkdir -p logs/lattice experiments/lattice/diagnostics

echo "host=$(hostname) start=$(date -Is)"
nvidia-smi | head -10 || true

PYTHONPATH=$PWD python3 -u -m scripts.lattice.diagnostics.extract_phase_a \
  --headline-dir experiments/lattice/headline \
  --output-dir experiments/lattice/diagnostics \
  --folds 1,2,3 \
  --seeds 42,43,44,45,46 \
  > logs/lattice/phase_a_extract.log 2>&1

PYTHONPATH=$PWD python3 -u -m scripts.lattice.diagnostics.summarise_phase_a \
  --diag-dir experiments/lattice/diagnostics \
  > logs/lattice/phase_a_summarise.log 2>&1

echo "end=$(date -Is)"
