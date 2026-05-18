#!/bin/bash -l
#SBATCH --job-name=lattice-phase-b-smoke
#SBATCH --partition=gpu
#SBATCH --account=hz79
#SBATCH --qos=standard
#SBATCH --time=00:30:00
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

# Smoke: build banks on F1 and verify size + non-zero retrieval output.
# Unit tests run locally (Wulver GPU env does not include pytest).
PYTHONPATH=$PWD python3 -u -m scripts.lattice.diagnostics.phase_b_smoke \
  > logs/lattice/phase_b_smoke.log 2>&1

echo "end=$(date -Is)"
