#!/bin/bash -l
#SBATCH --job-name=lattice-phase-d
#SBATCH --partition=gpu
#SBATCH --account=hz79
#SBATCH --qos=standard
#SBATCH --time=02:00:00
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
mkdir -p logs/lattice experiments/lattice/headline_v2

echo "host=$(hostname) start=$(date -Is)"
nvidia-smi | head -10 || true

# Phase D decision-gate runs: F2 seed 42 then F3 seed 42 with proper
# retrieval banks (Phase B) and earnings catalysts (Phase C).
for SPEC in "2 42" "3 42"; do
  read F S <<< "$SPEC"
  LF="logs/lattice/phase_d_fold${F}_seed${S}.log"
  echo "[$(date +%H:%M:%S)] starting fold=$F seed=$S"
  PYTHONPATH=$PWD python3 -u -m src.lattice.training.train \
    --fold "$F" --seed "$S" --epochs 10 \
    --output-dir experiments/lattice/headline_v2 \
    --save-predictions \
    > "$LF" 2>&1
done

echo "DONE $(date -Is)"
