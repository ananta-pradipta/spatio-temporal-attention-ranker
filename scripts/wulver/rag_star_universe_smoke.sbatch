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

REPO="${REPO:-$HOME/phd-research}"
VENV="${VENV:-$HOME/phd-research-gpu-env}"
cd "$REPO"
module load bright 2>/dev/null || true
module load python3 2>/dev/null || true
source "$VENV/bin/activate"

echo "host=$(hostname) start=$(date -Is)"
nvidia-smi | head -8 || true

mkdir -p logs/rag_star_universe results/rag_star_universe_smoke

# Smoke: F1 seed 42, epochs=2 via the smoke CLI flag. Validates the LATTICE
# adapter wires end-to-end without shape mismatches or NaN losses before
# committing to the full 25-run sweep.
PYTHONPATH=$PWD python3 -u -m src.v2.training.train_dow_epistar \
  --config configs/rag_star_universe.yaml \
  --fold 1 --seed 42 --smoke \
  2>&1 | tee logs/rag_star_universe/smoke_f1_s42.log

echo "DONE $(date -Is)"
