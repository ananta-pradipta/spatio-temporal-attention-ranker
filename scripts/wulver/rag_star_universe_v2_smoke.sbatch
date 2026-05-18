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

mkdir -p logs/rag_star_universe_v2 results/rag_star_universe_v2

# RAG-STAR Universe v2 smoke: F1 seed 42 epochs=2. Validates the
# lattice_native (26-col) adapter wires without shape mismatches.
PYTHONPATH=$PWD python3 -u -m src.v2.training.train_dow_epistar \
  --config configs/rag_star_universe_v2.yaml \
  --fold 1 --seed 42 --smoke \
  2>&1 | tee logs/rag_star_universe_v2/smoke_f1_s42.log

echo "DONE $(date -Is)"
