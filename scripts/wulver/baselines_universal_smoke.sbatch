#!/bin/bash -l
#SBATCH --job-name=model-eval-smoke
#SBATCH --partition=gpu
#SBATCH --account=hz79
#SBATCH --qos=standard
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --mem=48G
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

REPO="${REPO:-$HOME/phd-research}"
VENV="${VENV:-$HOME/phd-research-gpu-env}"
cd "$REPO"
source "$VENV/bin/activate"

mkdir -p logs/baselines_universal_smoke results/baselines_universal_smoke

echo "host=$(hostname) start=$(date -Is)"

# Smoke test: single baseline x single fold x single seed on universal panel.
# Confirms code wiring before launching the full 75-run sweep.
PYTHONPATH=$PWD python3 -u -m src.baselines.train_factorvae_v2 \
  --fold 1 --seed 42 \
  --panel_kind lattice_native \
  --two_regime_val \
  --output_dir results/baselines_universal_smoke \
  --max_epochs 2 \
  > logs/baselines_universal_smoke/factorvae_fold1_seed42.log 2>&1

echo "factorvae result:"
ls -la results/baselines_universal_smoke/
echo "DONE $(date -Is)"
