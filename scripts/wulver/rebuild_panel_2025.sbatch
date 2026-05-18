#!/bin/bash -l
#SBATCH --job-name=model-eval
#SBATCH --partition=general
#SBATCH --qos=standard
#SBATCH --account=hz79
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

set -euo pipefail
cd $HOME/phd-research
module load bright 2>/dev/null || true
module load python3 2>/dev/null || true
source $HOME/phd-research-gpu-env/bin/activate
mkdir -p logs/lattice

echo "host=$(hostname) start=$(date -Is)"

# Rebuild processed panel + macro state with extended 2015-2025 coverage.
echo "=== Rebuilding panel (build_phase1) and macro state (build_macro_state) ==="
PYTHONPATH=$PWD python3 -u scripts/lattice/build_panel_2025.py 2>&1 | tee logs/lattice/build_panel_2025.log

# Then refit fold scalers for F1, F2, F3 (existing) on the new panel data.
# Note: F1, F2, F3 train ranges don't change with the extension, but the
# active mask may include more (date, ticker) cells if a 2023+ addition
# becomes active before the train range ends. Refit to be safe.
echo "=== Refitting F1, F2, F3 fold scalers ==="
PYTHONPATH=$PWD python3 -u scripts/lattice/build_phase2.py 2>&1 | tee logs/lattice/build_phase2.log

echo "DONE $(date -Is)"
