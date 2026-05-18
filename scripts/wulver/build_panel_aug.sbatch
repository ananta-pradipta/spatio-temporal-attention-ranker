#!/bin/bash -l
#SBATCH --job-name=panel-aug-build
#SBATCH --partition=general
#SBATCH --account=hz79
#SBATCH --qos=standard
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

set -euo pipefail
cd $HOME/phd-research
module load bright 2>/dev/null || true
module load python3 2>/dev/null || true
source $HOME/phd-research-gpu-env/bin/activate

mkdir -p data/lattice_aug/processed experiments/lattice_aug logs/panel_aug

echo "host=$(hostname) start=$(date -Is)"

# Augmented panel: 37 columns (Tier A1+A2). Rebuilds panel_features,
# cohorts, active_mask, macro_state at data/lattice_aug/processed and
# fits per-fold scalers (F1-F5) at experiments/lattice_aug/foldX/scalers.pkl.
PYTHONPATH=$PWD python3 -u -m scripts.lattice.build_panel_augmented \
    2>&1 | tee logs/panel_aug/build.log

echo "DONE $(date -Is)"
