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

mkdir -p logs/panel_aug data/lattice_f2feats/processed experiments/lattice_f2feats

echo "host=$(hostname) start=$(date -Is)"

# F2-targeted panel build (Items 1+2 of the 2026-05-13 F2 deep-dive):
# rate_beta_60d + delta_rv_20d + low_vol_decile_flag appended to the
# canonical 26-col panel at a sibling directory data/lattice_f2feats/
# processed/. Does NOT modify the canonical panel; safe to run while
# the in-flight v2/v2-catalyst sweeps continue.
PYTHONPATH=$PWD python3 -u -m scripts.lattice.build_panel_with_f2_features \
    2>&1 | tee logs/panel_aug/f2feats_panel.log

echo "DONE $(date -Is)"
