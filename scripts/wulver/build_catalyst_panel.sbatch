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

mkdir -p logs/panel_aug data/lattice_catalyst/processed experiments/lattice_catalyst

echo "host=$(hostname) start=$(date -Is)"

# Solution G data ingestion: build the catalyst-populated Universal Ticker
# panel at data/lattice_catalyst/processed/ from the canonical 1.27M-row
# panel + the existing 17.5K-row earnings calendar. Refits per-fold
# scalers. Does NOT modify the canonical panel; safe to run while the
# in-flight 1026227 sweep continues against data/lattice/processed/.
PYTHONPATH=$PWD python3 -u -m scripts.lattice.build_panel_with_catalyst \
    2>&1 | tee logs/panel_aug/catalyst_panel.log

echo "DONE $(date -Is)"
