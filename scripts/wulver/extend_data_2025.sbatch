#!/bin/bash -l
#SBATCH --job-name=model-eval
#SBATCH --partition=general
#SBATCH --qos=standard
#SBATCH --account=hz79
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

set -euo pipefail
cd $HOME/phd-research
module load bright 2>/dev/null || true
module load python3 2>/dev/null || true
source $HOME/phd-research-gpu-env/bin/activate
mkdir -p logs/lattice

echo "host=$(hostname) start=$(date -Is)"

# Step 1: extend prices forward (yfinance, ~30 min for ~750 tickers).
echo "=== Step 1: extend prices ==="
python3 -u scripts/lattice/extend_prices.py 2>&1 | tee logs/lattice/extend_prices.log

# Step 2: extend macro FRED + ETFs through current date.
echo "=== Step 2: extend macro to 2025 ==="
python3 -u scripts/lattice/extend_macro_2025.py 2>&1 | tee logs/lattice/extend_macro_2025.log || true

echo "DONE $(date -Is)"
