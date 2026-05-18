#!/bin/bash -l
#SBATCH --job-name=msginfo-pull
#SBATCH --partition=general
#SBATCH --account=hz79
#SBATCH --qos=standard
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export PYTHONUNBUFFERED=1

echo "=== StockTwits msg_info pull for S&P 500 universe ==="
echo "host=$(hostname)  start=$(date -Is)  cpus=$SLURM_CPUS_PER_TASK"

REPO="${REPO:-$HOME/phd-research}"
VENV="${VENV:-$HOME/phd-research-gpu-env}"
cd "$REPO"

module load bright 2>/dev/null || true
module load python3 2>/dev/null || true
source "$VENV/bin/activate"

PYTHONPATH="$REPO" python3 scripts/pull_sp500_stocktwits_msginfo.py

echo "=== done at $(date -Is) ==="
