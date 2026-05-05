#!/bin/bash -l
#SBATCH --job-name=CS785-FeatAblation
#SBATCH --partition=course
#SBATCH --account=2026-spring-cs-785-hz79-adp232
#SBATCH --qos=course
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

set -euo pipefail

# 9-cell feature-group ablation on 244 tickers. Three groups (PRICE only,
# ST only, ALL) x three models (Ridge, MLP, GAT on mechanistic graph).
# Single-job sequential; ~15-20 min total.

echo "=== CS785-FeatAblation  host=$(hostname)  start=$(date -Is) ==="

REPO="${REPO:-$HOME/phd-research}"
VENV="${VENV:-$HOME/phd-research-env}"
cd "$REPO"

module load bright 2>/dev/null || true
module load python3 2>/dev/null || true
# shellcheck disable=SC1091
source "$VENV/bin/activate"

python3 -c "import torch, torch_geometric, faiss; print('deps ok')"

python3 -m scripts.ablation_features

echo "=== CS785-FeatAblation  done  end=$(date -Is) ==="
