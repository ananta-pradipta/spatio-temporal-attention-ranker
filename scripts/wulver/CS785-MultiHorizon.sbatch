#!/bin/bash -l
#SBATCH --job-name=CS785-MultiHorizon
#SBATCH --partition=course
#SBATCH --account=2026-spring-cs-785-hz79-adp232
#SBATCH --qos=course
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

set -euo pipefail

# Single-job sweep (per memory: no Slurm arrays).
# Three horizon configurations x 3 seeds = 9 runs. Tests whether
# per-horizon ranking heads exploit feature-group-specific
# timescales better than a single 5d target.

SEEDS=(11 22 33)

echo "=== CS785-MultiHorizon  host=$(hostname)  start=$(date -Is) ==="

REPO="${REPO:-$HOME/phd-research}"
VENV="${VENV:-$HOME/phd-research-env}"
cd "$REPO"

module load bright 2>/dev/null || true
module load python3 2>/dev/null || true
# shellcheck disable=SC1091
source "$VENV/bin/activate"

python3 -c "import torch, torch_geometric; print('deps ok')"

OUT_DIR="results/multi_horizon"
mkdir -p "$OUT_DIR"

# Config 1: single horizon 5d (baseline, matches existing fixed sweep)
for SEED in "${SEEDS[@]}"; do
    OUT="$OUT_DIR/h5only_seed${SEED}.json"
    echo ""
    echo "--- h=5 only  seed=$SEED  $(date -Is) ---"
    python3 -m src.mtgn.training.train_mtgn_multi \
        --max-tickers 300 --horizons 5 --epochs 15 --patience 4 --seed "$SEED" \
        --output "$OUT"
done

# Config 2: single horizon 1d
for SEED in "${SEEDS[@]}"; do
    OUT="$OUT_DIR/h1only_seed${SEED}.json"
    echo ""
    echo "--- h=1 only  seed=$SEED  $(date -Is) ---"
    python3 -m src.mtgn.training.train_mtgn_multi \
        --max-tickers 300 --horizons 1 --epochs 15 --patience 4 --seed "$SEED" \
        --output "$OUT"
done

# Config 3: multi-horizon {1, 5}
for SEED in "${SEEDS[@]}"; do
    OUT="$OUT_DIR/h1_5_seed${SEED}.json"
    echo ""
    echo "--- h=[1,5]  seed=$SEED  $(date -Is) ---"
    python3 -m src.mtgn.training.train_mtgn_multi \
        --max-tickers 300 --horizons 1 5 --epochs 15 --patience 4 --seed "$SEED" \
        --output "$OUT"
done

echo ""
echo "=== CS785-MultiHorizon  done  end=$(date -Is) ==="
ls -la "$OUT_DIR"
