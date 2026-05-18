#!/bin/bash -l
#SBATCH --job-name=model-eval
#SBATCH --partition=gpu
#SBATCH --account=hz79
#SBATCH --qos=standard
#SBATCH --time=06:00:00
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

mkdir -p logs/rag_star_universe logs/rag_star_universe_v2 logs/rag_star_universe_ablation_b
mkdir -p results/rag_star_universe results/rag_star_universe_v2 results/rag_star_universe_ablation_b

echo "host=$(hostname) start=$(date -Is)"
nvidia-smi | head -8 || true

# Post-maintenance fill-in for the 2026-05-12 RAG-STAR Universe corrected
# sweeps. All skip-guards short-circuit if results.json already exists.
# Order: v2 first (highest priority, the new leader), then v1, then Ablation B.

# v2 (26-col LATTICE-native) — F1-F5 sweep. ext config covers F4+F5 via
# the *_ext.parquet auxiliary files.
echo "=== v2 sweep (lattice_native, F1-F5) ==="
for FOLD in 3 4 5 1 2; do
  for S in 42 43 44 45 46; do
    OUT="results/rag_star_universe_v2/fold${FOLD}_seed${S}.json"
    if [ -f "$OUT" ] && [ "$(grep -c '"epoch"' "$OUT")" -ge 5 ]; then
      echo "[$(date +%H:%M:%S)] skip $OUT"
      continue
    fi
    LF="logs/rag_star_universe_v2/full_fold${FOLD}_seed${S}.log"
    echo "[$(date +%H:%M:%S)] v2 fold=$FOLD seed=$S"
    PYTHONPATH=$PWD python3 -u -m src.v2.training.train_dow_epistar \
      --config configs/rag_star_universe_v2.yaml \
      --fold "$FOLD" --seed "$S" > "$LF" 2>&1
  done
done

# v1 (22-col mapped) — F2 seed46 + F3 all + F4-F5 via ext config.
echo "=== v1 sweep (lattice, F2-F5 fill) ==="
for FOLD in 2 3; do
  for S in 42 43 44 45 46; do
    OUT="results/rag_star_universe/fold${FOLD}_seed${S}.json"
    if [ -f "$OUT" ] && [ "$(grep -c '"epoch"' "$OUT")" -ge 5 ]; then
      echo "[$(date +%H:%M:%S)] skip $OUT"
      continue
    fi
    LF="logs/rag_star_universe/full_fold${FOLD}_seed${S}.log"
    echo "[$(date +%H:%M:%S)] v1 fold=$FOLD seed=$S"
    PYTHONPATH=$PWD python3 -u -m src.v2.training.train_dow_epistar \
      --config configs/rag_star_universe.yaml \
      --fold "$FOLD" --seed "$S" > "$LF" 2>&1
  done
done

# Ablation B — F3 fill.
echo "=== Ablation B (lattice, ST=0, F3 fill) ==="
export LATTICE_PANEL_ZERO_ST=1
TMP_CFG=$(mktemp /tmp/rsu_ablb.XXXXXX.yaml)
trap "rm -f $TMP_CFG" EXIT
sed 's|results/rag_star_universe|results/rag_star_universe_ablation_b|' \
  configs/rag_star_universe.yaml > "$TMP_CFG"
for FOLD in 3; do
  for S in 42 43 44 45 46; do
    OUT="results/rag_star_universe_ablation_b/fold${FOLD}_seed${S}.json"
    if [ -f "$OUT" ] && [ "$(grep -c '"epoch"' "$OUT")" -ge 5 ]; then
      echo "[$(date +%H:%M:%S)] skip $OUT"
      continue
    fi
    LF="logs/rag_star_universe_ablation_b/fold${FOLD}_seed${S}.log"
    echo "[$(date +%H:%M:%S)] ablation-b fold=$FOLD seed=$S"
    PYTHONPATH=$PWD python3 -u -m src.v2.training.train_dow_epistar \
      --config "$TMP_CFG" --fold "$FOLD" --seed "$S" > "$LF" 2>&1
  done
done
unset LATTICE_PANEL_ZERO_ST

echo "DONE $(date -Is)"
