#!/bin/bash -l
#SBATCH --job-name=model-eval
#SBATCH --partition=gpu
#SBATCH --account=hz79
#SBATCH --qos=standard
#SBATCH --time=05:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --mem=32G
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

# F2-targeted feature sweep: rate_beta_60d + delta_rv_20d + low_vol_decile.
# Reads data/lattice_f2feats/processed/panel_features.parquet via the
# new lattice_native_f2 panel_kind branch (feature_dim=29).
export LATTICE_PROCESSED_DIR=data/lattice_f2feats/processed
export LATTICE_SCALER_DIR=experiments/lattice_f2feats

REPO="${REPO:-$HOME/phd-research}"
VENV="${VENV:-$HOME/phd-research-gpu-env}"
cd "$REPO"
module load bright 2>/dev/null || true
module load python3 2>/dev/null || true
source "$VENV/bin/activate"

mkdir -p logs/rag_star_universe_v2_f2feats results/rag_star_universe_v2_f2feats

echo "host=$(hostname) start=$(date -Is)"
echo "LATTICE_PROCESSED_DIR=$LATTICE_PROCESSED_DIR"

# F1 + F2 + F3 x 5 seeds = 15 runs (~10-16 min each, ~3-4h wall).
for FOLD in 1 2 3; do
  for S in 42 43 44 45 46; do
    OUT="results/rag_star_universe_v2_f2feats/fold${FOLD}_seed${S}.json"
    if [ -f "$OUT" ] && [ "$(grep -c '"epoch"' "$OUT")" -ge 5 ]; then
      echo "[$(date +%H:%M:%S)] skip $OUT"
      continue
    fi
    LF="logs/rag_star_universe_v2_f2feats/fold${FOLD}_seed${S}.log"
    echo "[$(date +%H:%M:%S)] v2-f2feats fold=$FOLD seed=$S"
    PYTHONPATH=$PWD python3 -u -m src.v2.training.train_dow_epistar \
      --config configs/rag_star_universe_v2_f2feats.yaml \
      --fold "$FOLD" --seed "$S" > "$LF" 2>&1
  done
done

echo "DONE $(date -Is)"
