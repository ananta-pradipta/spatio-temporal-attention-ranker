#!/bin/bash -l
#SBATCH --job-name=model-eval
#SBATCH --partition=gpu
#SBATCH --account=hz79
#SBATCH --qos=standard
#SBATCH --time=02:00:00
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

mkdir -p logs/rag_star_universe results/rag_star_universe

echo "host=$(hostname) start=$(date -Is)"
nvidia-smi | head -8 || true

# Rerun the 3 RAG-STAR Universe seeds that ran on the OLD adapter (ST cols
# stuck at zero). 2026-05-12 ~01:11 fix populated ST via direct v2-name
# lookup in stocktwits_features_sp500.parquet.
for SPEC in 1:44 1:46 2:42; do
  FOLD=${SPEC%%:*}
  SEED=${SPEC##*:}
  OUT="results/rag_star_universe/fold${FOLD}_seed${SEED}.json"
  if [ -f "$OUT" ]; then
    echo "[$(date +%H:%M:%S)] skip $OUT"
    continue
  fi
  LF="logs/rag_star_universe/fix_fold${FOLD}_seed${SEED}.log"
  echo "[$(date +%H:%M:%S)] rerun fold=$FOLD seed=$SEED (ST fix)"
  PYTHONPATH=$PWD python3 -u -m src.v2.training.train_dow_epistar \
    --config configs/rag_star_universe.yaml \
    --fold "$FOLD" --seed "$SEED" \
    > "$LF" 2>&1
done

echo "DONE $(date -Is)"
