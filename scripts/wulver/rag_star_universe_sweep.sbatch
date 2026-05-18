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
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

REPO="${REPO:-$HOME/phd-research}"
VENV="${VENV:-$HOME/phd-research-gpu-env}"
cd "$REPO"
module load bright 2>/dev/null || true
module load python3 2>/dev/null || true
source "$VENV/bin/activate"

echo "host=$(hostname) start=$(date -Is)"
nvidia-smi | head -8 || true

mkdir -p logs/rag_star_universe results/rag_star_universe

# RAG-STAR Universe full sweep: LATTICE 26-col panel mapped to v2 22-col
# schema; LATTICE FOLDS F1+F2+F3 (canonical 6-month F3 test). F4/F5
# blocked by raw S&P 500 prices ending 2023-01 on the v2 auxiliary
# parquets. 5 seeds x 3 folds = 15 runs at ~30-40 min/seed = ~8-10h.
for FOLD in 1 2 3; do
  for S in 42 43 44 45 46; do
    OUT="results/rag_star_universe/fold${FOLD}_seed${S}.json"
    if [ -f "$OUT" ] && grep -q '"history"' "$OUT" 2>/dev/null; then
      # Has real history -> not a smoke stub. Skip.
      if [ "$(grep -c '"epoch"' "$OUT")" -ge 5 ]; then
        echo "[$(date +%H:%M:%S)] skip $OUT (>=5 epochs in history)"
        continue
      fi
    fi
    LF="logs/rag_star_universe/full_fold${FOLD}_seed${S}.log"
    echo "[$(date +%H:%M:%S)] rag-star-universe fold=$FOLD seed=$S"
    PYTHONPATH=$PWD python3 -u -m src.v2.training.train_dow_epistar \
      --config configs/rag_star_universe.yaml \
      --fold "$FOLD" --seed "$S" \
      > "$LF" 2>&1
  done
done

echo "DONE $(date -Is)"
