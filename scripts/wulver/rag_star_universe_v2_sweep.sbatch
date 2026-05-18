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
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

REPO="${REPO:-$HOME/phd-research}"
VENV="${VENV:-$HOME/phd-research-gpu-env}"
cd "$REPO"
module load bright 2>/dev/null || true
module load python3 2>/dev/null || true
source "$VENV/bin/activate"

mkdir -p logs/rag_star_universe_v2 results/rag_star_universe_v2

echo "host=$(hostname) start=$(date -Is)"
nvidia-smi | head -8 || true

# RAG-STAR Universe v2 sweep: LATTICE-native 26-col panel. F1-F5 x 5 seeds.
# Walltime 6h sized to fit pre-maintenance (maintenance 09:00-21:00 today).
for FOLD in 1 2 3 4 5; do
  for S in 42 43 44 45 46; do
    OUT="results/rag_star_universe_v2/fold${FOLD}_seed${S}.json"
    if [ -f "$OUT" ] && [ "$(grep -c '"epoch"' "$OUT")" -ge 5 ]; then
      echo "[$(date +%H:%M:%S)] skip $OUT"
      continue
    fi
    LF="logs/rag_star_universe_v2/full_fold${FOLD}_seed${S}.log"
    echo "[$(date +%H:%M:%S)] rag-star-universe-v2 fold=$FOLD seed=$S"
    PYTHONPATH=$PWD python3 -u -m src.v2.training.train_dow_epistar \
      --config configs/rag_star_universe_v2.yaml \
      --fold "$FOLD" --seed "$S" \
      > "$LF" 2>&1
  done
done

echo "DONE $(date -Is)"
