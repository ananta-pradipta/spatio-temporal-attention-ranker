#!/bin/bash -l
#SBATCH --job-name=model-eval
#SBATCH --partition=gpu
#SBATCH --account=hz79
#SBATCH --qos=standard
#SBATCH --time=04:00:00
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

# Solution G evaluation: route the trainer at the catalyst-populated panel.
# Architecture is identical; only the 3 catalyst columns now carry the
# earnings-calendar signal instead of being zero-filled.
export LATTICE_PROCESSED_DIR=data/lattice_catalyst/processed
export LATTICE_SCALER_DIR=experiments/lattice_catalyst

REPO="${REPO:-$HOME/phd-research}"
VENV="${VENV:-$HOME/phd-research-gpu-env}"
cd "$REPO"
module load bright 2>/dev/null || true
module load python3 2>/dev/null || true
source "$VENV/bin/activate"

mkdir -p logs/rag_star_universe_v2_catalyst results/rag_star_universe_v2_catalyst

echo "host=$(hostname) start=$(date -Is)"
echo "LATTICE_PROCESSED_DIR=$LATTICE_PROCESSED_DIR"
echo "LATTICE_SCALER_DIR=$LATTICE_SCALER_DIR"

# Universal RAG-STAR v2 with catalyst features populated. F1+F2+F3 = 15
# runs at ~10 min each on the a100_20g MIG slice. Calendar caps at
# 2023-01 so F4/F5 catalyst columns degrade to zero-fill; F1-F3 is the
# scope where catalyst signal is fully populated.
TMP_CFG=$(mktemp /tmp/rsu_v2_catalyst.XXXXXX.yaml)
trap "rm -f $TMP_CFG" EXIT
sed 's|results/rag_star_universe_v2|results/rag_star_universe_v2_catalyst|' \
  configs/rag_star_universe_v2.yaml > "$TMP_CFG"

for FOLD in 1 2 3; do
  for S in 42 43 44 45 46; do
    OUT="results/rag_star_universe_v2_catalyst/fold${FOLD}_seed${S}.json"
    if [ -f "$OUT" ] && [ "$(grep -c '"epoch"' "$OUT")" -ge 5 ]; then
      echo "[$(date +%H:%M:%S)] skip $OUT"
      continue
    fi
    LF="logs/rag_star_universe_v2_catalyst/fold${FOLD}_seed${S}.log"
    echo "[$(date +%H:%M:%S)] v2-catalyst fold=$FOLD seed=$S"
    PYTHONPATH=$PWD python3 -u -m src.v2.training.train_dow_epistar \
      --config "$TMP_CFG" --fold "$FOLD" --seed "$S" > "$LF" 2>&1
  done
done

echo "DONE $(date -Is)"
