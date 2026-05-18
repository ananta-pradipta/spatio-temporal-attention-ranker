#!/bin/bash -l
#SBATCH --job-name=model-eval
#SBATCH --partition=gpu
#SBATCH --account=hz79
#SBATCH --qos=standard
#SBATCH --time=02:30:00
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

mkdir -p logs/rag_star_universe results/rag_star_universe

echo "host=$(hostname) start=$(date -Is)"
nvidia-smi | head -8 || true

# RAG-STAR Universe F4 + F5 on extended LATTICE panel + extended v2 aux
# parquets (risk_features_sp500_ext, macro_duration_features_sp500_ext,
# sp500_rolling_betas_ext). 5 seeds x 2 folds = 10 runs. StockTwits ends
# 2022-12 so F4/F5 ST cols are zero-filled (the model still gets PRICE +
# FUND + FLAG + auxiliary parquets in full).
for FOLD in 4 5; do
  for S in 42 43 44 45 46; do
    OUT="results/rag_star_universe/fold${FOLD}_seed${S}.json"
    if [ -f "$OUT" ] && [ "$(grep -c '"epoch"' "$OUT")" -ge 5 ]; then
      echo "[$(date +%H:%M:%S)] skip $OUT"
      continue
    fi
    LF="logs/rag_star_universe/ext_fold${FOLD}_seed${S}.log"
    echo "[$(date +%H:%M:%S)] rag-star-universe-ext fold=$FOLD seed=$S"
    PYTHONPATH=$PWD python3 -u -m src.v2.training.train_dow_epistar \
      --config configs/rag_star_universe_ext.yaml \
      --fold "$FOLD" --seed "$S" \
      > "$LF" 2>&1
  done
done

echo "DONE $(date -Is)"
