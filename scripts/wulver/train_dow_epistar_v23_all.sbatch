#!/bin/bash -l
#SBATCH --job-name=dow-epistar-v23-all
#SBATCH --partition=gpu
#SBATCH --account=hz79
#SBATCH --qos=standard
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --mem=24G
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

echo "=== DOW-epiSTAR v2.3 sweep (gates init -3.0, 9-d lambda gate, 17-d rate key, duration-aware rate value, deterministic duration graph) ==="
echo "host=$(hostname)  start=$(date -Is)"
nvidia-smi | head -20 || true

REPO="${REPO:-$HOME/phd-research}"
VENV="${VENV:-$HOME/phd-research-gpu-env}"
cd "$REPO"

module load bright 2>/dev/null || true
module load python3 2>/dev/null || true
if [ ! -d "$VENV" ]; then python3 -m venv "$VENV"; fi
source "$VENV/bin/activate"
pip install --upgrade pip --quiet
if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  pip install --quiet --index-url https://download.pytorch.org/whl/cu121 torch
fi
pip install --quiet -r requirements.txt
pip install --quiet pandas-datareader yfinance
python3 -c "import torch; print('torch', torch.__version__, 'cuda:', torch.cuda.is_available())"

if [ ! -f "data/processed/macro_duration_features.parquet" ]; then
  python3 -c "from src.v2.data.macro_duration_features import build_macro_duration_features; build_macro_duration_features()"
fi
if [ ! -f "data/processed/rolling_macro_betas.parquet" ]; then
  python3 -c "from src.v2.data.rolling_macro_betas import build_rolling_betas; build_rolling_betas()"
fi

mkdir -p logs/dow_epistar_v23

# Phase 1: full sweep for the headline config (3 folds x 5 seeds).
HEADLINE_CONFIG="configs/dow_epistar_v23_full.yaml"
HEADLINE_OUT="results/dow_epistar_v23_full"
HEADLINE_LOG="logs/dow_epistar_v23/full"
mkdir -p "$HEADLINE_OUT" "$HEADLINE_LOG"
for FOLD in 1 2 3; do
  for S in 42 43 44 45 46; do
    OUT="$HEADLINE_OUT/fold${FOLD}_seed${S}.json"
    LOG="$HEADLINE_LOG/fold${FOLD}_seed${S}.log"
    if [ -f "$OUT" ]; then echo "[$(date +%H:%M:%S)] skip $OUT"; continue; fi
    echo "[$(date +%H:%M:%S)] FULL fold=$FOLD seed=$S"
    python3 -u -m src.v2.training.train_dow_epistar \
      --config "$HEADLINE_CONFIG" --fold "$FOLD" --seed "$S" \
      > "$LOG" 2>&1
  done
done

# Phase 2: ablation on fold 3 only (5 seeds each), 6 configs.
ABLATIONS=(
  "macro_only"
  "no_rate_memory"
  "no_duration_graph"
  "shuffled_macro"
  "shuffled_duration"
  "random_rate_retrieval"
)
for VAR in "${ABLATIONS[@]}"; do
  CONFIG="configs/dow_epistar_v23_${VAR}.yaml"
  OUTD="results/dow_epistar_v23_${VAR}"
  LOGD="logs/dow_epistar_v23/${VAR}"
  mkdir -p "$OUTD" "$LOGD"
  for S in 42 43 44 45 46; do
    OUT="$OUTD/fold3_seed${S}.json"
    LOG="$LOGD/fold3_seed${S}.log"
    if [ -f "$OUT" ]; then echo "[$(date +%H:%M:%S)] skip $OUT"; continue; fi
    echo "[$(date +%H:%M:%S)] ABLATION $VAR fold=3 seed=$S"
    python3 -u -m src.v2.training.train_dow_epistar \
      --config "$CONFIG" --fold 3 --seed "$S" \
      > "$LOG" 2>&1
  done
done

echo "[$(date +%H:%M:%S)] DONE"
echo "end=$(date -Is)"
