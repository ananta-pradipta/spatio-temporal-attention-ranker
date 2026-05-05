#!/bin/bash -l
#SBATCH --job-name=epistar-sbp-v1-all
#SBATCH --partition=gpu
#SBATCH --account=hz79
#SBATCH --qos=standard
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100_10g:1
#SBATCH --mem=24G
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

echo "=== epiSTAR-SBP v1 (revised: blackout 21d + delisting + macro 26-feat + cohort retrieval + IRF + V-REx + alpha-prior) ==="
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

# Ensure macro features parquet exists.
if [ ! -f "data/macro_features_v1.parquet" ]; then
  echo "[$(date +%H:%M:%S)] Building macro features..."
  python3 -c "from src.v2.data.macro_features import build_macro_features; build_macro_features()" || true
fi
# Ensure delisting CSV exists (auto-detect template).
python3 -c "from src.v2.data.delisting_v1 import write_template_csv; write_template_csv()" || true

OUT_CONTROL="results/epistar_sbp_v1_control"
LOG_CONTROL="logs/epistar_sbp_v1_control"
OUT_FULL="results/epistar_sbp_v1"
LOG_FULL="logs/epistar_sbp_v1"
mkdir -p "$OUT_CONTROL" "$LOG_CONTROL" "$OUT_FULL" "$LOG_FULL"

CONTROL_CONFIG="configs/epistar_sbp_v1_control.yaml"
FULL_CONFIG="configs/epistar_sbp_v1.yaml"
FOLDS=(1 2 3)
SEEDS=(42 43 44 45 46)

# Stage 1: control runs (Acceptance Criterion 1: rebuilt panel must
# reproduce epiSTAR-full numbers within 5-seed std error).
for FOLD in "${FOLDS[@]}"; do
  for S in "${SEEDS[@]}"; do
    OUT="$OUT_CONTROL/fold${FOLD}_seed${S}.json"
    LOG="$LOG_CONTROL/fold${FOLD}_seed${S}.log"
    if [ -f "$OUT" ]; then
      echo "[$(date +%H:%M:%S)] skip existing CONTROL fold=$FOLD seed=$S"
      continue
    fi
    echo "[$(date +%H:%M:%S)] CONTROL fold=$FOLD seed=$S -> $OUT"
    python3 -u -m src.v2.training.train_epistar_sbp_v1 \
      --config "$CONTROL_CONFIG" --fold "$FOLD" --seed "$S" \
      > "$LOG" 2>&1
  done
done

# Stage 2: full SBP-v1 runs (retrieval + loss revisions enabled).
for FOLD in "${FOLDS[@]}"; do
  for S in "${SEEDS[@]}"; do
    OUT="$OUT_FULL/fold${FOLD}_seed${S}.json"
    LOG="$LOG_FULL/fold${FOLD}_seed${S}.log"
    if [ -f "$OUT" ]; then
      echo "[$(date +%H:%M:%S)] skip existing FULL fold=$FOLD seed=$S"
      continue
    fi
    echo "[$(date +%H:%M:%S)] FULL fold=$FOLD seed=$S -> $OUT"
    python3 -u -m src.v2.training.train_epistar_sbp_v1 \
      --config "$FULL_CONFIG" --fold "$FOLD" --seed "$S" \
      > "$LOG" 2>&1
  done
done

echo "[$(date +%H:%M:%S)] DONE"
echo "end=$(date -Is)"
