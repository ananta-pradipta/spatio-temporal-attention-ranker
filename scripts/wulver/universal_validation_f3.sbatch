#!/bin/bash -l
#SBATCH --job-name=model-eval
#SBATCH --partition=gpu
#SBATCH --account=hz79
#SBATCH --qos=standard
#SBATCH --time=12:00:00
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

echo "host=$(hostname) start=$(date -Is)"

REPO="${REPO:-$HOME/phd-research}"
VENV="${VENV:-$HOME/phd-research-gpu-env}"
cd "$REPO"

module load bright 2>/dev/null || true
module load python3 2>/dev/null || true
source "$VENV/bin/activate"
python3 -c "import torch; print('torch', torch.__version__, 'cuda:', torch.cuda.is_available())"

LOG_BASE="logs/universal_validation"
mkdir -p "$LOG_BASE"

# F3 canonical (test 2022-07-01 to 2022-12-22) on the S&P 500 panel.
# RAG-STAR full + corr_only ablation, 5 seeds each = 10 runs.
# Per-seed wall ~30-45 min on 550 tickers; total ~5-8h.
VARIANTS=(
  full
  corr_only
)
FOLD=3

for V in "${VARIANTS[@]}"; do
  for S in 42 43 44 45 46; do
    OUT="results/universal_validation_${V}/fold${FOLD}_seed${S}.json"
    if [ "$V" = "full" ]; then
      OUT="results/universal_validation/fold${FOLD}_seed${S}.json"
    fi
    LF="$LOG_BASE/${V}_fold${FOLD}_seed${S}.log"
    if [ -f "$OUT" ]; then
      echo "[$(date +%H:%M:%S)] skip $OUT"
      continue
    fi
    if [ "$V" = "full" ]; then
      CFG="configs/universal_validation_full.yaml"
    else
      CFG="configs/universal_validation_corr_only.yaml"
    fi
    echo "[$(date +%H:%M:%S)] variant=$V fold=$FOLD seed=$S"
    python3 -u -m src.v2.training.train_dow_epistar \
      --config "$CFG" --fold "$FOLD" --seed "$S" > "$LF" 2>&1
  done
done

echo "DONE $(date -Is)"
