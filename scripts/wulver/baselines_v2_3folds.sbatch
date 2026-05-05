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
#SBATCH --mem=24G
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

echo "=== v2 baselines sweep: MASTER + StockMixer + DySTAGE on RAG-STAR protocol ==="
echo "host=$(hostname)  start=$(date -Is)"
nvidia-smi | head -20 || true

REPO="${REPO:-$HOME/phd-research}"
VENV="${VENV:-$HOME/phd-research-gpu-env}"
cd "$REPO"

module load bright 2>/dev/null || true
module load python3 2>/dev/null || true
source "$VENV/bin/activate"
python3 -c "import torch; print('torch', torch.__version__, 'cuda:', torch.cuda.is_available())"

LOG_BASE="logs/baselines_v2"
mkdir -p "$LOG_BASE"

# Step 1: build the DySTAGE graph cache once (shared across all DySTAGE folds/seeds).
CACHE="data/processed/dystage_graph_cache.pt"
if [ ! -f "$CACHE" ]; then
  echo "[$(date +%H:%M:%S)] building DySTAGE graph cache..."
  python3 -u -m src.baselines.dystage_graph_cache > "$LOG_BASE/graph_cache.log" 2>&1
fi

# Step 2: run all 3 baselines x 3 folds x 5 seeds = 45 runs.
for MODEL in master stockmixer dystage; do
  for FOLD in 1 2 3; do
    for S in 42 43 44 45 46; do
      OUT="results/baselines_244/${MODEL}_v2/fold${FOLD}_seed${S}.json"
      LF="$LOG_BASE/${MODEL}_fold${FOLD}_seed${S}.log"
      if [ -f "$OUT" ]; then echo "[$(date +%H:%M:%S)] skip $OUT"; continue; fi
      echo "[$(date +%H:%M:%S)] $MODEL fold=$FOLD seed=$S"
      python3 -u -m src.baselines.train_${MODEL}_v2 --fold "$FOLD" --seed "$S" \
        > "$LF" 2>&1
    done
  done
done

echo "[$(date +%H:%M:%S)] DONE"
echo "end=$(date -Is)"
