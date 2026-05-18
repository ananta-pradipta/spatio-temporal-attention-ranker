#!/bin/bash -l
#SBATCH --job-name=invar-phase-3-master-redo
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
cd $HOME/phd-research
module load bright 2>/dev/null || true
module load python3 2>/dev/null || true
source $HOME/phd-research-gpu-env/bin/activate
mkdir -p logs/invar experiments/invar/baselines

echo "host=$(hostname) start=$(date -Is)"
nvidia-smi | head -10 || true

# Redo MASTER + the 2 contaminated StockMixer F2 seeds with the
# corrected ranking-aware defaults (huber 1.0 + listwise 0.5 +
# pairwise 0.3, no aux heads).
for FOLD in 1 2; do
  for S in 42 43 44 45 46; do
    OUT="experiments/invar/baselines/master/fold${FOLD}/seed${S}/results.json"
    if [ -f "$OUT" ]; then
      echo "[$(date +%H:%M:%S)] skip $OUT"
      continue
    fi
    LF="logs/invar/baseline_master_fold${FOLD}_seed${S}.log"
    PYTHONPATH=$PWD python3 -u -m src.invar.baselines.train_baseline \
      --baseline master \
      --fold "$FOLD" --seed "$S" --epochs 10 \
      --output-dir experiments/invar/baselines \
      --save-predictions \
      > "$LF" 2>&1
  done
done

# Redo StockMixer F2 s45 and s46 (post-commit contamination).
for S in 45 46; do
  OUT="experiments/invar/baselines/stockmixer/fold2/seed${S}/results.json"
  if [ -f "$OUT" ]; then
    echo "[$(date +%H:%M:%S)] skip $OUT"
    continue
  fi
  LF="logs/invar/baseline_stockmixer_fold2_seed${S}.log"
  PYTHONPATH=$PWD python3 -u -m src.invar.baselines.train_baseline \
    --baseline stockmixer \
    --fold 2 --seed "$S" --epochs 10 \
    --output-dir experiments/invar/baselines \
    --save-predictions \
    > "$LF" 2>&1
done

echo "DONE $(date -Is)"
