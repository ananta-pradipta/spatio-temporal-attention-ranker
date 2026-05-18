#!/bin/bash -l
#SBATCH --job-name=invar-phase-3-baselines-tier1
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
cd $HOME/phd-research
module load bright 2>/dev/null || true
module load python3 2>/dev/null || true
source $HOME/phd-research-gpu-env/bin/activate
mkdir -p logs/invar experiments/invar/baselines

echo "host=$(hostname) start=$(date -Is)"
nvidia-smi | head -10 || true

# Tier 1 baselines (iTransformer, StockMixer, MASTER) on F1 + F2 x 5 seeds.
# 3 baselines x 2 folds x 5 seeds = 30 runs at approximately 8 to 12 min
# each on A100 -> approximately 4 to 6h. Walltime cap 12h.
for BASELINE in itransformer stockmixer master; do
  for FOLD in 1 2; do
    for S in 42 43 44 45 46; do
      OUT="experiments/invar/baselines/${BASELINE}/fold${FOLD}/seed${S}/results.json"
      if [ -f "$OUT" ]; then
        echo "[$(date +%H:%M:%S)] skip $OUT"
        continue
      fi
      LF="logs/invar/baseline_${BASELINE}_fold${FOLD}_seed${S}.log"
      echo "[$(date +%H:%M:%S)] baseline=$BASELINE fold=$FOLD seed=$S"
      PYTHONPATH=$PWD python3 -u -m src.invar.baselines.train_baseline \
        --baseline "$BASELINE" \
        --fold "$FOLD" --seed "$S" --epochs 10 \
        --output-dir experiments/invar/baselines \
        --save-predictions \
        > "$LF" 2>&1
    done
  done
done

echo "DONE $(date -Is)"
