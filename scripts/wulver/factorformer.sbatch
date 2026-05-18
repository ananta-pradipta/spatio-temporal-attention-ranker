#!/bin/bash -l
#SBATCH --job-name=model-eval
#SBATCH --partition=gpu
#SBATCH --account=hz79
#SBATCH --qos=standard
#SBATCH --time=08:00:00
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
mkdir -p logs/factorformer experiments/factorformer

echo "host=$(hostname) start=$(date -Is)"
nvidia-smi | head -10 || true

# FactorFormer full sweep: 5 seeds x 3 folds = 15 runs. Skips already
# completed runs (e.g., the smoke F1 seed 42 already done). Fold 3 runs
# require --allow-fold-3 (the dataset gate is on the WAVE side; this
# trainer also enforces explicit opt-in).
for FOLD in 1 2 3; do
  for S in 42 43 44 45 46; do
    OUT="experiments/factorformer/fold${FOLD}/seed${S}_designFF/results.json"
    if [ -f "$OUT" ]; then
      echo "[$(date +%H:%M:%S)] skip $OUT"
      continue
    fi
    LF="logs/factorformer/ff_f${FOLD}_s${S}.log"
    echo "[$(date +%H:%M:%S)] factorformer fold=$FOLD seed=$S"
    PYTHONPATH=$PWD python3 -u -m src.factorformer.train \
      --fold "$FOLD" --seed "$S" --epochs 10 \
      --output-dir "experiments/factorformer/fold${FOLD}" \
      --kl-weight 1.0e-3 \
      --n-factors 8 \
      --save-predictions \
      --allow-fold-3 \
      > "$LF" 2>&1
  done
done

echo "DONE $(date -Is)"
