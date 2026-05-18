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
cd $HOME/phd-research
module load bright 2>/dev/null || true
module load python3 2>/dev/null || true
source $HOME/phd-research-gpu-env/bin/activate
mkdir -p logs/itransformer_reeval experiments/itransformer_reeval

echo "host=$(hostname) start=$(date -Is)"
nvidia-smi | head -10 || true

# iTransformer re-eval, 5 seeds x 3 folds. Identical CLI to the existing
# scripts/wulver/itrans_redo.sbatch except (a) fold loop now includes 3 and
# (b) output dir separated so the original baseline numbers in
# experiments/invar/baselines/itransformer/ are not overwritten. Used as
# the reproducibility check before adopting iTransformer as the foundation
# for a new model line.
for FOLD in 1 2 3; do
  for S in 42 43 44 45 46; do
    OUT="experiments/itransformer_reeval/itransformer/fold${FOLD}/seed${S}/results.json"
    if [ -f "$OUT" ]; then
      echo "[$(date +%H:%M:%S)] skip $OUT"
      continue
    fi
    LF="logs/itransformer_reeval/fold${FOLD}_seed${S}.log"
    echo "[$(date +%H:%M:%S)] itransformer fold=$FOLD seed=$S"
    PYTHONPATH=$PWD python3 -u -m src.invar.baselines.train_baseline \
      --baseline itransformer --fold "$FOLD" --seed "$S" --epochs 10 \
      --output-dir experiments/itransformer_reeval \
      --save-predictions \
      > "$LF" 2>&1
  done
done

echo "DONE $(date -Is)"
