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
mkdir -p logs/mait experiments/invar/baselines_minimal

echo "host=$(hostname) start=$(date -Is)"
nvidia-smi | head -10 || true

# MAiT minimal-macro retry: 8-feature regime-discriminating macro subset
# per docs/macro_feature_analysis.md. Pivoted 2026-05-11 from the
# 17-feature MAiT (F2 5-seed mean -0.0064, below the 0.025 bar).
#
# Output goes to experiments/invar/baselines_minimal/ to keep the failed
# 17-feature MAiT results in experiments/invar/baselines/ for the paper's
# negative-result discussion.
#
# 5 seeds x 3 folds = 15 runs. Each up to 30 epochs with patience-5
# early stop; F1 patterns from yesterday show ~5 min/run wall.
for FOLD in 1 2 3; do
  for S in 42 43 44 45 46; do
    OUT="experiments/invar/baselines_minimal/mait/fold${FOLD}/seed${S}/results.json"
    if [ -f "$OUT" ]; then
      echo "[$(date +%H:%M:%S)] skip $OUT"
      continue
    fi
    LF="logs/mait/minimal_fold${FOLD}_seed${S}.log"
    echo "[$(date +%H:%M:%S)] mait_minimal fold=$FOLD seed=$S"
    PYTHONPATH=$PWD python3 -u -m src.invar.baselines.train_baseline \
      --baseline mait \
      --fold "$FOLD" --seed "$S" --epochs 30 \
      --output-dir experiments/invar/baselines_minimal \
      --save-predictions \
      > "$LF" 2>&1
  done
done

echo "DONE $(date -Is)"
