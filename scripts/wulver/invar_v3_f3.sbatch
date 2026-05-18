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
mkdir -p logs/invar experiments/invar/headline_v3/fold3

echo "host=$(hostname) start=$(date -Is)"
nvidia-smi | head -10 || true

# InVAR v3 (surgical fixes), F3 only, 5 seeds. Flags identical to
# scripts/wulver/invar_phase3_v3_full.sbatch which produced the
# existing F1+F2 5-seed v3 numbers, with --fold switched to 3.
for S in 42 43 44 45 46; do
  OUT="experiments/invar/headline_v3/fold3/seed${S}_designR/results.json"
  if [ -f "$OUT" ]; then
    echo "[$(date +%H:%M:%S)] skip $OUT"
    continue
  fi
  LF="logs/invar/headline_v3_fold3_seed${S}.log"
  echo "[$(date +%H:%M:%S)] v3 fold=3 seed=$S"
  PYTHONPATH=$PWD python3 -u -m src.invar.training.train \
    --fold 3 --seed "$S" --epochs 10 \
    --output-dir "experiments/invar/headline_v3/fold3" \
    --regime-axis retrieval \
    --loss-config ranking \
    --bank-size 64 \
    --save-predictions \
    > "$LF" 2>&1
done

echo "DONE $(date -Is)"
