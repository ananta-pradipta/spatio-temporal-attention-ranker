#!/bin/bash -l
#SBATCH --job-name=ginvar-full
#SBATCH --partition=gpu
#SBATCH --account=hz79
#SBATCH --qos=standard
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --mem=64G
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
cd $HOME/phd-research
module load bright 2>/dev/null || true
module load python3 2>/dev/null || true
source $HOME/phd-research-gpu-env/bin/activate
mkdir -p logs/ginvar experiments/ginvar/full

echo "host=$(hostname) start=$(date -Is)"
nvidia-smi | head -10 || true

# Full G-InVAR: sector + correlation + factor + social, regime-blended.
# graph_mode=graph_bias_and_mask, top_k=16. F1+F2 x 5 seeds = 10 runs.
for FOLD in 1 2; do
  for S in 42 43 44 45 46; do
    OUT="experiments/ginvar/full/fold${FOLD}/seed${S}/results.json"
    if [ -f "$OUT" ]; then
      echo "[$(date +%H:%M:%S)] skip $OUT"
      continue
    fi
    LF="logs/ginvar/full_fold${FOLD}_seed${S}.log"
    echo "[$(date +%H:%M:%S)] full fold=$FOLD seed=$S"
    PYTHONPATH=$PWD python3 -u -m scripts.lattice.train_ginvar \
      --fold "$FOLD" --seed "$S" --epochs 10 \
      --output-dir "experiments/ginvar/full" \
      --graph-mode graph_bias_and_mask --top-k 16 \
      --use-sector-graph --use-corr-graph \
      --use-factor-graph --use-social-graph \
      --save-predictions \
      > "$LF" 2>&1
  done
done

echo "DONE $(date -Is)"
