#!/bin/bash -l
#SBATCH --job-name=ginvar-a0-a3-full
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
mkdir -p logs/ginvar experiments/ginvar/{a0_dense,a3_sector_bias_mask}

echo "host=$(hostname) start=$(date -Is)"
nvidia-smi | head -10 || true

# G-InVAR A0 (dense, no graph) and A3 (sector + graph_bias_and_mask, top_k=16).
# F1 + F2 x 5 seeds = 20 runs. Existing F1 seed 42 results auto-skip.
for ABL in a0_dense a3_sector_bias_mask; do
  if [ "$ABL" = "a0_dense" ]; then
    GRAPH_MODE=dense
    USE_SECTOR=""
  else
    GRAPH_MODE=graph_bias_and_mask
    USE_SECTOR="--use-sector-graph"
  fi
  for FOLD in 1 2; do
    for S in 42 43 44 45 46; do
      OUT="experiments/ginvar/${ABL}/fold${FOLD}/seed${S}/results.json"
      if [ -f "$OUT" ]; then
        echo "[$(date +%H:%M:%S)] skip $OUT"
        continue
      fi
      LF="logs/ginvar/${ABL}_fold${FOLD}_seed${S}.log"
      echo "[$(date +%H:%M:%S)] $ABL fold=$FOLD seed=$S"
      PYTHONPATH=$PWD python3 -u -m scripts.lattice.train_ginvar \
        --fold "$FOLD" --seed "$S" --epochs 10 \
        --output-dir "experiments/ginvar/${ABL}" \
        --graph-mode "$GRAPH_MODE" --top-k 16 \
        $USE_SECTOR \
        --save-predictions \
        > "$LF" 2>&1
    done
  done
done

echo "DONE $(date -Is)"
