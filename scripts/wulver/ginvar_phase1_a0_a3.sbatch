#!/bin/bash -l
#SBATCH --job-name=ginvar-phase-1
#SBATCH --partition=gpu
#SBATCH --account=hz79
#SBATCH --qos=standard
#SBATCH --time=02:00:00
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

# A0: dense inverted attention baseline (no graph). F1 seed 42 only.
LF="logs/ginvar/a0_dense_fold1_seed42.log"
PYTHONPATH=$PWD python3 -u -m scripts.lattice.train_ginvar \
  --fold 1 --seed 42 --epochs 10 \
  --output-dir experiments/ginvar/a0_dense \
  --graph-mode dense \
  --save-predictions \
  > "$LF" 2>&1

# A3: sector graph + graph_bias_and_mask, top_k=16. F1 seed 42 only.
LF="logs/ginvar/a3_sector_bias_mask_fold1_seed42.log"
PYTHONPATH=$PWD python3 -u -m scripts.lattice.train_ginvar \
  --fold 1 --seed 42 --epochs 10 \
  --output-dir experiments/ginvar/a3_sector_bias_mask \
  --graph-mode graph_bias_and_mask --top-k 16 \
  --use-sector-graph \
  --save-predictions \
  > "$LF" 2>&1

echo "DONE $(date -Is)"
