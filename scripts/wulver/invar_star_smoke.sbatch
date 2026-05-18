#!/bin/bash -l
#SBATCH --job-name=model-eval
#SBATCH --partition=gpu
#SBATCH --account=hz79
#SBATCH --qos=standard
#SBATCH --time=01:00:00
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
mkdir -p logs/invar_star experiments/invar_star/fold1

echo "host=$(hostname) start=$(date -Is)"
nvidia-smi | head -10 || true

# InVAR-STAR smoke: 8 epochs on F1 seed 42. Phase 2 acceptance test, with
# enough epochs that tau and lr schedules are not collapsed into the same
# step. With smoke=8 the loop sets swa_start to 7, so epochs 0..6 run with
# tau annealing from 1.0 to ~0.23 and the normal AdamW schedule, then
# epoch 7 transitions to SWA. We expect monotonic loss decrease across
# epochs 0..6 and beta variation tied to phi (not pinned at 0 or 1).
LF="logs/invar_star/smoke8_f1_s42.log"
PYTHONPATH=$PWD python3 -u -m src.invar_star.train \
  --fold 1 --seed 42 --smoke 8 \
  --config configs/invar_star.yaml \
  --out-dir experiments/invar_star/fold1 \
  > "$LF" 2>&1

echo "DONE $(date -Is)"
