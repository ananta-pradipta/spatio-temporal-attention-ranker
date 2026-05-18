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
mkdir -p logs/invar_star experiments/invar_star/fold1

echo "host=$(hostname) start=$(date -Is)"
nvidia-smi | head -10 || true

# InVAR-STAR full schedule, F1 seed 42 only. ~60 epochs, swa_start 40,
# tau anneals 1.0 -> 0.1 smoothly over the run (vs the compressed
# schedule of the smoke). Diagnostic for whether beta recovers
# regime-adaptive behaviour over more training time, or stays collapsed
# in the beta-near-zero basin as the 8-epoch smokes all showed.
LF="logs/invar_star/f1_full_s42.log"
PYTHONPATH=$PWD python3 -u -m src.invar_star.train \
  --fold 1 --seed 42 \
  --config configs/invar_star.yaml \
  --out-dir experiments/invar_star/fold1 \
  > "$LF" 2>&1

echo "DONE $(date -Is)"
