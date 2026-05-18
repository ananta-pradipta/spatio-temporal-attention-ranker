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
mkdir -p logs/invar_star experiments/invar_star_a1/fold3

echo "host=$(hostname) start=$(date -Is)"
nvidia-smi | head -10 || true

# InVAR-STAR A1 ablation, F3 seed 42, full 60 epochs.
# Same setup as scripts/wulver/invar_star_a1_f{1,2}.sbatch with --fold 3
# (with --allow-fold-3). F3 test window: 2022-07 to 2022-12 (post-stress
# stabilisation).
#
# Note: design predicts beta near 0 on F3 (macro disengaged), so forcing
# beta = 1 on F3 should HURT vs vanilla iTransformer's F3 +0.0299
# rank IC. This run completes the F1+F2+F3 A1 picture before deciding
# the model line's final fate.
LF="logs/invar_star/a1_f3_s42.log"
PYTHONPATH=$PWD python3 -u -m src.invar_star.train \
  --fold 3 --seed 42 \
  --config configs/invar_star.yaml \
  --out-dir experiments/invar_star_a1/fold3 \
  --fixed-beta 1.0 \
  --allow-fold-3 \
  > "$LF" 2>&1

echo "DONE $(date -Is)"
