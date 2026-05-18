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
mkdir -p logs/invar_star experiments/invar_star_a1/fold2

echo "host=$(hostname) start=$(date -Is)"
nvidia-smi | head -10 || true

# InVAR-STAR A1 ablation, F2 seed 42, full 60 epochs.
# Same setup as scripts/wulver/invar_star_a1_f1.sbatch with --fold 2.
# F2 test window: 2021-07 to 2022-06 (rate stress, the regime where the
# design predicted macro engagement is strongest).
#
# Decision rule on test rank IC (sharpened 2026-05-10):
#   - >= +0.0220: macro pathway is the F2 strength; SWA-InVAR's gate is
#       the bug. Reframe the model line around fixed-beta or gate
#       warmup.
#   - <= +0.0150 (or near vanilla iTransformer F2 baseline +0.0031):
#       macro pathway claim is broadly falsified, not just on F1.
#       Sideline the design as a negative result.
#   - In between: deeper architectural diagnosis needed.
LF="logs/invar_star/a1_f2_s42.log"
PYTHONPATH=$PWD python3 -u -m src.invar_star.train \
  --fold 2 --seed 42 \
  --config configs/invar_star.yaml \
  --out-dir experiments/invar_star_a1/fold2 \
  --fixed-beta 1.0 \
  > "$LF" 2>&1

echo "DONE $(date -Is)"
