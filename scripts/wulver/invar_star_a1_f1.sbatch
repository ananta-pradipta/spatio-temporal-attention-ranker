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
mkdir -p logs/invar_star experiments/invar_star_a1/fold1

echo "host=$(hostname) start=$(date -Is)"
nvidia-smi | head -10 || true

# InVAR-STAR A1 ablation, F1 seed 42, full 60 epochs. Gate is bypassed
# (--fixed-beta 1.0), so macro tokens are always engaged in stock-row
# attention and the macro_proj(phi) term in the MoE router is fully
# present. The throttle MLP receives no gradient (gate not called) and
# stays frozen. lambda_throttle = 0 in the config anyway.
#
# Diagnostic question: is the macro pathway capable of being useful on
# F1 when the gate is not allowed to collapse? Decision rule (per
# 2026-05-10 discussion):
#   - test rank IC >= +0.0180: macro IS useful, gate is the bug.
#       Next: run gate warmup.
#   - test rank IC <= +0.0120: macro NOT useful on F1, sideline or
#       reframe. Skip 3b.
#   - In between: ambiguous, run F2 A1 next.
LF="logs/invar_star/a1_f1_s42.log"
PYTHONPATH=$PWD python3 -u -m src.invar_star.train \
  --fold 1 --seed 42 \
  --config configs/invar_star.yaml \
  --out-dir experiments/invar_star_a1/fold1 \
  --fixed-beta 1.0 \
  > "$LF" 2>&1

echo "DONE $(date -Is)"
