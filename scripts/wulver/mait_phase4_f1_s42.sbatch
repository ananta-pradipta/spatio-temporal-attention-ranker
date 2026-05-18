#!/bin/bash -l
#SBATCH --job-name=model-eval
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
cd $HOME/phd-research
module load bright 2>/dev/null || true
module load python3 2>/dev/null || true
source $HOME/phd-research-gpu-env/bin/activate
mkdir -p logs/mait

echo "host=$(hostname) start=$(date -Is)"

# Phase 4: MAiT full 30-epoch single-fold smoke (F1 seed 42).
# Output dir matches the production baselines so the Phase 5 aggregator
# can pick this row up alongside the 5-seed sweep.
LF="logs/mait/phase4_f1_s42.log"
PYTHONPATH=$PWD python3 -u -m src.invar.baselines.train_baseline \
  --baseline mait --fold 1 --seed 42 --epochs 30 \
  --output-dir experiments/invar/baselines --save-predictions \
  2>&1 | tee "$LF"

echo "DONE $(date -Is)"
