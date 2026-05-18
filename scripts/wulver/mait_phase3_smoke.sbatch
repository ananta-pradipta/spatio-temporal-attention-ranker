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
cd $HOME/phd-research
module load bright 2>/dev/null || true
module load python3 2>/dev/null || true
source $HOME/phd-research-gpu-env/bin/activate
mkdir -p logs/mait

echo "host=$(hostname) start=$(date -Is)"

# Phase 3 smoke: iTransformer 2 epochs (regression check), then MAiT 2 epochs.
# Output dir kept separate from production baselines to avoid mixing 2-epoch
# smoke results with the 30-epoch headline numbers.

echo "=== iTransformer regression smoke (2 epochs) ==="
LF1="logs/mait/phase3_smoke_itransformer.log"
PYTHONPATH=$PWD python3 -u -m src.invar.baselines.train_baseline \
  --baseline itransformer --fold 1 --seed 42 --epochs 2 \
  --output-dir experiments/invar/baselines_phase3_smoke \
  2>&1 | tee "$LF1"

echo "=== MAiT smoke (2 epochs) ==="
LF2="logs/mait/phase3_smoke_mait.log"
PYTHONPATH=$PWD python3 -u -m src.invar.baselines.train_baseline \
  --baseline mait --fold 1 --seed 42 --epochs 2 \
  --output-dir experiments/invar/baselines_phase3_smoke \
  2>&1 | tee "$LF2"

echo "DONE $(date -Is)"
