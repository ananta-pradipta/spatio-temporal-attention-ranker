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
mkdir -p logs/wave results/wave_v3

echo "host=$(hostname) start=$(date -Is)"
nvidia-smi | head -10 || true

# WAVE v3, F3 only, 5 seeds. Config identical to scripts/wulver/wave_v3.sbatch
# which produced the existing F1+F2 5-seed v3 numbers, with --fold switched to 3.
for seed in 42 43 44 45 46; do
  LF="logs/wave/wave_v3_f3_s${seed}.log"
  echo "[$(date -Is)] v3 fold=3 seed=${seed}"
  PYTHONPATH=$PWD python3 -u -m src.wave.train \
    --fold 3 --seed "$seed" \
    --config configs/wave_v3.yaml \
    --output-dir results/wave_v3 \
    --allow-fold-3 \
    > "$LF" 2>&1
done

PYTHONPATH=$PWD python3 -c "
from pathlib import Path
from src.wave.eval import write_metrics_json
write_metrics_json(Path('results/wave_v3/fold3'), [42, 43, 44, 45, 46])
"

echo "DONE $(date -Is)"
