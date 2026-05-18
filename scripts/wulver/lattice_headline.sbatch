#!/bin/bash -l
#SBATCH --job-name=lattice-headline
#SBATCH --partition=gpu
#SBATCH --account=hz79
#SBATCH --qos=standard
#SBATCH --time=14:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --mem=32G
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
cd $HOME/phd-research
module load bright 2>/dev/null || true
module load python3 2>/dev/null || true
source $HOME/phd-research-gpu-env/bin/activate
mkdir -p logs/lattice

echo "host=$(hostname) start=$(date -Is)"
nvidia-smi | head -10 || true

# Headline sweep: 3 folds × 5 seeds = 15 runs.
# Pace target: ~30 min/run -> ~7.5h. Walltime cap 14h leaves margin.
for FOLD in 1 2 3; do
  for S in 42 43 44 45 46; do
    OUT="experiments/lattice/headline/fold${FOLD}_seed${S}/results.json"
    if [ -f "$OUT" ]; then
      echo "[$(date +%H:%M:%S)] skip $OUT"
      continue
    fi
    LF="logs/lattice/headline_fold${FOLD}_seed${S}.log"
    echo "[$(date +%H:%M:%S)] fold=$FOLD seed=$S"
    python3 -u -m src.lattice.training.train \
      --fold "$FOLD" --seed "$S" --epochs 10 \
      --output-dir experiments/lattice/headline > "$LF" 2>&1
  done
done

echo "DONE $(date -Is)"
