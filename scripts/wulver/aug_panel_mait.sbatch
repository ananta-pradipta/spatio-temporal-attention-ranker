#!/bin/bash -l
#SBATCH --job-name=model-eval
#SBATCH --partition=gpu
#SBATCH --account=hz79
#SBATCH --qos=standard
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --mem=32G
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

# Select the augmented 37-feature panel + matching scalers.
export LATTICE_PROCESSED_DIR=data/lattice_aug/processed
export LATTICE_SCALER_DIR=experiments/lattice_aug

# Keep the 17-feature macro preset (original MAiT spec). The minimal preset
# was sidelined after the macro pruning sweep didn't lift F2.
export MAIT_PRESET=v17

cd $HOME/phd-research
module load bright 2>/dev/null || true
module load python3 2>/dev/null || true
source $HOME/phd-research-gpu-env/bin/activate
mkdir -p logs/invar_aug experiments/invar_aug/baselines

echo "host=$(hostname) start=$(date -Is)"
echo "LATTICE_PROCESSED_DIR=$LATTICE_PROCESSED_DIR"
echo "LATTICE_SCALER_DIR=$LATTICE_SCALER_DIR"
echo "MAIT_PRESET=$MAIT_PRESET"
nvidia-smi | head -10 || true

# Augmented-panel MAiT-17 sweep: F1 + F2 + F3 x 5 seeds = 15 runs at
# ~5-10 min per seed. The 11 new panel features automatically flow to MAiT
# via MaitBatchAdapter (drops catalyst_type_id + has_stocktwits, leaves 35).
for FOLD in 1 2 3; do
  for S in 42 43 44 45 46; do
    OUT="experiments/invar_aug/baselines/mait/fold${FOLD}/seed${S}/results.json"
    if [ -f "$OUT" ]; then
      echo "[$(date +%H:%M:%S)] skip $OUT"
      continue
    fi
    LF="logs/invar_aug/mait_v17_fold${FOLD}_seed${S}.log"
    echo "[$(date +%H:%M:%S)] mait-v17 (aug) fold=$FOLD seed=$S"
    PYTHONPATH=$PWD python3 -u -m src.invar.baselines.train_baseline \
      --baseline mait \
      --fold "$FOLD" --seed "$S" --epochs 30 \
      --output-dir experiments/invar_aug/baselines \
      --save-predictions \
      > "$LF" 2>&1
  done
done

echo "DONE $(date -Is)"
