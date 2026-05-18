#!/bin/bash -l
#SBATCH --job-name=model-eval
#SBATCH --partition=gpu
#SBATCH --account=hz79
#SBATCH --qos=standard
#SBATCH --time=12:00:00
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

# Select the augmented 37-feature panel + matching scalers.
export LATTICE_PROCESSED_DIR=data/lattice_aug/processed
export LATTICE_SCALER_DIR=experiments/lattice_aug

cd $HOME/phd-research
module load bright 2>/dev/null || true
module load python3 2>/dev/null || true
source $HOME/phd-research-gpu-env/bin/activate
mkdir -p logs/invar_aug experiments/invar_aug/swa

echo "host=$(hostname) start=$(date -Is)"
echo "LATTICE_PROCESSED_DIR=$LATTICE_PROCESSED_DIR"
echo "LATTICE_SCALER_DIR=$LATTICE_SCALER_DIR"
nvidia-smi | head -10 || true

# Augmented-panel SWA-InVAR: 5 seeds x F1+F2+F3 = 15 runs. Same flags as the
# canonical scripts/wulver/invar_swa.sbatch; only difference is the
# 37-feature panel (Tier A1+A2 additions) selected via the env vars above.
# Per-seed wall ~25-30 min; total ~7h.
for FOLD in 1 2 3; do
  for S in 42 43 44 45 46; do
    OUT_DIR="experiments/invar_aug/swa/fold${FOLD}"
    OUT="${OUT_DIR}/seed${S}_designR/results.json"
    if [ -f "$OUT" ]; then
      echo "[$(date +%H:%M:%S)] skip $OUT"
      continue
    fi
    LF="logs/invar_aug/swa_f${FOLD}_s${S}.log"
    echo "[$(date +%H:%M:%S)] swa-invar (aug) fold=$FOLD seed=$S"
    PYTHONPATH=$PWD python3 -u -m src.invar.training.train \
      --fold "$FOLD" --seed "$S" --epochs 30 \
      --output-dir "$OUT_DIR" \
      --regime-axis retrieval --loss-config ranking --bank-size 64 \
      --use-market-gate-v2 \
      --macro-encoder-mode mlp_flat \
      --macro-state-dim 64 \
      --market-gate-v2-form softmax_F \
      --market-gate-v2-hidden-dim 64 \
      --use-dynamic-bank-controller \
      --bank-controller-mode hybrid \
      --bank-controller-min-weight 0.05 \
      --bank-controller-max-weight 1.00 \
      --retrieval-mode gumbel_topk \
      --use-swa --swa-decay 0.995 --swa-warmup-epochs 0 \
      --save-predictions \
      > "$LF" 2>&1
  done
done

echo "DONE $(date -Is)"
