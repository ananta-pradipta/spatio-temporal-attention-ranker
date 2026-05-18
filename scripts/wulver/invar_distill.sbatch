#!/bin/bash -l
#SBATCH --job-name=model-eval
#SBATCH --partition=gpu
#SBATCH --account=hz79
#SBATCH --qos=standard
#SBATCH --time=08:00:00
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
mkdir -p logs/invar experiments/invar/distill

echo "host=$(hostname) start=$(date -Is)"
nvidia-smi | head -10 || true

# Self-distillation pass: SWA-InVAR teacher (frozen) -> SWA-InVAR student
# (with distill loss). Identical flag set to invar_swa.sbatch except for the
# three new --teacher-ckpt / --distill-weight / --distill-temp flags.
#
# Teacher checkpoint per (fold, seed): experiments/invar/swa/fold{F}/seed{S}_designR/ckpt.pt
# Output:                                experiments/invar/distill/fold{F}/seed{S}_designR/
#
# 5 seeds * 2 folds = 10 runs.
for FOLD in 1 2; do
  for S in 42 43 44 45 46; do
    OUT_DIR="experiments/invar/distill/fold${FOLD}"
    OUT="${OUT_DIR}/seed${S}_designR/results.json"
    TEACHER="experiments/invar/swa/fold${FOLD}/seed${S}_designR/ckpt.pt"
    if [ -f "$OUT" ]; then
      echo "[$(date +%H:%M:%S)] skip $OUT"
      continue
    fi
    if [ ! -f "$TEACHER" ]; then
      echo "[$(date +%H:%M:%S)] missing teacher $TEACHER, skip"
      continue
    fi
    LF="logs/invar/distill_f${FOLD}_s${S}.log"
    echo "[$(date +%H:%M:%S)] distill-invar fold=$FOLD seed=$S teacher=$TEACHER"
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
      --teacher-ckpt "$TEACHER" \
      --distill-weight 0.5 \
      --distill-temp 2.0 \
      --save-predictions \
      > "$LF" 2>&1
  done
done

echo "DONE $(date -Is)"
