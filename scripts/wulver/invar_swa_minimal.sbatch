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
mkdir -p logs/invar experiments/invar/swa_minimal

echo "host=$(hostname) start=$(date -Is)"
nvidia-smi | head -10 || true

# SWA-InVAR with the 8-feature minimal macro subset per
# docs/macro_feature_analysis.md. Identical flag profile to the canonical
# scripts/wulver/invar_swa.sbatch (which produced the +0.0236 F1+F2 pooled
# headline) plus --macro-subset minimal and a separate output dir.
#
# Hypothesis: with 16 of 24 macro features being noise on F2 (max |z| < 2
# across all folds), the MarketGateV2 + MacroEncoder + RegimeAxisRetrieval
# pathway should be at least as strong, and potentially stronger, on F2
# with a 3x-smaller but cleaner macro vocabulary.
#
# 5 seeds x 3 folds = 15 runs. Per-seed wall ~25-30 min on A100, total ~6h.
# F3 included because we want the full 3-fold comparison; SWA-InVAR's F3 was
# -0.0054 (5-seed mean) with full 24 macro and we want to see if the
# minimal subset moves F3.
for FOLD in 1 2 3; do
  for S in 42 43 44 45 46; do
    OUT_DIR="experiments/invar/swa_minimal/fold${FOLD}"
    OUT="${OUT_DIR}/seed${S}_designR/results.json"
    if [ -f "$OUT" ]; then
      echo "[$(date +%H:%M:%S)] skip $OUT"
      continue
    fi
    LF="logs/invar/swa_minimal_f${FOLD}_s${S}.log"
    echo "[$(date +%H:%M:%S)] swa-invar minimal fold=$FOLD seed=$S"
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
      --macro-subset minimal \
      --save-predictions \
      > "$LF" 2>&1
  done
done

echo "DONE $(date -Is)"
