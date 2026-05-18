#!/bin/bash -l
#SBATCH --job-name=model-eval
#SBATCH --partition=gpu
#SBATCH --account=hz79
#SBATCH --qos=standard
#SBATCH --time=10:00:00
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
mkdir -p logs/invar experiments/invar/diff_retrieval

echo "host=$(hostname) start=$(date -Is)"
nvidia-smi | head -10 || true

# InVAR differentiable retrieval ablation: 4 modes x 5 seeds x 2 folds
# = 40 runs. Per-run target ~7-10 min on A100 -> ~6h budget; cap 10h.
#
# All non-ablation flags MUST match invar_v6_f1_f2.sbatch (the canonical
# v6 headline sbatch) so this is a v6-comparable ablation. The only
# ablation knob is --retrieval-mode.
#
# Critical fix from the 2026-05-09 first-attempt sbatch:
#   - Was missing --loss-config ranking (was using TrainConfig default
#     loss_config="minimal" which is Huber-only, no listwise/pairwise)
#   - Was incorrectly enabling --use-market-gate (the v4 gate; v6 uses
#     only v2 via --use-market-gate-v2)
#   - Was incorrectly enabling --use-regime-loss (v6 headline does not)
#   - Was using --epochs 10 (v6 uses 30 with patience-3 early stop)
for MODE in hard_topk softmax_full softmax_topk gumbel_topk; do
  for FOLD in 1 2; do
    for S in 42 43 44 45 46; do
      OUT_DIR="experiments/invar/diff_retrieval/${MODE}/fold${FOLD}"
      OUT="${OUT_DIR}/seed${S}_designR/results.json"
      if [ -f "$OUT" ]; then
        echo "[$(date +%H:%M:%S)] skip $OUT"
        continue
      fi
      LF="logs/invar/diff_retrieval_${MODE}_f${FOLD}_s${S}.log"
      echo "[$(date +%H:%M:%S)] mode=${MODE} fold=${FOLD} seed=${S}"
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
        --retrieval-mode "$MODE" \
        --save-predictions \
        > "$LF" 2>&1
    done
  done
done

echo "DONE $(date -Is)"
