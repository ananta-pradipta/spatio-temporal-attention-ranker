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
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
cd $HOME/phd-research
module load bright 2>/dev/null || true
module load python3 2>/dev/null || true
source $HOME/phd-research-gpu-env/bin/activate
mkdir -p logs/invar
mkdir -p experiments/invar/swa_no_bank experiments/invar/swa_no_gate experiments/invar/swa_no_gate_no_bank

echo "host=$(hostname) start=$(date -Is)"
nvidia-smi | head -10 || true

# Fold 3 component-removal ablation, matched to the SWA-InVAR active head.
# Hypothesis: the macro-conditioned RegimeAxisRetrieval bank is the F3 killer.
# If we disable the bank, F3 should recover; if we disable just the gate,
# F3 should still collapse.
#
# Three new cells, each on F3 with 5 seeds:
#   1. swa_no_bank        : gate ON, bank OFF, SWA ON
#   2. swa_no_gate        : gate OFF, bank ON (gumbel_topk), SWA ON
#   3. swa_no_gate_no_bank: gate OFF, bank OFF, SWA ON
#
# Compare against the existing full SWA-InVAR active head F3 (mean -0.0054).
# 3 cells * 5 seeds = 15 runs at ~5-7 min each -> ~80 min total.
FOLD=3
for S in 42 43 44 45 46; do
  # --- Cell 1: gate ON, bank OFF, SWA ON ---
  OUT_DIR="experiments/invar/swa_no_bank/fold${FOLD}"
  OUT="${OUT_DIR}/seed${S}_designR/results.json"
  if [ ! -f "$OUT" ]; then
    LF="logs/invar/swa_no_bank_f${FOLD}_s${S}.log"
    echo "[$(date +%H:%M:%S)] swa_no_bank fold=$FOLD seed=$S"
    PYTHONPATH=$PWD python3 -u -m src.invar.training.train \
      --fold "$FOLD" --seed "$S" --epochs 30 \
      --output-dir "$OUT_DIR" \
      --regime-axis retrieval --loss-config ranking --bank-size 64 \
      --use-market-gate-v2 \
      --macro-encoder-mode mlp_flat \
      --macro-state-dim 64 \
      --market-gate-v2-form softmax_F \
      --market-gate-v2-hidden-dim 64 \
      --disable-bank \
      --use-swa --swa-decay 0.995 --swa-warmup-epochs 0 \
      --save-predictions \
      > "$LF" 2>&1
  fi

  # --- Cell 2: gate OFF, bank ON (gumbel_topk), SWA ON ---
  OUT_DIR="experiments/invar/swa_no_gate/fold${FOLD}"
  OUT="${OUT_DIR}/seed${S}_designR/results.json"
  if [ ! -f "$OUT" ]; then
    LF="logs/invar/swa_no_gate_f${FOLD}_s${S}.log"
    echo "[$(date +%H:%M:%S)] swa_no_gate fold=$FOLD seed=$S"
    PYTHONPATH=$PWD python3 -u -m src.invar.training.train \
      --fold "$FOLD" --seed "$S" --epochs 30 \
      --output-dir "$OUT_DIR" \
      --regime-axis retrieval --loss-config ranking --bank-size 64 \
      --use-dynamic-bank-controller \
      --bank-controller-mode hybrid \
      --bank-controller-min-weight 0.05 \
      --bank-controller-max-weight 1.00 \
      --retrieval-mode gumbel_topk \
      --use-swa --swa-decay 0.995 --swa-warmup-epochs 0 \
      --save-predictions \
      > "$LF" 2>&1
  fi

  # --- Cell 3: gate OFF, bank OFF, SWA ON (= bare backbone with SWA) ---
  OUT_DIR="experiments/invar/swa_no_gate_no_bank/fold${FOLD}"
  OUT="${OUT_DIR}/seed${S}_designR/results.json"
  if [ ! -f "$OUT" ]; then
    LF="logs/invar/swa_no_gate_no_bank_f${FOLD}_s${S}.log"
    echo "[$(date +%H:%M:%S)] swa_no_gate_no_bank fold=$FOLD seed=$S"
    PYTHONPATH=$PWD python3 -u -m src.invar.training.train \
      --fold "$FOLD" --seed "$S" --epochs 30 \
      --output-dir "$OUT_DIR" \
      --regime-axis retrieval --loss-config ranking --bank-size 64 \
      --disable-bank \
      --use-swa --swa-decay 0.995 --swa-warmup-epochs 0 \
      --save-predictions \
      > "$LF" 2>&1
  fi
done

echo "DONE $(date -Is)"
