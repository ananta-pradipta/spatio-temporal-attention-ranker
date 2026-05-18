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
mkdir -p logs/invar logs/baselines experiments/invar/swa experiments/invar/baselines

echo "host=$(hostname) start=$(date -Is)"
nvidia-smi | head -10 || true

# Fold 3 evaluation: InVAR-SWA active head + 4 external baselines.
# 5 models * 5 seeds = 25 runs. Per-run wall ~5-12 min on A100;
# total ~3-5 hours within the 8h walltime budget.
#
# Fold 3 train ends 2022-06-30; test is the late-2022 / early-2023 window.

# --- InVAR-SWA active head (gumbel_topk + SWA + market gate v2 + bank
# controller). All flags identical to invar_swa.sbatch.
for S in 42 43 44 45 46; do
  OUT_DIR="experiments/invar/swa/fold3"
  OUT="${OUT_DIR}/seed${S}_designR/results.json"
  if [ -f "$OUT" ]; then
    echo "[$(date +%H:%M:%S)] skip $OUT"
    continue
  fi
  LF="logs/invar/swa_f3_s${S}.log"
  echo "[$(date +%H:%M:%S)] swa-invar fold=3 seed=$S"
  PYTHONPATH=$PWD python3 -u -m src.invar.training.train \
    --fold 3 --seed "$S" --epochs 30 \
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

# --- External baselines: MASTER, iTransformer, StockMixer (Tier 1 trio).
for BASELINE in itransformer stockmixer master; do
  for S in 42 43 44 45 46; do
    OUT="experiments/invar/baselines/${BASELINE}/fold3/seed${S}/results.json"
    if [ -f "$OUT" ]; then
      echo "[$(date +%H:%M:%S)] skip $OUT"
      continue
    fi
    LF="logs/baselines/${BASELINE}_f3_s${S}.log"
    echo "[$(date +%H:%M:%S)] baseline=$BASELINE fold=3 seed=$S"
    PYTHONPATH=$PWD python3 -u -m src.invar.baselines.train_baseline \
      --baseline "$BASELINE" \
      --fold 3 --seed "$S" --epochs 10 \
      --output-dir experiments/invar/baselines \
      --save-predictions \
      > "$LF" 2>&1
  done
done

# --- FactorVAE (separate sbatch in F1+F2 history; included here for F3).
for S in 42 43 44 45 46; do
  OUT="experiments/invar/baselines/factorvae/fold3/seed${S}/results.json"
  if [ -f "$OUT" ]; then
    echo "[$(date +%H:%M:%S)] skip $OUT"
    continue
  fi
  LF="logs/baselines/factorvae_f3_s${S}.log"
  echo "[$(date +%H:%M:%S)] baseline=factorvae fold=3 seed=$S"
  PYTHONPATH=$PWD python3 -u -m src.invar.baselines.train_baseline \
    --baseline factorvae \
    --fold 3 --seed "$S" --epochs 30 \
    --output-dir experiments/invar/baselines \
    --save-predictions \
    > "$LF" 2>&1
done

echo "DONE $(date -Is)"
