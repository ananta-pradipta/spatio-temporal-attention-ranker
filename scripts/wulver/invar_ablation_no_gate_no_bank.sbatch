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
mkdir -p logs/invar experiments/invar/ablation_no_gate_no_bank

echo "host=$(hostname) start=$(date -Is)"
nvidia-smi | head -10 || true

# Tier A4: InVAR with both MarketGateV2 AND RegimeAxisRetrieval OFF.
# This is the "bare backbone" cell: 4 InvarBlocks doing variate self-attn
# only (no regime cross-attn path). Effectively iTransformer over variate
# tokens with the InVAR loss.
#
# 5 seeds * 2 folds = 10 runs. All other flags match invar_v6_f1_f2.sbatch
# minus --use-market-gate-v2 and --use-dynamic-bank-controller, plus
# --disable-bank.
for FOLD in 1 2; do
  for S in 42 43 44 45 46; do
    OUT_DIR="experiments/invar/ablation_no_gate_no_bank/fold${FOLD}"
    OUT="${OUT_DIR}/seed${S}_designR/results.json"
    if [ -f "$OUT" ]; then
      echo "[$(date +%H:%M:%S)] skip $OUT"
      continue
    fi
    LF="logs/invar/ablation_no_gate_no_bank_f${FOLD}_s${S}.log"
    echo "[$(date +%H:%M:%S)] no-gate-no-bank fold=$FOLD seed=$S"
    PYTHONPATH=$PWD python3 -u -m src.invar.training.train \
      --fold "$FOLD" --seed "$S" --epochs 30 \
      --output-dir "$OUT_DIR" \
      --regime-axis retrieval --loss-config ranking --bank-size 64 \
      --disable-bank \
      --save-predictions \
      > "$LF" 2>&1
  done
done

echo "DONE $(date -Is)"
