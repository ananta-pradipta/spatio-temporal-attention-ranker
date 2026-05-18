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
mkdir -p logs/invar experiments/invar/ablation_no_gate

echo "host=$(hostname) start=$(date -Is)"
nvidia-smi | head -10 || true

# Tier A3: InVAR with MarketGateV2 OFF; bank ON (hard_topk); SWA OFF.
# 5 seeds * 2 folds = 10 runs. All other flags match invar_v6_f1_f2.sbatch
# exactly except --use-market-gate-v2 is removed.
for FOLD in 1 2; do
  for S in 42 43 44 45 46; do
    OUT_DIR="experiments/invar/ablation_no_gate/fold${FOLD}"
    OUT="${OUT_DIR}/seed${S}_designR/results.json"
    if [ -f "$OUT" ]; then
      echo "[$(date +%H:%M:%S)] skip $OUT"
      continue
    fi
    LF="logs/invar/ablation_no_gate_f${FOLD}_s${S}.log"
    echo "[$(date +%H:%M:%S)] no-gate fold=$FOLD seed=$S"
    PYTHONPATH=$PWD python3 -u -m src.invar.training.train \
      --fold "$FOLD" --seed "$S" --epochs 30 \
      --output-dir "$OUT_DIR" \
      --regime-axis retrieval --loss-config ranking --bank-size 64 \
      --use-dynamic-bank-controller \
      --bank-controller-mode hybrid \
      --bank-controller-min-weight 0.05 \
      --bank-controller-max-weight 1.00 \
      --save-predictions \
      > "$LF" 2>&1
  done
done

echo "DONE $(date -Is)"
