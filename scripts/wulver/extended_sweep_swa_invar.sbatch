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
cd $HOME/phd-research
module load bright 2>/dev/null || true
module load python3 2>/dev/null || true
source $HOME/phd-research-gpu-env/bin/activate
mkdir -p logs/invar_2025 experiments/invar_2025/swa

echo "host=$(hostname) start=$(date -Is)"
nvidia-smi | head -10 || true

# Extended-panel SWA-InVAR sweep: F3, F4, F5 x 5 seeds = 15 runs at ~25-30
# min per seed. Flags identical to scripts/wulver/invar_swa.sbatch which
# produced the 2015-2022 headline. Output to experiments/invar_2025/.
for FOLD in 3 4 5; do
  for S in 42 43 44 45 46; do
    OUT_DIR="experiments/invar_2025/swa/fold${FOLD}"
    OUT="${OUT_DIR}/seed${S}_designR/results.json"
    if [ -f "$OUT" ]; then
      echo "[$(date +%H:%M:%S)] skip $OUT"
      continue
    fi
    LF="logs/invar_2025/swa_f${FOLD}_s${S}.log"
    echo "[$(date +%H:%M:%S)] swa-invar fold=$FOLD seed=$S"
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
