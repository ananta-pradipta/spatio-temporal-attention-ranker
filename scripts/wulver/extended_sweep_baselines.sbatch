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
mkdir -p logs/invar_2025 experiments/invar_2025/baselines

echo "host=$(hostname) start=$(date -Is)"
nvidia-smi | head -10 || true

# Extended-panel (2015-2025/26) sweep: iT, MASTER, FactorVAE, StockMixer
# x F3, F4, F5 x 5 seeds = 60 runs. Output to experiments/invar_2025/.
# The 2015-2022 panel results in experiments/invar/baselines/ are preserved.
for BASELINE in itransformer master factorvae stockmixer; do
  for FOLD in 3 4 5; do
    for S in 42 43 44 45 46; do
      OUT="experiments/invar_2025/baselines/${BASELINE}/fold${FOLD}/seed${S}/results.json"
      if [ -f "$OUT" ]; then
        echo "[$(date +%H:%M:%S)] skip $OUT"
        continue
      fi
      LF="logs/invar_2025/${BASELINE}_fold${FOLD}_seed${S}.log"
      echo "[$(date +%H:%M:%S)] baseline=$BASELINE fold=$FOLD seed=$S"
      PYTHONPATH=$PWD python3 -u -m src.invar.baselines.train_baseline \
        --baseline "$BASELINE" \
        --fold "$FOLD" --seed "$S" --epochs 10 \
        --output-dir experiments/invar_2025/baselines \
        --save-predictions \
        > "$LF" 2>&1
    done
  done
done

echo "DONE $(date -Is)"
