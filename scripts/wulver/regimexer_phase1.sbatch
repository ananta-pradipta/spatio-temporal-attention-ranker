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
cd $HOME/phd-research
module load bright 2>/dev/null || true
module load python3 2>/dev/null || true
source $HOME/phd-research-gpu-env/bin/activate
mkdir -p logs/regimexer experiments/invar/baselines

echo "host=$(hostname) start=$(date -Is)"
nvidia-smi | head -10 || true

# RegimeXer-iT Phase 1: A1 (macro_tokens_only) and A2 (film) on all
# 3 folds x 5 seeds = 30 runs. Single sbatch with internal loops per repo
# convention. Each run is up to 30 epochs with patience-5 early stop;
# expected ~5-8 min wall per run -> 30 runs in 2.5-4 hours. 8h walltime
# is safe. Each run skips if its results.json already exists, so a
# re-launch after wall-time-out resumes.
for MODE in macro_tokens_only film; do
  for FOLD in 1 2 3; do
    for S in 42 43 44 45 46; do
      OUT="experiments/invar/baselines/regimexer/${MODE}/fold${FOLD}/seed${S}/results.json"
      if [ -f "$OUT" ]; then
        echo "[$(date +%H:%M:%S)] skip $OUT"
        continue
      fi
      LF="logs/regimexer/${MODE}_fold${FOLD}_seed${S}.log"
      echo "[$(date +%H:%M:%S)] regimexer mode=$MODE fold=$FOLD seed=$S"
      PYTHONPATH=$PWD python3 -u -m src.invar.baselines.train_baseline \
        --baseline regimexer --regimexer-mode "$MODE" \
        --fold "$FOLD" --seed "$S" --epochs 30 \
        --output-dir experiments/invar/baselines \
        --save-predictions \
        > "$LF" 2>&1
    done
  done
done

echo "DONE $(date -Is)"
