#!/bin/bash -l
#SBATCH --job-name=model-eval
#SBATCH --partition=gpu
#SBATCH --account=hz79
#SBATCH --qos=standard
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --mem=32G
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

# Ablation B: force all 5 StockTwits cols to zero in the v1 (22-col) adapter.
# Output to results/rag_star_universe_ablation_b/ to keep the headline v1
# numbers intact.
export LATTICE_PANEL_ZERO_ST=1

REPO="${REPO:-$HOME/phd-research}"
VENV="${VENV:-$HOME/phd-research-gpu-env}"
cd "$REPO"
module load bright 2>/dev/null || true
module load python3 2>/dev/null || true
source "$VENV/bin/activate"

mkdir -p logs/rag_star_universe_ablation_b results/rag_star_universe_ablation_b

echo "host=$(hostname) start=$(date -Is)"
echo "LATTICE_PANEL_ZERO_ST=$LATTICE_PANEL_ZERO_ST"
nvidia-smi | head -8 || true

# Hijack the output_dir via env var to avoid editing the config.
# train_dow_epistar.py reads cfg.train.output_dir at runtime, so the
# cleanest path is a small per-fold loop with results_dir override.
# We'll instead use the v1 config but redirect output_dir to the ablation dir
# by overriding through a temp config.
TMP_CFG=$(mktemp /tmp/rag_star_universe_ablation_b.XXXXXX.yaml)
trap "rm -f $TMP_CFG" EXIT
sed 's|results/rag_star_universe|results/rag_star_universe_ablation_b|' \
  configs/rag_star_universe.yaml > "$TMP_CFG"
grep "output_dir" "$TMP_CFG" || true

for FOLD in 1 2 3; do
  for S in 42 43 44 45 46; do
    OUT="results/rag_star_universe_ablation_b/fold${FOLD}_seed${S}.json"
    if [ -f "$OUT" ] && [ "$(grep -c '"epoch"' "$OUT")" -ge 5 ]; then
      echo "[$(date +%H:%M:%S)] skip $OUT"
      continue
    fi
    LF="logs/rag_star_universe_ablation_b/fold${FOLD}_seed${S}.log"
    echo "[$(date +%H:%M:%S)] ablation-b fold=$FOLD seed=$S"
    PYTHONPATH=$PWD python3 -u -m src.v2.training.train_dow_epistar \
      --config "$TMP_CFG" --fold "$FOLD" --seed "$S" \
      > "$LF" 2>&1
  done
done

echo "DONE $(date -Is)"
