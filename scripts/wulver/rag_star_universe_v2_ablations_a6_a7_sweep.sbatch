#!/bin/bash -l
#SBATCH --job-name=model-eval-ablations2
#SBATCH --partition=gpu
#SBATCH --account=hz79
#SBATCH --qos=standard
#SBATCH --time=18:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --mem=48G
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

REPO="${REPO:-$HOME/phd-research}"
VENV="${VENV:-$HOME/phd-research-gpu-env}"
cd "$REPO"
source "$VENV/bin/activate"

mkdir -p logs/rag_star_universe_v2_ablations \
         results/rag_star_universe_v2_ablations \
         configs/_ablation_tmp

echo "host=$(hostname) start=$(date -Is)"
nvidia-smi | head -8 || true

# Phase 2 universal-panel ablations: tests whether the components NOT covered
# by A1-A5 (day-level regime memory, macro rate-sensitivity head) are
# load-bearing. These are the components left in the architecture if A1-A3
# were removed; the question is whether they account for RAG-STAR's
# competitive performance vs SOTA baselines on universal panel.
#
# A6: --disable_day_retrieval        no day-level regime memory
# A7: --disable_macro_duration       no macro rate-sensitivity head
#
# 2 ablations x 5 folds x 5 seeds = 50 runs. Walltime cap 18h.

TMPL=configs/rag_star_universe_v2.yaml
NO_DAY_CFG=configs/_ablation_tmp/rag_star_universe_v2_no_day_memory.yaml
NO_MACRO_HEAD_CFG=configs/_ablation_tmp/rag_star_universe_v2_no_macro_head.yaml

python3 -c "
import yaml
with open('$TMPL') as f: cfg = yaml.safe_load(f)
cfg.setdefault('train', {})
cfg['train']['disable_day_retrieval'] = True
with open('$NO_DAY_CFG', 'w') as f: yaml.safe_dump(cfg, f)
"

python3 -c "
import yaml
with open('$TMPL') as f: cfg = yaml.safe_load(f)
cfg.setdefault('train', {})
cfg['train']['disable_macro_duration'] = True
with open('$NO_MACRO_HEAD_CFG', 'w') as f: yaml.safe_dump(cfg, f)
"

run_one() {
  local NAME="$1"
  local CFG="$2"
  local FOLD="$3"; local SEED="$4"
  local OUT_DIR="results/rag_star_universe_v2_ablations/${NAME}"
  mkdir -p "$OUT_DIR"
  local OUT="${OUT_DIR}/fold${FOLD}_seed${SEED}.json"
  if [ -f "$OUT" ] && [ "$(grep -c '"epoch"' "$OUT")" -ge 1 ]; then
    echo "[$(date +%H:%M:%S)] skip $OUT"
    return
  fi
  local LF="logs/rag_star_universe_v2_ablations/${NAME}_fold${FOLD}_seed${SEED}.log"
  echo "[$(date +%H:%M:%S)] ablation=${NAME} fold=$FOLD seed=$SEED"
  PYTHONPATH=$PWD python3 -u -m src.v2.training.train_dow_epistar \
    --config "$CFG" --fold "$FOLD" --seed "$SEED" \
    --two_regime_val --output_dir "$OUT_DIR" \
    > "$LF" 2>&1
}

for FOLD in 1 2 3 4 5; do
  for S in 42 43 44 45 46; do
    run_one A6_no_day_level_regime_memory "$NO_DAY_CFG" "$FOLD" "$S"
    run_one A7_no_macro_rate_sensitivity_head "$NO_MACRO_HEAD_CFG" "$FOLD" "$S"
  done
done

echo "DONE $(date -Is)"
