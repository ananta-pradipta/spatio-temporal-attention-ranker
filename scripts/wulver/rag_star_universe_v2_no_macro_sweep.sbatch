#!/bin/bash -l
#SBATCH --job-name=model-eval-nomacro
#SBATCH --partition=gpu
#SBATCH --account=hz79
#SBATCH --qos=standard
#SBATCH --time=12:00:00
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

mkdir -p logs/rag_star_universe_v2_no_ipo_no_macro \
         results/rag_star_universe_v2_no_ipo_no_macro \
         configs/_ablation_tmp

echo "host=$(hostname) start=$(date -Is)"
nvidia-smi | head -8 || true

# RAG-STAR-NoIPO without the macro rate-sensitivity head.
#
# Starts from the SAME base as the headline RAG-STAR-NoIPO
# (configs/rag_star_universe_v2.yaml + disable_ipo_retrieval=True,
# two-regime val, idxfix, VIX fix). The ONLY additional change vs the
# headline is disable_macro_duration=True, which sets
# use_macro_duration_head=False (train_dow_epistar.py:456), removing the
# MacroRateSensitivityHead from the forward path.
#
# Separate output dir so the active headline run
# (results/rag_star_universe_v2_no_ipo) is untouched; this is purely a
# component-removal probe for the ablation table.

CFG=configs/_ablation_tmp/rag_star_universe_v2_no_ipo_no_macro.yaml
python3 -c "
import yaml
with open('configs/rag_star_universe_v2.yaml') as f: cfg = yaml.safe_load(f)
cfg.setdefault('train', {})
cfg['train']['disable_ipo_retrieval'] = True
cfg['train']['disable_macro_duration'] = True
with open('$CFG', 'w') as f: yaml.safe_dump(cfg, f)
"

for FOLD in 1 2 3 4 5; do
  for S in 42 43 44 45 46; do
    OUT="results/rag_star_universe_v2_no_ipo_no_macro/fold${FOLD}_seed${S}.json"
    if [ -f "$OUT" ] && [ "$(grep -c '"epoch"' "$OUT")" -ge 1 ]; then
      echo "[$(date +%H:%M:%S)] skip $OUT"
      continue
    fi
    LF="logs/rag_star_universe_v2_no_ipo_no_macro/fold${FOLD}_seed${S}.log"
    echo "[$(date +%H:%M:%S)] no-ipo-no-macro fold=$FOLD seed=$S"
    PYTHONPATH=$PWD python3 -u -m src.v2.training.train_dow_epistar \
      --config "$CFG" --fold "$FOLD" --seed "$S" \
      --two_regime_val \
      --output_dir "results/rag_star_universe_v2_no_ipo_no_macro" \
      > "$LF" 2>&1
  done
done

echo "DONE $(date -Is)"
