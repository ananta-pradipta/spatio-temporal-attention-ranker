#!/bin/bash -l
#SBATCH --job-name=model-eval-noipo
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

mkdir -p logs/rag_star_universe_v2_no_ipo \
         results/rag_star_universe_v2_no_ipo \
         configs/_ablation_tmp

echo "host=$(hostname) start=$(date -Is)"
nvidia-smi | head -8 || true

# RAG-STAR-NoIPO: headline-class model variant with the IPO analogue memory
# component removed. Same panel, fold protocol (two-regime val), idxfix, VIX
# fix as the headline RAG-STAR. The only change vs HEADLINE is
# disable_ipo_retrieval=True.
#
# Motivated by the A3 ablation finding that IPO analogue memory is hurting
# net IC by +0.003 on F1-F3 partial. Run as a candidate "simplified" RAG-STAR
# variant for the headline comparison table.
#
# Independent of the running ablation A3 (separate output dir), so the
# A3 ablation table stays clean and this serves as a headline-class entry.

NO_IPO_CFG=configs/_ablation_tmp/rag_star_universe_v2_no_ipo_headline.yaml
python3 -c "
import yaml
with open('configs/rag_star_universe_v2.yaml') as f: cfg = yaml.safe_load(f)
cfg.setdefault('train', {})
cfg['train']['disable_ipo_retrieval'] = True
with open('$NO_IPO_CFG', 'w') as f: yaml.safe_dump(cfg, f)
"

for FOLD in 1 2 3 4 5; do
  for S in 42 43 44 45 46; do
    OUT="results/rag_star_universe_v2_no_ipo/fold${FOLD}_seed${S}.json"
    if [ -f "$OUT" ] && [ "$(grep -c '"epoch"' "$OUT")" -ge 1 ]; then
      echo "[$(date +%H:%M:%S)] skip $OUT"
      continue
    fi
    LF="logs/rag_star_universe_v2_no_ipo/fold${FOLD}_seed${S}.log"
    echo "[$(date +%H:%M:%S)] no-ipo fold=$FOLD seed=$S"
    PYTHONPATH=$PWD python3 -u -m src.v2.training.train_dow_epistar \
      --config "$NO_IPO_CFG" --fold "$FOLD" --seed "$S" \
      --two_regime_val \
      --output_dir "results/rag_star_universe_v2_no_ipo" \
      > "$LF" 2>&1
  done
done

echo "DONE $(date -Is)"
