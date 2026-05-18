#!/bin/bash -l
#SBATCH --job-name=model-eval-noipo-abl
#SBATCH --partition=gpu
#SBATCH --account=hz79
#SBATCH --qos=standard
#SBATCH --time=24:00:00
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

mkdir -p logs/rag_star_universe_v2_no_ipo_ablations \
         results/rag_star_universe_v2_no_ipo_ablations \
         configs/_ablation_tmp

echo "host=$(hostname) start=$(date -Is)"
nvidia-smi | head -8 || true

# Component ablations measured ON TOP OF the locked NoIPO headline
# (results/rag_star_universe_v2_no_ipo, 5-fold pooled IC +0.0213), so
# every row is a true delta from the +0.0213 headline rather than from
# the superseded with-IPO base used by results/rag_star_universe_v2_ablations.
#
# Base for every run: configs/rag_star_universe_v2.yaml
#   + train.disable_ipo_retrieval = true   (NoIPO headline)
#   + --two_regime_val                     (locked validation protocol)
# plus exactly ONE additional toggle per variant:
#   C1 minus rate-sensitivity memory : train.use_rate_memory       = false
#   C2 minus rate-sensitivity graph  : train.use_duration_graph    = false
#   C3 minus day-level regime memory : train.disable_day_retrieval = true
#
# (full arch. = results/rag_star_universe_v2_no_ipo = +0.0213, already have;
#  minus macro head = results/rag_star_universe_v2_no_ipo_no_macro = +0.0196,
#  already have. Only these 3 rows are missing on the NoIPO base.)
#
# 3 ablations x 5 folds x 5 seeds = 75 runs. Single sbatch, internal loop,
# skip-if-done. CLI flags identical to the canonical no_ipo sbatch except
# the one extra ablation key per variant.

TMPL=configs/rag_star_universe_v2.yaml
C1_CFG=configs/_ablation_tmp/noipo_abl_c1_no_rate_mem.yaml
C2_CFG=configs/_ablation_tmp/noipo_abl_c2_no_rate_graph.yaml
C3_CFG=configs/_ablation_tmp/noipo_abl_c3_no_day_mem.yaml

python3 -c "
import yaml
with open('$TMPL') as f: cfg = yaml.safe_load(f)
cfg.setdefault('train', {})
cfg['train']['disable_ipo_retrieval'] = True
cfg['train']['use_rate_memory'] = False
with open('$C1_CFG', 'w') as f: yaml.safe_dump(cfg, f)
"
python3 -c "
import yaml
with open('$TMPL') as f: cfg = yaml.safe_load(f)
cfg.setdefault('train', {})
cfg['train']['disable_ipo_retrieval'] = True
cfg['train']['use_duration_graph'] = False
with open('$C2_CFG', 'w') as f: yaml.safe_dump(cfg, f)
"
python3 -c "
import yaml
with open('$TMPL') as f: cfg = yaml.safe_load(f)
cfg.setdefault('train', {})
cfg['train']['disable_ipo_retrieval'] = True
cfg['train']['disable_day_retrieval'] = True
with open('$C3_CFG', 'w') as f: yaml.safe_dump(cfg, f)
"

run_one() {
  local NAME="$1"; local CFG="$2"; local FOLD="$3"; local S="$4"
  local OUT_DIR="results/rag_star_universe_v2_no_ipo_ablations/${NAME}"
  mkdir -p "$OUT_DIR"
  local OUT="${OUT_DIR}/fold${FOLD}_seed${S}.json"
  if [ -f "$OUT" ] && [ "$(grep -c '"epoch"' "$OUT")" -ge 1 ]; then
    echo "[$(date +%H:%M:%S)] skip $OUT"
    return
  fi
  local LF="logs/rag_star_universe_v2_no_ipo_ablations/${NAME}_fold${FOLD}_seed${S}.log"
  echo "[$(date +%H:%M:%S)] ${NAME} fold=$FOLD seed=$S"
  PYTHONPATH=$PWD python3 -u -m src.v2.training.train_dow_epistar \
    --config "$CFG" --fold "$FOLD" --seed "$S" \
    --two_regime_val \
    --output_dir "$OUT_DIR" \
    > "$LF" 2>&1
}

for FOLD in 1 2 3 4 5; do
  for S in 42 43 44 45 46; do
    run_one C1_no_rate_sensitivity_memory "$C1_CFG" "$FOLD" "$S"
    run_one C2_no_rate_sensitivity_graph  "$C2_CFG" "$FOLD" "$S"
    run_one C3_no_day_level_regime_memory "$C3_CFG" "$FOLD" "$S"
  done
done

echo "DONE $(date -Is)"
