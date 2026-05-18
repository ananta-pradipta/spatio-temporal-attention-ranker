#!/bin/bash -l
#SBATCH --job-name=model-eval-ablations
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

mkdir -p logs/rag_star_universe_v2_ablations \
         results/rag_star_universe_v2_ablations

echo "host=$(hostname) start=$(date -Is)"
nvidia-smi | head -8 || true

# Universal RAG-STAR component ablations matching the biotech paper's
# Section 6 ablation structure. Same panel, fold protocol, idxfix, VIX
# fix as the headline at results/rag_star_universe_v2_two_regime_val/.
#
# Naming follows the biotech paper convention:
#   - "rate-sensitivity memory"  (code: rate memory)
#   - "rate-sensitivity graph"   (code: duration graph)
#   - "IPO analogue memory"      (code: IPO retrieval)
#   - "day-level regime memory"  (code: day retrieval)
#   - "macro state"              (code: macro state)
#
# 5 ablations x 5 folds x 5 seeds = 125 runs. Walltime cap 24h.

run_ablation() {
  local NAME="$1"
  local FOLD="$2"; local SEED="$3"
  shift 3
  local OUT_DIR="results/rag_star_universe_v2_ablations/${NAME}"
  mkdir -p "$OUT_DIR"
  local OUT="${OUT_DIR}/fold${FOLD}_seed${SEED}.json"
  if [ -f "$OUT" ] && [ "$(grep -c '"epoch"' "$OUT")" -ge 1 ]; then
    echo "[$(date +%H:%M:%S)] skip $OUT"
    return
  fi
  local LF="logs/rag_star_universe_v2_ablations/${NAME}_fold${FOLD}_seed${SEED}.log"
  echo "[$(date +%H:%M:%S)] ablation=${NAME} fold=$FOLD seed=$SEED $@"
  PYTHONPATH=$PWD python3 -u -m src.v2.training.train_dow_epistar \
    --config configs/rag_star_universe_v2.yaml \
    --fold "$FOLD" --seed "$SEED" \
    --two_regime_val \
    --output_dir "$OUT_DIR" \
    "$@" \
    > "$LF" 2>&1
}

# A1. No rate-sensitivity memory  (--use_rate_memory false equivalent)
# A2. No rate-sensitivity graph   (--use_duration_graph false)
# A3. No IPO analogue memory      (--disable_ipo_retrieval true)
# A4. Random rate-sensitivity retrieval (sanity: random keys < real keys?)
# A5. Shuffled macro state         (sanity: real macro > shuffled?)
#
# Note: ablation flags pass via the trainer's CLI; some are TrainConfig
# fields, others are popped from train_kwargs (disable_day/ipo_retrieval).
# We override via a temp config that flips those flags.

# Helper: render a temp YAML with disable_ipo_retrieval=true and pass via
# --config. Other ablations work by overriding TrainConfig keys directly.

mkdir -p configs/_ablation_tmp
TMPL=configs/rag_star_universe_v2.yaml
NO_IPO_CFG=configs/_ablation_tmp/rag_star_universe_v2_no_ipo.yaml
python3 -c "
import yaml
with open('$TMPL') as f: cfg = yaml.safe_load(f)
cfg.setdefault('train', {})
cfg['train']['disable_ipo_retrieval'] = True
with open('$NO_IPO_CFG', 'w') as f: yaml.safe_dump(cfg, f)
"

# Drive the 5 ablations through CLI flags / config overrides.
for FOLD in 1 2 3 4 5; do
  for S in 42 43 44 45 46; do
    # A1: turn off rate-sensitivity memory.
    # use_rate_memory is a TrainConfig field (default True); to disable,
    # pass via a one-off config override.
    NO_RATE_MEM_CFG=configs/_ablation_tmp/rag_star_universe_v2_no_rate_mem.yaml
    python3 -c "
import yaml
with open('$TMPL') as f: cfg = yaml.safe_load(f)
cfg.setdefault('train', {})
cfg['train']['use_rate_memory'] = False
with open('$NO_RATE_MEM_CFG', 'w') as f: yaml.safe_dump(cfg, f)
"
    OUT_DIR=results/rag_star_universe_v2_ablations/A1_no_rate_sensitivity_memory
    mkdir -p "$OUT_DIR"
    OUT="${OUT_DIR}/fold${FOLD}_seed${S}.json"
    if [ ! -f "$OUT" ] || [ "$(grep -c '"epoch"' "$OUT")" -lt 1 ]; then
      LF="logs/rag_star_universe_v2_ablations/A1_no_rate_sensitivity_memory_fold${FOLD}_seed${S}.log"
      echo "[$(date +%H:%M:%S)] A1 no-rate-sensitivity-memory fold=$FOLD seed=$S"
      PYTHONPATH=$PWD python3 -u -m src.v2.training.train_dow_epistar \
        --config "$NO_RATE_MEM_CFG" --fold "$FOLD" --seed "$S" \
        --two_regime_val --output_dir "$OUT_DIR" > "$LF" 2>&1
    fi

    # A2: turn off rate-sensitivity graph.
    NO_GRAPH_CFG=configs/_ablation_tmp/rag_star_universe_v2_no_rate_graph.yaml
    python3 -c "
import yaml
with open('$TMPL') as f: cfg = yaml.safe_load(f)
cfg.setdefault('train', {})
cfg['train']['use_duration_graph'] = False
with open('$NO_GRAPH_CFG', 'w') as f: yaml.safe_dump(cfg, f)
"
    OUT_DIR=results/rag_star_universe_v2_ablations/A2_no_rate_sensitivity_graph
    mkdir -p "$OUT_DIR"
    OUT="${OUT_DIR}/fold${FOLD}_seed${S}.json"
    if [ ! -f "$OUT" ] || [ "$(grep -c '"epoch"' "$OUT")" -lt 1 ]; then
      LF="logs/rag_star_universe_v2_ablations/A2_no_rate_sensitivity_graph_fold${FOLD}_seed${S}.log"
      echo "[$(date +%H:%M:%S)] A2 no-rate-sensitivity-graph fold=$FOLD seed=$S"
      PYTHONPATH=$PWD python3 -u -m src.v2.training.train_dow_epistar \
        --config "$NO_GRAPH_CFG" --fold "$FOLD" --seed "$S" \
        --two_regime_val --output_dir "$OUT_DIR" > "$LF" 2>&1
    fi

    # A3: turn off IPO analogue memory.
    OUT_DIR=results/rag_star_universe_v2_ablations/A3_no_ipo_analogue_memory
    mkdir -p "$OUT_DIR"
    OUT="${OUT_DIR}/fold${FOLD}_seed${S}.json"
    if [ ! -f "$OUT" ] || [ "$(grep -c '"epoch"' "$OUT")" -lt 1 ]; then
      LF="logs/rag_star_universe_v2_ablations/A3_no_ipo_analogue_memory_fold${FOLD}_seed${S}.log"
      echo "[$(date +%H:%M:%S)] A3 no-ipo-analogue-memory fold=$FOLD seed=$S"
      PYTHONPATH=$PWD python3 -u -m src.v2.training.train_dow_epistar \
        --config "$NO_IPO_CFG" --fold "$FOLD" --seed "$S" \
        --two_regime_val --output_dir "$OUT_DIR" > "$LF" 2>&1
    fi

    # A4: random rate-sensitivity retrieval (sanity).
    RANDOM_RATE_CFG=configs/_ablation_tmp/rag_star_universe_v2_random_rate.yaml
    python3 -c "
import yaml
with open('$TMPL') as f: cfg = yaml.safe_load(f)
cfg.setdefault('train', {})
cfg['train']['random_rate_retrieval'] = True
with open('$RANDOM_RATE_CFG', 'w') as f: yaml.safe_dump(cfg, f)
"
    OUT_DIR=results/rag_star_universe_v2_ablations/A4_random_rate_sensitivity_retrieval
    mkdir -p "$OUT_DIR"
    OUT="${OUT_DIR}/fold${FOLD}_seed${S}.json"
    if [ ! -f "$OUT" ] || [ "$(grep -c '"epoch"' "$OUT")" -lt 1 ]; then
      LF="logs/rag_star_universe_v2_ablations/A4_random_rate_sensitivity_retrieval_fold${FOLD}_seed${S}.log"
      echo "[$(date +%H:%M:%S)] A4 random-rate-sensitivity-retrieval fold=$FOLD seed=$S"
      PYTHONPATH=$PWD python3 -u -m src.v2.training.train_dow_epistar \
        --config "$RANDOM_RATE_CFG" --fold "$FOLD" --seed "$S" \
        --two_regime_val --output_dir "$OUT_DIR" > "$LF" 2>&1
    fi

    # A5: shuffle macro state (sanity).
    SHUFFLE_MACRO_CFG=configs/_ablation_tmp/rag_star_universe_v2_shuffle_macro.yaml
    python3 -c "
import yaml
with open('$TMPL') as f: cfg = yaml.safe_load(f)
cfg.setdefault('train', {})
cfg['train']['shuffle_macro_state'] = True
with open('$SHUFFLE_MACRO_CFG', 'w') as f: yaml.safe_dump(cfg, f)
"
    OUT_DIR=results/rag_star_universe_v2_ablations/A5_shuffled_macro_state
    mkdir -p "$OUT_DIR"
    OUT="${OUT_DIR}/fold${FOLD}_seed${S}.json"
    if [ ! -f "$OUT" ] || [ "$(grep -c '"epoch"' "$OUT")" -lt 1 ]; then
      LF="logs/rag_star_universe_v2_ablations/A5_shuffled_macro_state_fold${FOLD}_seed${S}.log"
      echo "[$(date +%H:%M:%S)] A5 shuffled-macro-state fold=$FOLD seed=$S"
      PYTHONPATH=$PWD python3 -u -m src.v2.training.train_dow_epistar \
        --config "$SHUFFLE_MACRO_CFG" --fold "$FOLD" --seed "$S" \
        --two_regime_val --output_dir "$OUT_DIR" > "$LF" 2>&1
    fi

  done
done

echo "DONE $(date -Is)"
