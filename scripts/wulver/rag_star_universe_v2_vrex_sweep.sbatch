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
#SBATCH --mem=48G
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

REPO="${REPO:-$HOME/phd-research}"
VENV="${VENV:-$HOME/phd-research-gpu-env}"
cd "$REPO"
source "$VENV/bin/activate"

mkdir -p logs/rag_star_universe_v2_vrex \
         results/rag_star_universe_v2_vrex \
         configs/_ablation_tmp

echo "host=$(hostname) start=$(date -Is)"
nvidia-smi | head -8 || true

# Ablation arm 2 (pre-registered 2026-05-17): V-REx across macro-regime
# environments on the RAG-STAR NoIPO headline. IDENTICAL to
# rag_star_universe_v2_no_ipo_sweep.sbatch (same config = rag_star_universe_v2
# + disable_ipo_retrieval, same two_regime_val, same panel/fold protocol);
# the ONLY added variable is --vrex_beta (variance-of-environment-risk
# penalty). Separate OUT_DIR so the +0.0213 headline is never overwritten.
# beta=1.0 = canonical V-REx (Krueger et al. 2021); n_envs=4.

NO_IPO_CFG=configs/_ablation_tmp/rag_star_universe_v2_no_ipo_headline.yaml
python3 -c "
import yaml
with open('configs/rag_star_universe_v2.yaml') as f: cfg = yaml.safe_load(f)
cfg.setdefault('train', {})
cfg['train']['disable_ipo_retrieval'] = True
with open('$NO_IPO_CFG', 'w') as f: yaml.safe_dump(cfg, f)
"

VREX_BETA="${VREX_BETA:-1.0}"
VREX_N_ENVS="${VREX_N_ENVS:-4}"

for FOLD in 1 2 3 4 5; do
  for S in 42 43 44 45 46; do
    OUT="results/rag_star_universe_v2_vrex/fold${FOLD}_seed${S}.json"
    if [ -f "$OUT" ] && [ "$(grep -c '"epoch"' "$OUT")" -ge 1 ]; then
      echo "[$(date +%H:%M:%S)] skip $OUT"
      continue
    fi
    LF="logs/rag_star_universe_v2_vrex/fold${FOLD}_seed${S}.log"
    echo "[$(date +%H:%M:%S)] vrex fold=$FOLD seed=$S beta=$VREX_BETA"
    PYTHONPATH=$PWD python3 -u -m src.v2.training.train_dow_epistar \
      --config "$NO_IPO_CFG" --fold "$FOLD" --seed "$S" \
      --two_regime_val \
      --vrex_beta "$VREX_BETA" --vrex_n_envs "$VREX_N_ENVS" \
      --output_dir "results/rag_star_universe_v2_vrex" \
      > "$LF" 2>&1
  done
done

echo "DONE $(date -Is)"
