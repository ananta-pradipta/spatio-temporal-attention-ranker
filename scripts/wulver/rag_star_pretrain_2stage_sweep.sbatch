#!/bin/bash -l
#SBATCH --job-name=model-eval
#SBATCH --partition=gpu
#SBATCH --account=hz79
#SBATCH --qos=standard
#SBATCH --time=14:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100_40g:1
#SBATCH --mem=48G
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REPO="${REPO:-$HOME/phd-research}"
VENV="${VENV:-$HOME/phd-research-gpu-env}"
cd "$REPO"
source "$VENV/bin/activate"

# Ablation arms 1/3 (pre-registered 2026-05-17): two-stage pretrain ->
# finetune on the RAG-STAR NoIPO headline. IDENTICAL to
# rag_star_universe_v2_no_ipo_sweep.sbatch (same config, two_regime_val,
# panel/fold protocol); the ONLY change is a Stage-1 STARBackbone
# pretrain whose ckpt warm-starts Stage-2 via --init_backbone_ckpt.
# OBJ selects the arm: contrastive (arm 1) | adversarial (arm 3).
# Separate OUT_DIR; the +0.0213 headline is never overwritten.
OBJ="${OBJ:?set OBJ=contrastive|adversarial via --export}"
OUT="${OUT:?set OUT=results/... via --export}"

mkdir -p "logs/$(basename "$OUT")" "$OUT" "$OUT/_ckpt" configs/_ablation_tmp
echo "host=$(hostname) start=$(date -Is) OBJ=$OBJ OUT=$OUT"
nvidia-smi | head -8 || true

NO_IPO_CFG=configs/_ablation_tmp/rag_star_universe_v2_no_ipo_headline.yaml
python3 -c "
import yaml
with open('configs/rag_star_universe_v2.yaml') as f: cfg=yaml.safe_load(f)
cfg.setdefault('train',{}); cfg['train']['disable_ipo_retrieval']=True
open('$NO_IPO_CFG','w').write(yaml.safe_dump(cfg))
"

for FOLD in 1 2 3 4 5; do
  CKPT="$OUT/_ckpt/fold${FOLD}_backbone.pt"
  # ---- Stage 1: pretrain backbone once per fold (seed 42). ----
  if [ -f "$CKPT" ]; then
    echo "[$(date +%H:%M:%S)] skip stage-1 (ckpt exists: $CKPT)"
  else
    PLF="logs/$(basename "$OUT")/fold${FOLD}_pretrain.log"
    echo "[$(date +%H:%M:%S)] stage-1 $OBJ pretrain fold=$FOLD"
    PYTHONPATH=$PWD python3 -u -m src.v2.training.train_dow_epistar \
      --config "$NO_IPO_CFG" --fold "$FOLD" --seed 42 --two_regime_val \
      --pretrain_objective "$OBJ" --pretrain_only \
      --output_dir "$OUT" > "$PLF" 2>&1
  fi
  # ---- Stage 2: finetune all 5 seeds off the fold backbone ckpt. ----
  for S in 42 43 44 45 46; do
    OJSON="$OUT/fold${FOLD}_seed${S}.json"
    if [ -f "$OJSON" ] && [ "$(grep -c '"epoch"' "$OJSON")" -ge 1 ]; then
      echo "[$(date +%H:%M:%S)] skip $OJSON"
      continue
    fi
    LF="logs/$(basename "$OUT")/fold${FOLD}_seed${S}.log"
    echo "[$(date +%H:%M:%S)] stage-2 finetune fold=$FOLD seed=$S"
    PYTHONPATH=$PWD python3 -u -m src.v2.training.train_dow_epistar \
      --config "$NO_IPO_CFG" --fold "$FOLD" --seed "$S" --two_regime_val \
      --init_backbone_ckpt "$CKPT" \
      --output_dir "$OUT" > "$LF" 2>&1
  done
done

echo "DONE $(date -Is)"
