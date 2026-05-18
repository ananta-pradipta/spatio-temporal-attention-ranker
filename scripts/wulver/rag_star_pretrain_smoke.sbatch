#!/bin/bash -l
#SBATCH --job-name=model-eval
#SBATCH --partition=gpu
#SBATCH --account=hz79
#SBATCH --qos=standard
#SBATCH --time=00:50:00
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
mkdir -p logs/_pre_smoke results/_pre_smoke results/_pre_smoke_adv configs/_ablation_tmp
echo "host=$(hostname) start=$(date -Is)"

NO_IPO_CFG=configs/_ablation_tmp/rag_star_universe_v2_no_ipo_headline.yaml
python3 -c "
import yaml
with open('configs/rag_star_universe_v2.yaml') as f: cfg=yaml.safe_load(f)
cfg.setdefault('train',{}); cfg['train']['disable_ipo_retrieval']=True
open('$NO_IPO_CFG','w').write(yaml.safe_dump(cfg))
"

echo '=== (1) REGRESSION: plain NoIPO finetune still works post-hook ==='
PYTHONPATH=$PWD python3 -u -m src.v2.training.train_dow_epistar \
  --config "$NO_IPO_CFG" --fold 1 --seed 42 --two_regime_val --smoke \
  --output_dir results/_pre_smoke 2>&1 | tail -5

echo '=== (2) ARM1 Stage-1 contrastive pretrain (--pretrain_only --smoke) ==='
PYTHONPATH=$PWD python3 -u -m src.v2.training.train_dow_epistar \
  --config "$NO_IPO_CFG" --fold 1 --seed 42 --two_regime_val --smoke \
  --pretrain_objective contrastive --pretrain_only \
  --output_dir results/_pre_smoke 2>&1 | tail -8

echo '=== (3) ARM3 Stage-1 adversarial pretrain (--pretrain_only --smoke) ==='
PYTHONPATH=$PWD python3 -u -m src.v2.training.train_dow_epistar \
  --config "$NO_IPO_CFG" --fold 1 --seed 42 --two_regime_val --smoke \
  --pretrain_objective adversarial --pretrain_only \
  --output_dir results/_pre_smoke_adv 2>&1 | tail -8

echo '=== (4) Stage-2: load contrastive backbone ckpt + finetune (--smoke) ==='
PYTHONPATH=$PWD python3 -u -m src.v2.training.train_dow_epistar \
  --config "$NO_IPO_CFG" --fold 1 --seed 42 --two_regime_val --smoke \
  --init_backbone_ckpt results/_pre_smoke/_ckpt/fold1_backbone.pt \
  --output_dir results/_pre_smoke 2>&1 | tail -6

echo "DONE $(date -Is)"
ls -l results/_pre_smoke/_ckpt/ results/_pre_smoke_adv/_ckpt/ 2>/dev/null || true
