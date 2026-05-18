#!/bin/bash -l
#SBATCH --job-name=model-eval
#SBATCH --partition=gpu
#SBATCH --account=hz79
#SBATCH --qos=standard
#SBATCH --time=00:40:00
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
mkdir -p logs/_vrex_smoke results/_vrex_smoke configs/_ablation_tmp
echo "host=$(hostname) start=$(date -Is)"

NO_IPO_CFG=configs/_ablation_tmp/rag_star_universe_v2_no_ipo_headline.yaml
python3 -c "
import yaml
with open('configs/rag_star_universe_v2.yaml') as f: cfg = yaml.safe_load(f)
cfg.setdefault('train', {})
cfg['train']['disable_ipo_retrieval'] = True
with open('$NO_IPO_CFG', 'w') as f: yaml.safe_dump(cfg, f)
"

# Smoke: fold 1 seed 42, --smoke caps to 80 train steps. Validates the
# V-REx batched loop end-to-end before the full 25-cell launch.
PYTHONPATH=$PWD python3 -u -m src.v2.training.train_dow_epistar \
  --config "$NO_IPO_CFG" --fold 1 --seed 42 \
  --two_regime_val --smoke \
  --vrex_beta 1.0 --vrex_n_envs 4 \
  --output_dir "results/_vrex_smoke" \
  2>&1 | tail -40

echo "DONE $(date -Is)"
ls -l results/_vrex_smoke/ || true
