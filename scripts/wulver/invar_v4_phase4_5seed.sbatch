#!/bin/bash -l
#SBATCH --job-name=invar-v4-p4-5seed
#SBATCH --partition=gpu
#SBATCH --account=hz79
#SBATCH --qos=standard
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --mem=32G
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
cd $HOME/phd-research
module load bright 2>/dev/null || true
module load python3 2>/dev/null || true
source $HOME/phd-research-gpu-env/bin/activate
mkdir -p logs/invar experiments/invar/v4_phase4

echo "host=$(hostname) start=$(date -Is)"
nvidia-smi | head -10 || true

# Phase 4: 4 cells x 2 folds x 5 seeds = 40 runs.
# Cell A is the v4 default and already has 5-seed F1+F2 from Phase 3; not re-run here.

run_cell() {
  local cell="$1"; shift
  local extra="$*"
  for fold in 1 2; do
    for seed in 42 43 44 45 46; do
      local out="experiments/invar/v4_phase4/cell${cell}/fold${fold}/seed${seed}"
      local lf="logs/invar/v4_p4_cell${cell}_f${fold}_s${seed}.log"
      mkdir -p "$out"
      echo "[$(date -Is)] cell=${cell} fold=${fold} seed=${seed}"
      PYTHONPATH=$PWD python3 -u -m src.invar.training.train \
        --fold "$fold" --seed "$seed" --epochs 30 \
        --output-dir "$out" \
        --regime-axis retrieval --loss-config ranking --bank-size 64 \
        --use-market-gate --gate-beta-init 2.0 --gate-l2-weight 1.0e-4 \
        --save-predictions \
        $extra \
        > "$lf" 2>&1
    done
  done
  echo "Cell ${cell} done"
}

run_cell B --gate-location post_tokenizer --gate-form softmax_F
run_cell C --gate-location input          --gate-form sigmoid
run_cell D --gate-location post_tokenizer --gate-form sigmoid
run_cell F --gate-location input          --gate-form softmax_F --disable-bank

echo "DONE $(date -Is)"
