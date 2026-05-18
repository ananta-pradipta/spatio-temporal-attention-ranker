#!/bin/bash -l
#SBATCH --job-name=factorvae-lattice
#SBATCH --partition=gpu
#SBATCH --account=hz79
#SBATCH --qos=standard
#SBATCH --time=04:00:00
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
mkdir -p logs/baselines experiments/invar/baselines/factorvae

echo "host=$(hostname) start=$(date -Is)"
nvidia-smi | head -10 || true

# FactorVAE on LATTICE: F1 + F2 5-seed sweep using the InVAR baseline
# harness (same fold defs, embargo, mask, hybrid loss with aux=0).
# Vendored architecture from src/baselines/vendored/factorvae (AAAI 2022,
# unofficial PyTorch implementation; original authors did not release
# code).
for fold in 1 2; do
  for seed in 42 43 44 45 46; do
    LF="logs/baselines/factorvae_f${fold}_s${seed}.log"
    OD="experiments/invar/baselines"
    echo "[$(date -Is)] fold=${fold} seed=${seed}"
    PYTHONPATH=$PWD python3 -u -m src.invar.baselines.train_baseline \
      --baseline factorvae \
      --fold "$fold" --seed "$seed" --epochs 30 \
      --output-dir "$OD" \
      --save-predictions \
      > "$LF" 2>&1
  done
done

echo "DONE $(date -Is)"
