#!/bin/bash -l
#SBATCH --job-name=invar-itrans-redo
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
mkdir -p logs/invar experiments/invar/baselines/itransformer
echo "host=$(hostname) start=$(date -Is)"
nvidia-smi | head -10 || true
for FOLD in 1 2; do
  for S in 42 43 44 45 46; do
    OUT="experiments/invar/baselines/itransformer/fold${FOLD}/seed${S}/results.json"
    [ -f "$OUT" ] && echo "skip $OUT" && continue
    LF="logs/invar/baseline_itransformer_fold${FOLD}_seed${S}.log"
    PYTHONPATH=$PWD python3 -u -m src.invar.baselines.train_baseline \
      --baseline itransformer --fold "$FOLD" --seed "$S" --epochs 10 \
      --output-dir experiments/invar/baselines --save-predictions \
      > "$LF" 2>&1
  done
done
echo "DONE $(date -Is)"
