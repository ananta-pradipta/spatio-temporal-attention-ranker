#!/bin/bash -l
#SBATCH --job-name=model-eval-5
#SBATCH --partition=gpu
#SBATCH --account=hz79
#SBATCH --qos=standard
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --mem=24G
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

echo "=== MERA diagnostic: fold-1 seed-42 with paper-default Phase 2 + 16-epoch Phase 1 ==="
echo "host=$(hostname)  start=$(date -Is)"
nvidia-smi | head -20 || true

REPO="${REPO:-$HOME/phd-research}"
VENV="${VENV:-$HOME/phd-research-gpu-env}"
cd "$REPO"

module load bright 2>/dev/null || true
module load python3 2>/dev/null || true
source "$VENV/bin/activate"
python3 -c "import torch; print('torch', torch.__version__, 'cuda:', torch.cuda.is_available())"

LOG_BASE="logs/mera_diag"
mkdir -p "$LOG_BASE"

OUT_DIR="results/baselines_244/mera_v2"
ORIG_JSON="$OUT_DIR/fold1_seed42.json"
ORIG_NPZ="$OUT_DIR/fold1_seed42_predictions.npz"
BAK_JSON="$OUT_DIR/fold1_seed42.json.original"
BAK_NPZ="$OUT_DIR/fold1_seed42_predictions.npz.original"
DIAG_JSON="$OUT_DIR/fold1_seed42_diag.json"
DIAG_NPZ="$OUT_DIR/fold1_seed42_predictions_diag.npz"

# Backup the original (lr=3e-5, wd=1e-4, patience=1) result so we can restore
echo "[$(date +%H:%M:%S)] backing up original fold1_seed42"
[ -f "$ORIG_JSON" ] && mv "$ORIG_JSON" "$BAK_JSON"
[ -f "$ORIG_NPZ" ] && mv "$ORIG_NPZ" "$BAK_NPZ"

echo "[$(date +%H:%M:%S)] running diagnostic with phase1_epochs=16 phase2_lr=1e-4 phase2_wd=1e-5 phase2_patience=3"
python3 -u -m src.baselines.train_mera_v2 \
  --fold 1 --seed 42 \
  --phase1_epochs 16 \
  --phase2_lr 1e-4 \
  --phase2_weight_decay 1e-5 \
  --phase2_patience 3 \
  > "$LOG_BASE/fold1_seed42_diag.log" 2>&1

echo "[$(date +%H:%M:%S)] saving diagnostic outputs"
[ -f "$ORIG_JSON" ] && mv "$ORIG_JSON" "$DIAG_JSON"
[ -f "$ORIG_NPZ" ] && mv "$ORIG_NPZ" "$DIAG_NPZ"

echo "[$(date +%H:%M:%S)] restoring original fold1_seed42 result"
[ -f "$BAK_JSON" ] && mv "$BAK_JSON" "$ORIG_JSON"
[ -f "$BAK_NPZ" ] && mv "$BAK_NPZ" "$ORIG_NPZ"

echo "[$(date +%H:%M:%S)] diagnostic done"
python3 -c "
import json
orig = json.load(open('$ORIG_JSON'))
diag = json.load(open('$DIAG_JSON'))
print('--- COMPARISON ---')
print(f'Original  (lr=3e-5, wd=1e-4, patience=1, p1=8):  test_ic={orig[\"ic\"]:+.4f}  rank_ic={orig[\"rank_ic\"]:+.4f}')
print(f'Diagnostic (lr=1e-4, wd=1e-5, patience=3, p1=16): test_ic={diag[\"ic\"]:+.4f}  rank_ic={diag[\"rank_ic\"]:+.4f}')
"
echo "end=$(date -Is)"
