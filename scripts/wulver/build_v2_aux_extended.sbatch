#!/bin/bash -l
#SBATCH --job-name=panel-aug-build
#SBATCH --partition=general
#SBATCH --account=hz79
#SBATCH --qos=standard
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

set -euo pipefail
cd $HOME/phd-research
module load bright 2>/dev/null || true
module load python3 2>/dev/null || true
source $HOME/phd-research-gpu-env/bin/activate

mkdir -p logs/panel_aug data/processed

echo "host=$(hostname) start=$(date -Is)"

# Extended v2 auxiliary parquets for RAG-STAR Universe F4/F5 sweep:
#   data/processed/risk_features_sp500_ext.parquet         (2015-2026)
#   data/processed/macro_duration_features_sp500_ext.parquet
#   data/processed/sp500_rolling_betas_ext.parquet
# Reads extended LATTICE raw inputs (prices_sp500_extended, extended
# sector_etfs + FRED). Per-ticker rolling-beta is the long pole (~30 min).
PYTHONPATH=$PWD python3 -u -m scripts.lattice.build_v2_aux_extended \
    2>&1 | tee logs/panel_aug/v2_aux_ext.log

echo "DONE $(date -Is)"
