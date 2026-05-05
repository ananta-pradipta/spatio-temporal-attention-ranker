#!/bin/bash -l
#SBATCH --job-name=CS785-GetExtData
#SBATCH --partition=course
#SBATCH --account=2026-spring-cs-785-hz79-adp232
#SBATCH --qos=course
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

set -euo pipefail

echo "=== CS785-GetExtData ==="
echo "host=$(hostname) job_id=${SLURM_JOB_ID:-unset}"
echo "start=$(date -Is)"

# -------- Project layout -------- #
REPO="${REPO:-$HOME/phd-research}"
VENV="${VENV:-$HOME/phd-research-env}"

cd "$REPO"

# -------- Environment -------- #
module load bright 2>/dev/null || true
module load python3 2>/dev/null || true

if [ ! -d "$VENV" ]; then
    echo "Creating venv at $VENV"
    python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet

python3 -c "import pandas, dask, s3fs, pyarrow; print('deps ok')"

# -------- Scratch / cache -------- #
export TMPDIR="${SLURM_TMPDIR:-/tmp}/CS785-GetExtData-${SLURM_JOB_ID:-$$}"
mkdir -p "$TMPDIR"
export DASK_TEMPORARY_DIRECTORY="$TMPDIR"

# -------- Inputs -------- #
UNIVERSE_CSV="data/raw/biotech_universe_v1.csv"
OUT_DIR="data/raw/stocktwits"
CATEGORIES="symbols msg_info"

EXTRACTED="$OUT_DIR/_tickers.txt"
mkdir -p "$OUT_DIR"
python3 - <<PY
import pandas as pd
df = pd.read_csv("$UNIVERSE_CSV")
df["ticker"].dropna().astype(str).str.strip().str.upper().drop_duplicates().to_csv(
    "$EXTRACTED", index=False, header=False
)
print(f"wrote ticker list: $EXTRACTED")
PY

echo "tickers=$EXTRACTED out=$OUT_DIR categories=$CATEGORIES"

# -------- Run -------- #
python3 -m src.mtgn.data.download_stocktwits \
    --tickers-file "$EXTRACTED" \
    --categories $CATEGORIES \
    --out-dir "$OUT_DIR"

# -------- Package the results -------- #
ARCHIVE_BASENAME="stocktwits_biotech_$(date +%Y%m%d)"
ARCHIVE_DIR="data/raw"

if command -v zstd >/dev/null 2>&1; then
    tar --zstd -cf "$ARCHIVE_DIR/${ARCHIVE_BASENAME}.tar.zst" \
        -C "$ARCHIVE_DIR" "$(basename "$OUT_DIR")"
    ARCHIVE_PATH="$ARCHIVE_DIR/${ARCHIVE_BASENAME}.tar.zst"
else
    tar -czf "$ARCHIVE_DIR/${ARCHIVE_BASENAME}.tar.gz" \
        -C "$ARCHIVE_DIR" "$(basename "$OUT_DIR")"
    ARCHIVE_PATH="$ARCHIVE_DIR/${ARCHIVE_BASENAME}.tar.gz"
fi

echo "archive: $ARCHIVE_PATH"
ls -lh "$ARCHIVE_PATH" "$OUT_DIR" || true

echo "end=$(date -Is)"
