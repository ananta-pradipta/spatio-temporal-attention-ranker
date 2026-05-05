# Wulver (NJIT HPC) Jobs

Jobs intended to run on NJIT's Wulver HPC cluster with the Slurm scheduler.

## Submission workflow

1. SSH into Wulver: `ssh <ucid>@wulver.njit.edu` (from NJIT network or VPN).
2. Ensure the repository is up to date:
   ```bash
   cd ~/phd-research && git pull
   ```
3. (One-time) create a Python venv with the repo's deps:
   ```bash
   cd ~/phd-research
   module load anaconda3   # or the module that gives python3.10+
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
4. Submit the job:
   ```bash
   sbatch scripts/wulver/CS785-GetExtData.sbatch
   ```
5. Monitor:
   ```bash
   squeue -u $USER
   tail -f logs/wulver/CS785-GetExtData_*.out
   ```
6. Pull outputs back to your local machine when the job is done. The job
   packages the parquet output as a tar.zst (falls back to tar.gz) under
   `data/raw/stocktwits_biotech_<YYYYMMDD>.tar.zst`. Transfer only the
   archive, not the raw parquet directory:
   ```bash
   # from local machine
   rsync -avz --progress \
       <ucid>@wulver.njit.edu:~/phd-research/data/raw/stocktwits_biotech_*.tar.zst \
       data/raw/
   # unpack locally
   tar --zstd -xf data/raw/stocktwits_biotech_*.tar.zst -C data/raw/
   ```
   If the cluster lacks `zstd`, substitute `.tar.gz` and `tar -xzf`.

## Jobs

### `CS785-GetExtData`

Downloads the biotech-universe-filtered subset of the public StockTwits 2008
to 2022 corpus from `s3://stocktwits-nyu/dataset/v1/data/csv`. Filters
`symbols/` on `symbol.isin(universe)` via dask, then joins `msg_info/` on
the filtered `message_id` set. Writes two parquet files:

- `data/raw/stocktwits/symbols.parquet`
- `data/raw/stocktwits/msg_info.parquet`

Expected size after biotech filter: roughly 200 to 600 MB total (full
corpus is ~11 GB before filter; 284 tickers represent ~2 to 5 percent of
the 7M tickers on StockTwits).

Resources: 8 cores, 32 GB RAM, 6h wall time. Partition defaults to
`public`; override with `sbatch --partition=<name>`. Use the Wulver
allocation your advisor has access to.

Logs: `logs/wulver/CS785-GetExtData_<jobid>.out` and `.err`.
