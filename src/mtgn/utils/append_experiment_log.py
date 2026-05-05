"""Append an MTGN run result to results/experiment_log.md.

Reads a run JSON written by src.mtgn.training.train_mtgn (or train.py)
and appends a row to the MTGN ablation table under the bottom of the
experiment log. Keeps a markdown table that is easy to diff and that
accumulates all runs in chronological order.

Usage:
    python3 -m src.mtgn.utils.append_experiment_log results/mtgn_cross_full.json
    python3 -m src.mtgn.utils.append_experiment_log results/*.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path


MTGN_SECTION_MARKER = "## MTGN runs (auto-appended)"

TABLE_HEADER = (
    "| Timestamp | Tag | Mode | N_tickers | Start | End | Horizon | Epochs | "
    "Edges | Store final | Test IC | Test RankIC | Test loss |"
)
TABLE_SEPARATOR = (
    "|-----------|-----|------|----------:|-------|-----|--------:|-------:|"
    "------:|------------:|--------:|------------:|---------:|"
)


def _read_run(path: Path) -> dict:
    return json.loads(path.read_text())


def _row_for(run: dict, tag: str) -> str:
    cfg = run.get("config", {})
    return (
        f"| {dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} "
        f"| {tag} "
        f"| {cfg.get('retrieval_mode', 'n/a')} "
        f"| {run.get('panel_N', '?')} "
        f"| {cfg.get('start_date', '?')} "
        f"| {cfg.get('end_date', '?')} "
        f"| {cfg.get('horizon_days', '?')} "
        f"| {cfg.get('epochs', '?')} "
        f"| {run.get('edges', '?')} "
        f"| {run.get('final_store_size', '?')} "
        f"| {run.get('test_ic', float('nan')):+.4f} "
        f"| {run.get('test_rank_ic', float('nan')):+.4f} "
        f"| {run.get('test_loss', float('nan')):.4f} |"
    )


def ensure_section(log_path: Path) -> str:
    text = log_path.read_text() if log_path.exists() else "# Experiment Log\n\n"
    if MTGN_SECTION_MARKER not in text:
        text += (
            "\n\n"
            + MTGN_SECTION_MARKER
            + "\n\n"
            + TABLE_HEADER + "\n"
            + TABLE_SEPARATOR + "\n"
        )
    return text


def append_runs(run_paths: list[Path], log_path: Path, tag: str) -> None:
    text = ensure_section(log_path)
    for rp in run_paths:
        if not rp.exists():
            print(f"  skip (missing): {rp}")
            continue
        run = _read_run(rp)
        row = _row_for(run, tag or rp.stem)
        text = text.rstrip() + "\n" + row + "\n"
        print(f"  appended: {rp.name}  IC={run.get('test_ic', float('nan')):+.4f}")
    log_path.write_text(text)
    print(f"Updated {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--log", type=Path, default=Path("results/experiment_log.md"))
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    append_runs(args.runs, args.log, args.tag)


if __name__ == "__main__":
    main()
