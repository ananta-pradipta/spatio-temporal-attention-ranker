"""Cross-architecture per-sector IC analysis on the universal panel.

Reads prediction NPZ files from every sweep (RAG-STAR headline + each
baseline) and computes per-sector mean Pearson IC and Spearman rank IC
for every (sector, model, fold, seed) tuple. Aggregates to:

  Output 1 (table for Section 7.3): per-sector x per-model 5-seed mean IC,
                                    one stacked table per fold.
  Output 2 (case-study figures):    architecture-sector affinity matrix
                                    (which model wins each sector).

Run from repo root after the universal-panel sweeps complete:

    python -m scripts.analysis.cross_sector_ic \\
        --output drafts/universal_paper_aaai/data/sector_ic.json \\
        --markdown drafts/universal_paper_aaai/data/sector_ic.md
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


# Models to include and their result directories. Add new baselines here.
MODEL_DIRS = {
    "RAG-STAR":             "results/rag_star_universe_v2_no_ipo",
    "FactorVAE":            "results/baselines_universal_two_regime_val/factorvae",
    "MASTER":               "results/baselines_universal_two_regime_val/master",
    "DyStAGE":              "results/baselines_universal_two_regime_val/dystage",
    "iTransformer":         "results/baselines_universal_two_regime_val/itransformer",
    "MERA":                 "results/baselines_universal_two_regime_val/mera",
    "StockMixer":           "results/baselines_universal_two_regime_val/stockmixer",
}

GICS_SECTORS = [
    "Communication Services", "Consumer Discretionary", "Consumer Staples",
    "Energy", "Financials", "Health Care", "Industrials",
    "Information Technology", "Materials", "Real Estate", "Utilities",
]
SEEDS = (42, 43, 44, 45, 46)
FOLDS = (1, 2, 3, 4, 5)


def load_sector_mapping(path: Path) -> dict[str, str]:
    """Returns {ticker: sector_name} mapping from the cached JSON."""
    with open(path) as f:
        d = json.load(f)
    ticker_to_id = d["ticker_to_sector_id"]
    id_to_sector = d["id_to_sector"]
    return {
        ticker: id_to_sector[str(id_)]
        for ticker, id_ in ticker_to_id.items()
    }


def per_sector_ic(
    y_hat: np.ndarray, y_true: np.ndarray, loss_mask: np.ndarray,
    tickers: np.ndarray, ticker_to_sector: dict[str, str],
    rank: bool = False,
) -> dict[str, float]:
    """Return per-sector mean per-day IC across the test window.

    For each test day, compute IC over the cells that are (a) in the loss
    mask (b) belong to a given sector. Average across days.
    """
    sector_ids = np.array([
        ticker_to_sector.get(str(t).upper(), "UNK") for t in tickers
    ])
    sector_ic: dict[str, float] = {}
    for sector in GICS_SECTORS:
        in_sector = sector_ids == sector
        if in_sector.sum() < 5:
            sector_ic[sector] = np.nan
            continue
        per_day = []
        for t in range(y_hat.shape[0]):
            mask = loss_mask[t] & in_sector
            if mask.sum() < 5:
                continue
            a = y_hat[t, mask]
            b = y_true[t, mask]
            if rank:
                a = pd.Series(a).rank().to_numpy()
                b = pd.Series(b).rank().to_numpy()
            if a.std() < 1e-9 or b.std() < 1e-9:
                continue
            c = float(np.corrcoef(a, b)[0, 1])
            if not np.isnan(c):
                per_day.append(c)
        sector_ic[sector] = float(np.mean(per_day)) if per_day else np.nan
    return sector_ic


def aggregate_results(
    repo_root: Path, ticker_to_sector: dict[str, str],
) -> dict:
    """Returns nested dict[model][fold][sector] -> (mean_ic, std_ic, mean_rank_ic, std_rank_ic, n_seeds)."""
    results: dict[str, dict[int, dict[str, dict]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(dict))
    )

    for model_label, dir_ in MODEL_DIRS.items():
        model_dir = repo_root / dir_
        if not model_dir.exists():
            print(f"[skip] {model_label}: dir not found at {model_dir}", flush=True)
            continue
        # Per (fold, seed) collect per-sector IC + rank IC.
        per_fold_seed: dict[int, dict[int, dict[str, dict[str, float]]]] = defaultdict(dict)
        for fold in FOLDS:
            for seed in SEEDS:
                npz_path = model_dir / f"fold{fold}_seed{seed}_predictions.npz"
                if not npz_path.exists():
                    continue
                p = np.load(npz_path)
                y_hat = p["y_hat"]
                y_true = p["y_true"]
                loss_mask = p["loss_mask"].astype(bool)
                tickers = p["tickers"]
                ic = per_sector_ic(y_hat, y_true, loss_mask, tickers, ticker_to_sector, rank=False)
                rank_ic = per_sector_ic(y_hat, y_true, loss_mask, tickers, ticker_to_sector, rank=True)
                per_fold_seed[fold][seed] = {"ic": ic, "rank_ic": rank_ic}
                print(f"[{model_label}] fold={fold} seed={seed}: parsed", flush=True)

        # Aggregate across seeds.
        for fold, seeds_dict in per_fold_seed.items():
            for sector in GICS_SECTORS:
                ics = [seeds_dict[s]["ic"][sector]
                       for s in seeds_dict if not np.isnan(seeds_dict[s]["ic"][sector])]
                rics = [seeds_dict[s]["rank_ic"][sector]
                        for s in seeds_dict if not np.isnan(seeds_dict[s]["rank_ic"][sector])]
                if not ics:
                    continue
                m = float(np.mean(ics))
                sd = float(np.std(ics)) if len(ics) > 1 else 0.0
                rm = float(np.mean(rics))
                rsd = float(np.std(rics)) if len(rics) > 1 else 0.0
                results[model_label][fold][sector] = {
                    "mean_ic": m, "std_ic": sd,
                    "mean_rank_ic": rm, "std_rank_ic": rsd,
                    "n_seeds": len(ics),
                }
    return dict(results)


def render_markdown(results: dict, out_path: Path) -> None:
    """Render the aggregated results as a per-fold per-sector x per-model markdown table."""
    lines: list[str] = ["# Per-sector x per-model IC on the universal panel\n"]
    lines.append(f"Computed across {len(SEEDS)} seeds per (fold, model). "
                 "Cells are mean Pearson IC; cell `--` indicates n<3 seeds.\n")
    for fold in FOLDS:
        lines.append(f"\n## Fold {fold}\n")
        models = list(results.keys())
        if not models:
            continue
        # Build a sector x model table.
        header = "| sector |" + "|".join(f" {m} " for m in models) + "|"
        sep    = "|--------|" + "|".join(":------:" for _ in models) + "|"
        lines.append(header)
        lines.append(sep)
        for sector in GICS_SECTORS:
            row = f"| {sector} |"
            for m in models:
                cell_d = results.get(m, {}).get(fold, {}).get(sector, None)
                if cell_d is None or cell_d.get("n_seeds", 0) < 3:
                    row += " -- |"
                else:
                    row += f" {cell_d['mean_ic']:+.4f} |"
            lines.append(row)
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {out_path}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=str, default=".",
                   help="Repo root; default cwd.")
    p.add_argument("--sector_mapping", type=str,
                   default="data/cache/sector_mapping.json")
    p.add_argument("--output", type=str,
                   default="drafts/universal_paper_aaai/data/sector_ic.json")
    p.add_argument("--markdown", type=str,
                   default="drafts/universal_paper_aaai/data/sector_ic.md")
    args = p.parse_args()

    repo = Path(args.repo).resolve()
    ticker_to_sector = load_sector_mapping(repo / args.sector_mapping)
    print(f"[main] sector mapping: {len(ticker_to_sector)} tickers", flush=True)

    results = aggregate_results(repo, ticker_to_sector)

    out_json = repo / args.output
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out_json}", flush=True)

    render_markdown(results, repo / args.markdown)


if __name__ == "__main__":
    main()
