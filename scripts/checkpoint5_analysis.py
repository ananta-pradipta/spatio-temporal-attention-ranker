"""Phase 5 analysis: assemble CHECKPOINT 5 stats from the 10 Wulver runs.

Produces:
  - per-seed test IC, rIC, NDCG@10, NDCG@50 for full + corr_only
  - paired t-test (df=4) for full vs corr_only
  - cohort decomposition by S&P 500 membership tenure (young_in_index_252d
    vs seasoned_253d) using sp500_constituents_history
  - gate-weight diagnostics (test-window mean, std of w_dur) per run
  - bicotech F1 comparison (cached from RAG-STAR paper)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats as scipy_stats

from src.v2.training.folds import fold_indices


def load_run_metrics(folder: Path) -> dict:
    out = {}
    for f in sorted(folder.glob("fold1_seed*.json")):
        seed = int(f.stem.split("seed")[-1])
        d = json.loads(f.read_text())
        out[seed] = d
    return out


def per_seed_table(metrics: dict, label: str) -> pd.DataFrame:
    rows = []
    for s, d in sorted(metrics.items()):
        rows.append({
            "variant": label, "seed": s,
            "ic": d.get("ic"), "rank_ic": d.get("rank_ic"),
            "ndcg10": d.get("ndcg10"), "ndcg50": d.get("ndcg50"),
            "val_ic": d.get("val_ic"),
        })
    return pd.DataFrame(rows)


def paired_ttest(full: dict, corr: dict) -> dict:
    seeds = sorted(set(full.keys()) & set(corr.keys()))
    deltas = [full[s]["ic"] - corr[s]["ic"] for s in seeds]
    arr = np.asarray(deltas)
    mean = float(arr.mean()); sd = float(arr.std(ddof=1))
    se = sd / np.sqrt(len(arr))
    t_stat = mean / max(se, 1e-9)
    p_two = float(scipy_stats.t.sf(abs(t_stat), df=len(arr) - 1) * 2)
    return {
        "seeds": seeds, "deltas": deltas,
        "mean_delta": mean, "std_delta": sd, "se": float(se),
        "t_stat": float(t_stat), "df": len(arr) - 1,
        "p_two_sided": p_two,
        "ci_95": [float(mean - 2.776 * se), float(mean + 2.776 * se)],
    }


def cohort_membership_tenure(predictions_npz: Path) -> dict:
    """Compute test-window IC stratified by S&P 500 membership tenure.

    Cohorts (per spec 5d):
      - young_in_index_252d: ticker has been in S&P 500 < 252 trading days
        as of test day t
      - seasoned_253d: ticker >= 252 trading days in the index
    """
    snap = torch.load("data/processed/sp500_snapshots.pt", weights_only=False)
    tickers = snap["tickers"]
    dates = pd.to_datetime(snap["dates"])
    mask = snap["mask"].numpy() if hasattr(snap["mask"], "numpy") else np.asarray(snap["mask"])
    y = snap["y"].numpy() if hasattr(snap["y"], "numpy") else np.asarray(snap["y"])

    # Predictions: y_hat tensor [T, N]
    pred = np.load(predictions_npz)
    y_hat = pred["y_hat"] if "y_hat" in pred else pred["predictions"]

    # Membership intervals
    hist = pd.read_parquet("data/raw/sp500/sp500_constituents_history.parquet")
    hist["start_date"] = pd.to_datetime(hist["start_date"])
    hist["end_date"]   = pd.to_datetime(hist["end_date"])
    iv_by_ticker: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] = {}
    for _, r in hist.iterrows():
        iv_by_ticker.setdefault(r.ticker, []).append((r.start_date, r.end_date))

    # Fold 1 test indices
    train_idx, val_idx, test_idx = fold_indices(1, list(dates))
    cohort_young: list[float] = []
    cohort_seasoned: list[float] = []

    for t in test_idx:
        d_t = dates[t]
        active = np.where(mask[t])[0]
        if len(active) < 5: continue
        tenure_days = []
        for n in active:
            tk = tickers[n]
            ivs = iv_by_ticker.get(tk, [])
            current_iv_start = None
            for s, e in ivs:
                if s <= d_t <= e:
                    current_iv_start = s; break
            tenure_days.append(((d_t - current_iv_start).days)
                                if current_iv_start is not None else 99999)
        tenure_days = np.asarray(tenure_days)
        for cohort_name, mask_cohort in [
            ("young_in_index_252d", tenure_days < 252 * 1.5),  # 252 trading ≈ 1 calendar yr
            ("seasoned_253d",       tenure_days >= 252 * 1.5),
        ]:
            sel = active[mask_cohort]
            if len(sel) < 5: continue
            yh = y_hat[t, sel]; yt = y[t, sel]
            if yh.std() < 1e-9 or yt.std() < 1e-9: continue
            ic = float(np.corrcoef(yh, yt)[0, 1])
            if cohort_name == "young_in_index_252d":
                cohort_young.append(ic)
            else:
                cohort_seasoned.append(ic)

    return {
        "young_in_index_252d_ic":  float(np.mean(cohort_young))    if cohort_young else None,
        "young_in_index_252d_n":   len(cohort_young),
        "seasoned_253d_ic":        float(np.mean(cohort_seasoned)) if cohort_seasoned else None,
        "seasoned_253d_n":         len(cohort_seasoned),
    }


def gate_weight_diagnostics(predictions_npz: Path) -> dict:
    """Extract test-window w_dur statistics from the saved predictions."""
    pred = np.load(predictions_npz)
    if "test_w_corr" not in pred or "test_w_dur" not in pred:
        return {"available": False, "reason": "predictions_npz missing gate fields"}
    w_dur = pred["test_w_dur"]; w_corr = pred["test_w_corr"]
    return {
        "available": True,
        "w_dur_mean": float(w_dur.mean()),
        "w_dur_std":  float(w_dur.std()),
        "w_corr_mean": float(w_corr.mean()),
        "fraction_w_dur_above_0.05": float((w_dur > 0.05).mean()),
        "n_test_days": int(w_dur.shape[0]),
    }


def main() -> None:
    full_dir = Path("results/universal_validation")
    corr_dir = Path("results/universal_validation_corr_only")
    full = load_run_metrics(full_dir)
    corr = load_run_metrics(corr_dir)

    full_df = per_seed_table(full, "full")
    corr_df = per_seed_table(corr, "corr_only")
    print("=== Per-seed test metrics ===")
    print(pd.concat([full_df, corr_df], ignore_index=True).to_string(index=False))

    print("\n=== Aggregate (mean ± std across 5 seeds) ===")
    for label, df in [("full", full_df), ("corr_only", corr_df)]:
        for col in ("ic", "rank_ic", "ndcg10", "ndcg50"):
            arr = df[col].dropna().to_numpy(dtype=np.float64)
            if len(arr) == 0:
                print(f"  {label:<10} {col:<8}  no data")
            else:
                print(f"  {label:<10} {col:<8}  {arr.mean():+.4f} ± {arr.std(ddof=1):.4f}  (n={len(arr)})")

    print("\n=== Paired t-test (full vs corr_only on test IC) ===")
    tt = paired_ttest(full, corr)
    print(f"  per-seed deltas: {[f'{d:+.4f}' for d in tt['deltas']]}")
    print(f"  mean_delta: {tt['mean_delta']:+.4f}  std: {tt['std_delta']:.4f}  se: {tt['se']:.4f}")
    print(f"  t_stat: {tt['t_stat']:.3f}  df: {tt['df']}  two-sided p: {tt['p_two_sided']:.4f}")
    print(f"  95% CI for mean delta: [{tt['ci_95'][0]:+.4f}, {tt['ci_95'][1]:+.4f}]")

    print("\n=== Cohort decomposition by S&P 500 membership tenure (full variant) ===")
    cohort_results = []
    for s in sorted(full.keys()):
        npz_path = full_dir / f"fold1_seed{s}_predictions.npz"
        if not npz_path.exists():
            print(f"  seed {s}: predictions npz missing")
            continue
        coh = cohort_membership_tenure(npz_path)
        cohort_results.append({"seed": s, **coh})
        print(f"  seed {s}: young_in_index_252d ic={coh.get('young_in_index_252d_ic')}  "
              f"(n={coh.get('young_in_index_252d_n')})  "
              f"seasoned_253d ic={coh.get('seasoned_253d_ic')}  "
              f"(n={coh.get('seasoned_253d_n')})")

    if cohort_results:
        young_ics = [r['young_in_index_252d_ic'] for r in cohort_results if r['young_in_index_252d_ic'] is not None]
        seasoned_ics = [r['seasoned_253d_ic']   for r in cohort_results if r['seasoned_253d_ic'] is not None]
        print(f"\n  young_in_index_252d  mean={np.mean(young_ics):+.4f} ± {np.std(young_ics, ddof=1):.4f}  (across {len(young_ics)} seeds)")
        print(f"  seasoned_253d        mean={np.mean(seasoned_ics):+.4f} ± {np.std(seasoned_ics, ddof=1):.4f}  (across {len(seasoned_ics)} seeds)")

    print("\n=== Gate weight diagnostics (full variant) ===")
    gate_results = []
    for s in sorted(full.keys()):
        npz_path = full_dir / f"fold1_seed{s}_predictions.npz"
        if npz_path.exists():
            g = gate_weight_diagnostics(npz_path)
            print(f"  seed {s}: {g}")
            gate_results.append({"seed": s, **g})

    # Save full report
    out = {
        "full_per_seed": full_df.to_dict(orient="records"),
        "corr_per_seed": corr_df.to_dict(orient="records"),
        "paired_t_test": tt,
        "cohort_per_seed": cohort_results,
        "gate_per_seed": gate_results,
    }
    Path("logs/universal_validation/phase5_checkpoint5.json").write_text(
        json.dumps(out, indent=2, default=str))
    print(f"\nWrote logs/universal_validation/phase5_checkpoint5.json")


if __name__ == "__main__":
    main()
