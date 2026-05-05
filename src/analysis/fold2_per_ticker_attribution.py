"""Diagnostic 1: Per-ticker attribution on fold 2.

For each of the 84 tickers, measure its contribution to pure STAR's
and REM 3B's daily IC on fold 2.

Method: leave-one-out daily IC. For each ticker i, compute IC on each
day with ticker i masked out; the difference from full-set IC is
ticker i's marginal contribution. Sum over days = ticker i's total
contribution.

Uses SAVED predictions from pure STAR and REM 3B on fold 2 (5 seeds
averaged).

Output:
  docs/fold2_per_ticker_attribution.md + CSV exports.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


RESULTS_PURE_STAR = Path("results/star/audited_pure")
RESULTS_REM_3B = Path("results/investigation/regime_memory")
OUT_MD = Path("docs/fold2_per_ticker_attribution.md")
SEEDS = [42, 43, 44, 45, 46]


def daily_ic(preds: np.ndarray, y: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per-day IC. Returns [T] array."""
    T = preds.shape[0]
    out = np.full(T, np.nan, dtype=np.float64)
    for t in range(T):
        m = mask[t]
        if m.sum() < 3:
            continue
        p = preds[t, m]; q = y[t, m]
        if p.std() < 1e-12 or q.std() < 1e-12:
            continue
        out[t] = float(np.corrcoef(p, q)[0, 1])
    return out


def per_ticker_contribution(preds: np.ndarray, y: np.ndarray, mask: np.ndarray,
                            tickers: list) -> pd.DataFrame:
    """For each ticker i, compute mean daily IC WITH and WITHOUT that ticker
    (leave-one-out). Ticker i's contribution = full_ic - (ic without i).
    A POSITIVE contribution means including this ticker helps IC.
    A NEGATIVE contribution means this ticker drags IC.
    """
    T, N = preds.shape
    rows = []
    full_ic = np.nanmean(daily_ic(preds, y, mask))
    for i, tk in enumerate(tickers):
        mask_drop = mask.copy()
        mask_drop[:, i] = False
        if mask_drop.sum() == 0:
            continue
        ic_drop = np.nanmean(daily_ic(preds, y, mask_drop))
        contribution = full_ic - ic_drop
        # Also per-ticker ticker IC (correlation between this ticker's
        # predicted and true returns across days it is active)
        active = mask[:, i]
        if active.sum() > 5:
            ti_pred = preds[active, i]
            ti_true = y[active, i]
            ti_ic = float(np.corrcoef(ti_pred, ti_true)[0, 1]) if ti_pred.std() > 1e-12 and ti_true.std() > 1e-12 else np.nan
        else:
            ti_ic = np.nan
        # Mean realized return (summed) for that ticker during fold 2
        if active.sum() > 5:
            ti_return = float(y[active, i].sum())
        else:
            ti_return = np.nan
        rows.append({
            "ticker": tk, "n_active_days": int(active.sum()),
            "full_ic": float(full_ic), "ic_without": float(ic_drop),
            "contribution_to_ic": float(contribution),
            "ticker_ic": ti_ic,
            "total_realized_return": ti_return,
        })
    return pd.DataFrame(rows)


def load_averaged_preds(result_dir: Path, fold: int, file_prefix: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """Load and seed-average predictions and targets."""
    preds_stack = []
    y = mask = tickers = None
    for s in SEEDS:
        path = result_dir / f"{file_prefix}{fold}_seed{s}_n100.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        z = np.load(path, allow_pickle=True)
        preds_stack.append(z["preds"])
        if y is None:
            y = z["y"]; mask = z["mask"]; tickers = list(z["tickers"])
    return np.mean(np.stack(preds_stack, axis=0), axis=0), y, mask, tickers


def main():
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)

    # Pure STAR fold 2 (use audited_pure/fold2_seed*.npz; no file_prefix)
    print("Loading pure STAR fold 2 predictions (5 seeds)...")
    preds_ps, y_ps, mask_ps, tickers = load_averaged_preds(RESULTS_PURE_STAR, 2, "fold")
    # Note: audited_pure has filename "fold{F}_seed{S}_n100.npz"
    print(f"  preds shape: {preds_ps.shape}")

    df_ps = per_ticker_contribution(preds_ps, y_ps, mask_ps, tickers)
    df_ps = df_ps.sort_values("contribution_to_ic")

    # REM 3B fold 2 (iter3b_fold2_seed*.npz)
    print("\nLoading REM 3B fold 2 predictions (5 seeds)...")
    preds_rem_stack = []
    y_rem = mask_rem = None
    for s in SEEDS:
        path = RESULTS_REM_3B / f"iter3b_fold2_seed{s}.npz"
        z = np.load(path, allow_pickle=True)
        preds_rem_stack.append(z["preds"])
        if y_rem is None:
            y_rem = z["y"]; mask_rem = z["mask"]
    preds_rem = np.mean(np.stack(preds_rem_stack, axis=0), axis=0)

    df_rem = per_ticker_contribution(preds_rem, y_rem, mask_rem, tickers)
    df_rem = df_rem.sort_values("contribution_to_ic")

    # Save
    df_ps.to_csv(OUT_MD.parent / "fold2_per_ticker_pure_star.csv", index=False)
    df_rem.to_csv(OUT_MD.parent / "fold2_per_ticker_rem_3b.csv", index=False)

    # Build report
    def fmt_row(row: dict, fields: list) -> str:
        return "| " + " | ".join(
            f"{row[f]:+.4f}" if isinstance(row[f], (float, np.floating)) and not np.isnan(row[f])
            else (f"{int(row[f])}" if isinstance(row[f], (int, np.integer))
                  else (str(row[f]) if row[f] is not None else "n/a"))
            for f in fields
        ) + " |"

    worst_ps = df_ps.head(10)
    best_ps = df_ps.tail(10)
    worst_rem = df_rem.head(10)
    best_rem = df_rem.tail(10)

    def tbl(df, fields):
        lines = ["| " + " | ".join(fields) + " |", "|" + "|".join(["---"] * len(fields)) + "|"]
        for _, row in df.iterrows():
            lines.append(fmt_row(row, fields))
        return "\n".join(lines)

    report = f"""# Fold-2 Per-Ticker Attribution (Diagnostic 1, Phase 1)

Date: 2026-04-16
Method: leave-one-out daily IC. For each ticker i, compute mean daily
IC with ticker i included vs excluded. Difference = ticker i's
contribution. POSITIVE = ticker helps; NEGATIVE = ticker drags.

Predictions averaged across 5 seeds on fold 2.

## 1. Pure STAR — 10 biggest draggers

{tbl(worst_ps, ['ticker', 'n_active_days', 'contribution_to_ic', 'ticker_ic', 'total_realized_return'])}

## 2. Pure STAR — 10 biggest positive contributors

{tbl(best_ps, ['ticker', 'n_active_days', 'contribution_to_ic', 'ticker_ic', 'total_realized_return'])}

## 3. REM 3B — 10 biggest draggers

{tbl(worst_rem, ['ticker', 'n_active_days', 'contribution_to_ic', 'ticker_ic', 'total_realized_return'])}

## 4. REM 3B — 10 biggest positive contributors

{tbl(best_rem, ['ticker', 'n_active_days', 'contribution_to_ic', 'ticker_ic', 'total_realized_return'])}

## 5. Summary statistics

| Metric | Pure STAR | REM 3B |
|---|---|---|
| Mean contribution | {df_ps['contribution_to_ic'].mean():+.5f} | {df_rem['contribution_to_ic'].mean():+.5f} |
| Fraction NEGATIVE | {(df_ps['contribution_to_ic'] < 0).mean():.2f} | {(df_rem['contribution_to_ic'] < 0).mean():.2f} |
| Number NEGATIVE | {(df_ps['contribution_to_ic'] < 0).sum()} | {(df_rem['contribution_to_ic'] < 0).sum()} |
| Mean ticker IC | {df_ps['ticker_ic'].mean():+.4f} | {df_rem['ticker_ic'].mean():+.4f} |

## 6. Overlap between models

**Shared top-10 draggers (tickers that drag both pure STAR AND REM 3B):**
{', '.join(sorted(set(df_ps.head(10)['ticker'].tolist()) & set(df_rem.head(10)['ticker'].tolist())))}

**Shared top-10 contributors:**
{', '.join(sorted(set(df_ps.tail(10)['ticker'].tolist()) & set(df_rem.tail(10)['ticker'].tolist())))}

## 7. Interpretation

Check the "biggest draggers" list for common patterns:
- Are they all clinical-stage small-caps? (supports quality-factor-flip hypothesis)
- Do they share sector / stage / market-cap characteristics?
- Do the same tickers drag both models? (suggests universe-level issue, not architecture-specific)

If, say, 20 of 84 tickers account for >80% of the fold-2 IC drag,
pruning those tickers (with a principled criterion) could lift
fold-2 IC substantially. This would be a dataset design choice
for a revised sub-universe.

## 8. Files

- `docs/fold2_per_ticker_pure_star.csv`
- `docs/fold2_per_ticker_rem_3b.csv`
"""
    OUT_MD.write_text(report)
    print(f"\nwrote {OUT_MD}")

    print("\n=== Top 10 draggers (Pure STAR) ===")
    print(df_ps.head(10)[["ticker", "contribution_to_ic", "ticker_ic", "total_realized_return"]].to_string(index=False))
    print("\n=== Top 10 draggers (REM 3B) ===")
    print(df_rem.head(10)[["ticker", "contribution_to_ic", "ticker_ic", "total_realized_return"]].to_string(index=False))
    print("\n=== Overlap shared draggers ===")
    print(sorted(set(df_ps.head(10)['ticker'].tolist()) & set(df_rem.head(10)['ticker'].tolist())))


if __name__ == "__main__":
    main()
