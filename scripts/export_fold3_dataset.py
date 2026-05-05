"""Export the fold-3 dataset (panel + DOW-epiSTAR predictions) to CSV.

Outputs three CSV files under ``results/exports/``:

    1. fold3_panel.csv            -- date, ticker, all 22 panel features,
                                     y_true (5d forward log return),
                                     loss_mask, tradable_mask, age_days
    2. fold3_predictions.csv      -- per-seed DOW-epiSTAR predictions:
                                     date, ticker, seed, y_hat,
                                     score_idio, score_duration
    3. fold3_summary.csv          -- per-day fold-3 daily IC for each
                                     seed (auditing convenience)

Usage:
    python scripts/export_fold3_dataset.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RESULT_DIR = Path("results/dow_epistar_v23_no_rate_memory")
EXPORT_DIR = Path("results/exports")
SEEDS = [42, 43, 44, 45, 46]

# Panel feature names (panel_enriched.py FEATURE_COLS).
PANEL_FEATURE_COLS = [
    "log_return", "log_return_5d", "log_return_20d",
    "log_volume", "log_volume_ratio_20d",
    "realized_vol_20d", "realized_vol_60d",
    "high_low_range", "close_to_high",
    "st_volume_24h", "st_volume_change_30d",
    "st_bullish_ratio", "st_sentiment_dispersion", "st_labeled_ratio",
    "log_market_cap", "cash_runway_q", "rd_intensity",
    "revenue_growth_yoy", "cash_to_mc",
    "shares_outstanding_yoy", "total_assets_growth", "has_fundamentals",
]


def main() -> None:
    """Build and write the three fold-3 CSV files."""
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("[export] loading first seed for index data...")
    p0 = np.load(RESULT_DIR / f"fold3_seed{SEEDS[0]}_predictions.npz", allow_pickle=True)
    y_true = p0["y_true"]                         # [T, N]
    loss_mask = p0["loss_mask"]                   # [T, N]
    tradable = p0["tradable_mask"]                # [T, N]
    test_idx = p0["test_idx"]                     # [n_test_days]
    tickers = list(p0["tickers"])                 # [N]
    dates_raw = list(p0["dates"])                 # [T]
    age_days = p0["age_days"]                     # [T, N]

    panel_dates = pd.to_datetime(dates_raw)

    print(f"[export] fold-3 test span: {panel_dates[test_idx[0]].date()} "
          f"to {panel_dates[test_idx[-1]].date()} ({len(test_idx)} days)")

    # Pull the panel features. They are NOT saved in the predictions npz,
    # so we rebuild the panel here using the same loader.
    print("[export] rebuilding panel...")
    from src.mtgn.training.panel_enriched import (
        EnrichedPanelConfig, build_enriched_panel, panel_to_tensors,
    )
    panel_cfg = EnrichedPanelConfig(
        start_date=pd.Timestamp("2015-01-09"),
        end_date=pd.Timestamp("2022-12-31"),
        horizon_days=5,
        universe_csv=Path("data/raw/biotech_universe_v1.csv"),
    )
    panel, panel_tickers, panel_dates_loaded = build_enriched_panel(panel_cfg)
    tens = panel_to_tensors(panel, panel_tickers, panel_dates_loaded)
    x_raw = tens["x"]                             # [T, N, F=22]
    if list(panel_tickers) != tickers:
        # Defensive: align by intersection.
        ticker_to_idx_panel = {t: i for i, t in enumerate(panel_tickers)}
        keep = [ticker_to_idx_panel[t] for t in tickers if t in ticker_to_idx_panel]
        x_raw = x_raw[:, keep, :]
    assert x_raw.shape[2] == len(PANEL_FEATURE_COLS), (
        f"panel features {x_raw.shape[2]} != "
        f"len(PANEL_FEATURE_COLS)={len(PANEL_FEATURE_COLS)}"
    )

    print("[export] building fold3_panel.csv...")
    rows = []
    for t in test_idx:
        d = panel_dates[t].date()
        for j, ticker in enumerate(tickers):
            if not tradable[t, j]:
                continue
            row = {
                "date": d, "ticker": ticker,
                "y_true_5d_fwd_log_return": float(y_true[t, j]),
                "loss_mask": bool(loss_mask[t, j]),
                "tradable_mask": bool(tradable[t, j]),
                "age_trading_days": int(age_days[t, j]),
            }
            for k, col in enumerate(PANEL_FEATURE_COLS):
                row[col] = float(x_raw[t, j, k])
            rows.append(row)
    panel_df = pd.DataFrame(rows)
    panel_path = EXPORT_DIR / "fold3_panel.csv"
    panel_df.to_csv(panel_path, index=False)
    print(f"[export] wrote {panel_path}: {panel_df.shape}")

    print("[export] building fold3_predictions.csv (5 seeds)...")
    pred_rows = []
    for seed in SEEDS:
        p = np.load(
            RESULT_DIR / f"fold3_seed{seed}_predictions.npz", allow_pickle=True,
        )
        y_hat = p["y_hat"]
        s_idio = p.get("score_idio")
        s_dur = p.get("score_duration")
        for t in test_idx:
            d = panel_dates[t].date()
            for j, ticker in enumerate(tickers):
                if not tradable[t, j]:
                    continue
                row = {
                    "date": d, "ticker": ticker, "seed": int(seed),
                    "y_hat": float(y_hat[t, j]),
                }
                if s_idio is not None:
                    row["score_idio"] = float(s_idio[t, j])
                if s_dur is not None:
                    row["score_duration"] = float(s_dur[t, j])
                pred_rows.append(row)
    pred_df = pd.DataFrame(pred_rows)
    pred_path = EXPORT_DIR / "fold3_predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"[export] wrote {pred_path}: {pred_df.shape}")

    print("[export] building fold3_summary.csv (per-day per-seed IC)...")
    summary_rows = []
    for seed in SEEDS:
        p = np.load(
            RESULT_DIR / f"fold3_seed{seed}_predictions.npz", allow_pickle=True,
        )
        y_hat = p["y_hat"]
        for t in test_idx:
            m = loss_mask[t]
            if m.sum() < 5:
                continue
            a = y_hat[t, m]; b = y_true[t, m]
            if a.std() < 1e-9 or b.std() < 1e-9:
                continue
            ic_p = float(np.corrcoef(a, b)[0, 1])
            a_rank = pd.Series(a).rank().to_numpy()
            b_rank = pd.Series(b).rank().to_numpy()
            ic_s = float(np.corrcoef(a_rank, b_rank)[0, 1]) \
                if a_rank.std() > 1e-9 and b_rank.std() > 1e-9 else 0.0
            summary_rows.append({
                "date": panel_dates[t].date(),
                "seed": int(seed),
                "n_active": int(m.sum()),
                "daily_ic_pearson": ic_p,
                "daily_ic_spearman": ic_s,
            })
    summary_df = pd.DataFrame(summary_rows)
    summary_path = EXPORT_DIR / "fold3_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"[export] wrote {summary_path}: {summary_df.shape}")

    print()
    print("=== EXPORT SUMMARY ===")
    for path in [panel_path, pred_path, summary_path]:
        print(f"  {path}: {path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
