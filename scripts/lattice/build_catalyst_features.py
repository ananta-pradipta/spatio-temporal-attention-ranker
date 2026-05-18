"""Phase 5b C: build catalyst features from the earnings calendar.

Per spec section 6.2: for each (date, ticker) panel cell compute
``days_to_next_catalyst`` as the minimum positive number of trading days
to the next earnings event from the calendar. If the gap is more than 10,
all three catalyst features are zero (event outside the encoder window).
If the gap is 1 to 10, the features are::

    days_to_next_catalyst_sin = sin(2 * pi * days / 10)
    days_to_next_catalyst_cos = cos(2 * pi * days / 10)
    catalyst_type_id = 1                                          # earnings

Saves to ``data/lattice/processed/catalyst_features.parquet`` and updates
the existing ``panel_features.parquet`` in place by overwriting the three
catalyst columns. Panel feature order and column names are unchanged.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


HORIZON_TRADING_DAYS = 10
EARNINGS_TYPE_ID = 1


def trading_day_index(panel_dates: pd.Series) -> dict:
    """Return ``{date -> integer trading day index}`` for the panel calendar."""
    sorted_dates = sorted(panel_dates.dropna().unique())
    return {d: i for i, d in enumerate(sorted_dates)}


def map_event_to_next_trading_day(
    event_date: pd.Timestamp,
    sorted_panel_dates: np.ndarray,
) -> int | None:
    """Return integer trading-day index of the first panel date >= event_date.

    yfinance earnings dates are sometimes after-market on day t; we treat the
    event as happening on the closest trading day at or after the timestamp.
    Returns None if event_date is past the panel window.
    """
    pos = int(np.searchsorted(sorted_panel_dates, event_date, side="left"))
    if pos >= len(sorted_panel_dates):
        return None
    return pos


def compute_days_to_next(
    panel_di: np.ndarray,
    event_di_per_ticker: dict[str, list[int]],
    panel_tickers: np.ndarray,
) -> np.ndarray:
    """For each (date, ticker) panel row, compute trading days to the next
    event for that ticker. Returns an int array, ``-1`` for no future event.
    """
    out = np.full(len(panel_di), -1, dtype=np.int64)
    for ticker, events in event_di_per_ticker.items():
        if not events:
            continue
        events_arr = np.asarray(sorted(set(events)), dtype=np.int64)
        rows = (panel_tickers == ticker)
        if not rows.any():
            continue
        my_di = panel_di[rows]
        # For each my_di find the first event >= my_di
        idx = np.searchsorted(events_arr, my_di, side="left")
        # If on event day, treat as days=0 (immediate). If we want strictly
        # future events (days >= 1), use side='right'. Spec section 6.2 says
        # "minimum positive number of trading days to the next earnings
        # event"; we interpret 0 as "today is earnings day" and assign that
        # to the 10-day horizon by setting days = 0 -> sin(0) = 0,
        # cos(0) = 1, type_id = 1.
        valid = idx < len(events_arr)
        days = np.full(len(my_di), -1, dtype=np.int64)
        days[valid] = events_arr[idx[valid]] - my_di[valid]
        out[rows] = days
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--calendar", type=str,
                    default="data/lattice/raw/earnings_calendar.parquet")
    p.add_argument("--panel", type=str,
                    default="data/lattice/processed/panel_features.parquet")
    p.add_argument("--out", type=str,
                    default="data/lattice/processed/catalyst_features.parquet")
    p.add_argument("--coverage-out", type=str,
                    default="experiments/lattice/diagnostics/phase_c_earnings_coverage.md")
    p.add_argument("--update-panel", action="store_true",
                    help="Overwrite the catalyst columns in panel_features.parquet.")
    args = p.parse_args()

    cal = pd.read_parquet(args.calendar)
    print(f"[catalyst build] calendar: {len(cal)} rows, "
          f"{cal['ticker'].nunique()} tickers", flush=True)
    cal["event_date"] = pd.to_datetime(cal["event_date"])

    panel = pd.read_parquet(args.panel)
    panel["date"] = pd.to_datetime(panel["date"])
    print(f"[catalyst build] panel: {len(panel)} rows, "
          f"{panel['ticker'].nunique()} tickers", flush=True)

    sorted_panel_dates = np.array(sorted(panel["date"].unique()))
    di_lookup = {d: i for i, d in enumerate(sorted_panel_dates)}
    panel_di = panel["date"].map(di_lookup).to_numpy(dtype=np.int64)
    panel_tickers = panel["ticker"].to_numpy()

    # Build event_di_per_ticker
    event_di_per_ticker: dict[str, list[int]] = {}
    for ticker, group in cal.groupby("ticker"):
        events = []
        for ed in group["event_date"]:
            di = map_event_to_next_trading_day(ed, sorted_panel_dates)
            if di is not None:
                events.append(di)
        event_di_per_ticker[ticker] = events

    days_to_next = compute_days_to_next(panel_di, event_di_per_ticker, panel_tickers)
    panel = panel.copy()
    panel["days_to_next_catalyst"] = days_to_next

    # Compute the three catalyst features per spec.
    sin_col = np.zeros(len(panel), dtype=np.float32)
    cos_col = np.zeros(len(panel), dtype=np.float32)
    type_col = np.zeros(len(panel), dtype=np.int8)
    in_window = (days_to_next >= 0) & (days_to_next <= HORIZON_TRADING_DAYS)
    if in_window.any():
        d = days_to_next[in_window].astype(np.float32)
        sin_col[in_window] = np.sin(2 * np.pi * d / HORIZON_TRADING_DAYS).astype(np.float32)
        cos_col[in_window] = np.cos(2 * np.pi * d / HORIZON_TRADING_DAYS).astype(np.float32)
        type_col[in_window] = EARNINGS_TYPE_ID

    out_df = panel[["date", "ticker"]].copy()
    out_df["days_to_next_catalyst_sin"] = sin_col
    out_df["days_to_next_catalyst_cos"] = cos_col
    out_df["catalyst_type_id"] = type_col.astype(np.int64)
    out_df["days_to_next_catalyst"] = days_to_next.astype(np.int64)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.out, index=False)
    print(f"[catalyst build] wrote {args.out}: {len(out_df)} rows", flush=True)

    # Coverage report
    n_cells = len(panel)
    n_within_90 = int(((days_to_next >= 0) & (days_to_next <= 90)).sum())
    n_within_horizon = int(in_window.sum())
    pct_within_90 = 100.0 * n_within_90 / n_cells if n_cells else 0.0
    pct_within_horizon = 100.0 * n_within_horizon / n_cells if n_cells else 0.0
    n_no_future_event = int((days_to_next < 0).sum())

    n_tickers = panel["ticker"].nunique()
    tickers_with_events = sum(1 for v in event_di_per_ticker.values() if v)
    tickers_no_events = [t for t, v in event_di_per_ticker.items() if not v]

    Path(args.coverage_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.coverage_out, "w") as f:
        f.write("# Phase 5b C: earnings calendar coverage\n\n")
        f.write(f"Calendar source: yfinance get_earnings_dates\n\n")
        f.write(f"- Calendar rows: {len(cal)}\n")
        f.write(f"- Tickers with at least one event in window: "
                f"{tickers_with_events} / {n_tickers}\n")
        f.write(f"- Tickers with NO events: {len(tickers_no_events)}\n")
        f.write(f"- Panel cells with next event within 90 trading days: "
                f"{n_within_90}/{n_cells} ({pct_within_90:.2f}%)\n")
        f.write(f"- Panel cells with next event within {HORIZON_TRADING_DAYS} "
                f"trading days (catalyst signal active): "
                f"{n_within_horizon}/{n_cells} ({pct_within_horizon:.2f}%)\n")
        f.write(f"- Panel cells with no future event in calendar: "
                f"{n_no_future_event}/{n_cells} "
                f"({100.0 * n_no_future_event / n_cells:.2f}%)\n\n")
        threshold_pct = 95.0
        if pct_within_90 < threshold_pct:
            f.write(f"## EDGAR fallback verdict\n\n")
            f.write(f"Coverage at the cell level is {pct_within_90:.2f}%, below "
                    f"the {threshold_pct}% threshold. EDGAR 8-K Item 2.02 fallback "
                    f"recommended.\n")
        else:
            f.write(f"## EDGAR fallback verdict\n\n")
            f.write(f"Coverage at the cell level is {pct_within_90:.2f}%, above "
                    f"the {threshold_pct}% threshold. EDGAR fallback NOT triggered.\n")
        if tickers_no_events:
            f.write(f"\n## Tickers with no events (first 50)\n\n")
            for t in tickers_no_events[:50]:
                f.write(f"- {t}\n")
    print(f"[catalyst build] wrote {args.coverage_out}", flush=True)
    print(f"[catalyst build] coverage: {pct_within_90:.2f}% cells with event "
          f"within 90 trading days; {pct_within_horizon:.2f}% in 10-day horizon",
          flush=True)

    if args.update_panel:
        # Overwrite the three catalyst columns in panel_features.parquet
        panel_orig = pd.read_parquet(args.panel)
        panel_orig["date"] = pd.to_datetime(panel_orig["date"])
        merge_cols = ["date", "ticker", "days_to_next_catalyst_sin",
                       "days_to_next_catalyst_cos", "catalyst_type_id"]
        merged = panel_orig.drop(columns=[
            "days_to_next_catalyst_sin", "days_to_next_catalyst_cos",
            "catalyst_type_id",
        ]).merge(out_df[merge_cols], on=["date", "ticker"], how="left")
        # Cells without an event in calendar inherit zeros from the merge.
        for c in ("days_to_next_catalyst_sin", "days_to_next_catalyst_cos"):
            merged[c] = merged[c].fillna(0.0).astype(np.float32)
        merged["catalyst_type_id"] = merged["catalyst_type_id"].fillna(0).astype(np.int64)
        # Restore original column order
        original_cols = list(panel_orig.columns)
        merged = merged[original_cols]
        merged.to_parquet(args.panel, index=False)
        print(f"[catalyst build] updated {args.panel}; new catalyst column "
              f"non-zero counts: "
              f"sin={int((merged['days_to_next_catalyst_sin'] != 0).sum())}, "
              f"cos={int((merged['days_to_next_catalyst_cos'] != 0).sum())}, "
              f"type={int((merged['catalyst_type_id'] != 0).sum())}", flush=True)


if __name__ == "__main__":
    main()
