"""Continuous-time event stream for the MTGN biotech universe.

Merges three event sources into a single timestamp-sorted stream consumed
by the TGN substrate:

    1. Price events: one per ticker per trading day. Payload includes
       log-return, log-volume, realized volatility.
    2. StockTwits events: one per biotech ticker mention; payload includes
       user-declared sentiment (1.0 / -1.0 / null) and message length.
       Post-day timestamping is used (the public corpus gives date-only
       for symbols/; feature_wo_messages/ has ISO timestamps but is not
       used in Phase 1 Scenario C).
    3. Catalyst events: FDA actions, trial readouts, M&A, earnings.
       Placeholder sources; not wired to live APIs in Phase 1 beyond a
       CSV stub for future expansion.

Output: a `pandas.DataFrame` with columns
    [event_id, ts, event_type, src_ticker, dst_ticker, payload_json]

where event_type in {'price', 'st', 'catalyst'}. For events that describe
only a single stock (price, unary catalysts), dst_ticker equals src_ticker.
For co-mention StockTwits events spanning multiple tickers, one row per
unordered pair is emitted.

The stream is then sorted by `ts` and used by downstream memory update
and graph attention modules. Following Rossi et al. 2020 ordering
discipline, the training loop is responsible for ensuring memory updates
derived from event t do not influence predictions for event t.

Phase 1 scope: price + StockTwits. Reddit is Phase 2.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_PRICES = Path("data/raw/prices_universe.parquet")
DEFAULT_STOCKTWITS = Path("data/raw/stocktwits/symbols.parquet")
DEFAULT_UNIVERSE = Path("data/raw/biotech_universe_v1.csv")


@dataclass
class EventStreamConfig:
    price_parquet: Path = DEFAULT_PRICES
    stocktwits_parquet: Path = DEFAULT_STOCKTWITS
    universe_csv: Path = DEFAULT_UNIVERSE
    start_date: str = "2020-01-01"
    end_date: str = "2024-12-31"
    vol_window: int = 20   # for realized vol feature
    include_costmention: bool = True
    co_mention_window_days: int = 1


def _load_active_tickers(universe_csv: Path) -> set[str]:
    df = pd.read_csv(universe_csv)
    if "status" in df.columns:
        df = df[df["status"] == "active"]
    return set(df["ticker"].dropna().astype(str).str.upper())


def _build_price_events(cfg: EventStreamConfig, tickers: set[str]) -> pd.DataFrame:
    prices = pd.read_parquet(cfg.price_parquet)
    prices["ticker"] = prices["ticker"].astype(str).str.upper()
    prices = prices[prices["ticker"].isin(tickers)]
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices[(prices["date"] >= cfg.start_date) & (prices["date"] <= cfg.end_date)]
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)

    # derived features per ticker
    out_frames = []
    for t, sub in prices.groupby("ticker", sort=False):
        sub = sub.copy()
        sub["log_return"] = np.log(sub["close"]).diff()
        sub["log_volume"] = np.log(sub["volume"].replace(0, np.nan))
        sub["realized_vol"] = sub["log_return"].rolling(cfg.vol_window, min_periods=5).std()
        out_frames.append(sub)
    prices = pd.concat(out_frames, ignore_index=True).dropna(subset=["log_return"])

    events = pd.DataFrame(
        {
            "ts": prices["date"],
            "event_type": "price",
            "src_ticker": prices["ticker"],
            "dst_ticker": prices["ticker"],
        }
    )
    payload = prices[["log_return", "log_volume", "realized_vol", "close", "volume"]]
    events["payload_json"] = payload.apply(
        lambda row: json.dumps({k: (None if pd.isna(v) else float(v)) for k, v in row.items()}),
        axis=1,
    )
    return events


def _build_stocktwits_events(cfg: EventStreamConfig, tickers: set[str]) -> pd.DataFrame:
    st = pd.read_parquet(
        cfg.stocktwits_parquet,
        columns=["message_id", "symbol", "created_at", "sentiment"],
    )
    st["ticker"] = st["symbol"].astype(str).str.upper()
    st = st[st["ticker"].isin(tickers)]
    st["ts"] = pd.to_datetime(st["created_at"])
    st = st[(st["ts"] >= cfg.start_date) & (st["ts"] <= cfg.end_date)]

    # Unary StockTwits events: one per (message, ticker) row.
    unary = pd.DataFrame(
        {
            "ts": st["ts"],
            "event_type": "st",
            "src_ticker": st["ticker"],
            "dst_ticker": st["ticker"],
            "payload_json": st["sentiment"].apply(
                lambda v: json.dumps({"sentiment": (None if pd.isna(v) else float(v))})
            ),
        }
    )
    events = [unary]

    # Co-mention edges: messages with multiple ticker rows sharing message_id.
    if cfg.include_costmention:
        by_msg = st.groupby("message_id")["ticker"].apply(list)
        multi = by_msg[by_msg.apply(len) >= 2]
        if not multi.empty:
            # Vectorize: explode pairs per message
            pairs = []
            ts_lookup = st.drop_duplicates("message_id").set_index("message_id")["ts"]
            for mid, ts_list in multi.items():
                uniq = sorted(set(t for t in ts_list if t in tickers))
                if len(uniq) < 2:
                    continue
                ts_val = ts_lookup.loc[mid]
                for i in range(len(uniq)):
                    for j in range(i + 1, len(uniq)):
                        pairs.append((ts_val, uniq[i], uniq[j]))
            if pairs:
                co = pd.DataFrame(pairs, columns=["ts", "src_ticker", "dst_ticker"])
                co["event_type"] = "st_comention"
                co["payload_json"] = "{}"
                events.append(co)

    return pd.concat(events, ignore_index=True)


def build_event_stream(cfg: EventStreamConfig | None = None) -> pd.DataFrame:
    cfg = cfg or EventStreamConfig()
    tickers = _load_active_tickers(cfg.universe_csv)
    print(f"universe size: {len(tickers)} tickers")

    price_events = _build_price_events(cfg, tickers)
    print(f"price events:       {len(price_events):,}")

    st_events = _build_stocktwits_events(cfg, tickers)
    print(f"stocktwits events:  {len(st_events):,}")

    all_events = pd.concat([price_events, st_events], ignore_index=True)
    all_events = all_events.sort_values("ts", kind="mergesort").reset_index(drop=True)
    all_events["event_id"] = np.arange(len(all_events))
    return all_events[
        ["event_id", "ts", "event_type", "src_ticker", "dst_ticker", "payload_json"]
    ]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--out", type=Path, default=Path("data/processed/event_stream.parquet"))
    args = parser.parse_args()

    cfg = EventStreamConfig(start_date=args.start, end_date=args.end)
    events = build_event_stream(cfg)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    events.to_parquet(args.out, index=False)
    print(f"Wrote {args.out}: {len(events):,} events over {events['ts'].min()} to {events['ts'].max()}")


if __name__ == "__main__":
    main()
