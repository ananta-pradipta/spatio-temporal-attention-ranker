"""Build the static relation graph used by the RSR baseline (per-fold, leakage-free).

RSR (Feng et al., TOIS 2019) requires a fixed adjacency
``A in {0,1}^{N x N}`` derived from industry / sector / wikidata
relations. The original paper used Wikidata first-order and
second-order edges plus GICS sector buckets for NASDAQ/NYSE.

Our biotech-244 panel has no NAICS or SIC code populated in the
universe CSV. The strongest classification we have on disk is
Yahoo Finance's ``industry`` (from
``data/processed/ticker_company.parquet``) plus historical
shares_outstanding from EDGAR (``data/raw/fundamentals_edgar.parquet``)
combined with point-in-time close prices from
``data/raw/prices_universe.parquet``.

Leakage discipline (CRITICAL):

  Earlier versions of this builder used ``current_market_cap`` from
  ``data/raw/fundamentals_quarterly.parquet``. That column is a
  2025/2026 snapshot (each ticker has exactly one distinct value
  across all quarters), so it baked post-test size information into
  a graph used to predict 2015-2022 returns and inflated test IC.

  This version computes log-mcap quintile bins ONLY from the
  fold's ``train_end_date``:
    - Latest EDGAR ``shares`` row whose ``filed_date <= train_end_date``.
    - Latest close price on or before ``train_end_date``.
    - mcap = close * shares.
  The industry classification still comes from Yahoo's stable
  industry labels. This is a documented small leakage source: a
  ticker's industry label is current as of data download time, but
  industries rarely change for biotech tickers within our window.

Construction (per fold):

  1. Industry buckets (Yahoo industry).
  2. Size bands (within Biotechnology only): five log-market-cap
     quintile buckets across tickers with a defined point-in-time
     mcap at ``train_end_date``. Tickers without an mcap reading
     by that date (e.g. pre-IPO) get ``size_bin = -1`` and are
     ISOLATED from other no-mcap tickers within Biotechnology
     (no edges among the size_bin = -1 cohort). They will only
     connect via their industry to non-Biotech industry peers if
     such peers exist; in the biotech-244 panel they end up
     effectively isolated within Biotechnology.

     Rationale (no-clique guard, 2026-05-05): the previous version
     placed all size_bin = -1 tickers into a single shared "no-mcap"
     bucket and let them connect to each other. On fold 1, that
     bucket held 131 tickers (post-train-end IPOs that had not
     filed EDGAR shares by 2018-12-31) and produced ~91% of all
     biotech-biotech edges, dominating the relation graph and
     potentially inflating test IC. This disambiguation test
     isolates the no-mcap tickers so the giant clique cannot
     drive ranking performance.

Adjacency rule (directed, non-self-loop):

    A[i, j] = 1   iff  industry(i) == industry(j)  AND
                       (industry != 'Biotechnology' OR
                        (size_bin(i) >= 0 AND size_bin(i) == size_bin(j)))

Outputs:
    ``data/processed/rsr_relation_graph_fold{F}.pt`` containing:
        {
            "A":        torch.uint8 (N, N) directed adjacency,
            "tickers":  list[str] in the same row/col order,
            "industry": list[str | None] per ticker,
            "size_bin": list[int] per ticker (-1 if mcap missing),
            "config":   dict with construction parameters,
        }
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path("/home/apradipta/phd-research")
TICKER_COMPANY_PATH = REPO_ROOT / "data/processed/ticker_company.parquet"
EDGAR_FUNDAMENTALS_PATH = REPO_ROOT / "data/raw/fundamentals_edgar.parquet"
PRICES_PATH = REPO_ROOT / "data/raw/prices_universe.parquet"


@dataclass
class RelationGraphConfig:
    """Parameters controlling the static relation graph."""

    n_size_bins: int = 5
    biotech_industry_label: str = "Biotechnology"
    train_end_date: pd.Timestamp | None = None  # required at build time
    fold: int | None = None  # required at build time, used in filename


def _load_pit_market_caps(train_end_date: pd.Timestamp) -> pd.DataFrame:
    """Compute point-in-time market caps as of ``train_end_date``.

    For each ticker:
      - Take the latest EDGAR ``shares`` row whose ``filed_date`` is on
        or before ``train_end_date``. This avoids using a disclosure
        that had not been filed yet.
      - Take the latest available close price on or before
        ``train_end_date`` from ``prices_universe.parquet``.
      - mcap = close * shares.

    Returns a DataFrame with columns ``["ticker", "mcap"]``. Tickers
    that lack either a filed-by-date EDGAR shares row or a price on or
    before ``train_end_date`` are simply omitted (they will get
    size_bin = -1 downstream).
    """
    edgar = pd.read_parquet(EDGAR_FUNDAMENTALS_PATH)
    edgar["ticker"] = edgar["ticker"].str.upper()
    # Only keep rows with a non-null shares value AND a filed_date.
    edgar = edgar.dropna(subset=["shares", "filed_date"]).copy()
    edgar["filed_date"] = pd.to_datetime(edgar["filed_date"])
    # Keep only filings disclosed on or before the fold's train end.
    edgar = edgar[edgar["filed_date"] <= train_end_date]
    # For each ticker, take the most recent filed_date row.
    edgar = edgar.sort_values(["ticker", "filed_date"])
    shares = (
        edgar.groupby("ticker")
        .tail(1)[["ticker", "filed_date", "quarter_end", "shares"]]
        .reset_index(drop=True)
    )

    prices = pd.read_parquet(PRICES_PATH, columns=["ticker", "date", "close"])
    prices["ticker"] = prices["ticker"].str.upper()
    prices = prices[prices["date"] <= train_end_date]
    prices = prices.sort_values(["ticker", "date"])
    last_close = (
        prices.groupby("ticker")
        .tail(1)[["ticker", "date", "close"]]
        .rename(columns={"date": "price_date"})
        .reset_index(drop=True)
    )

    merged = shares.merge(last_close, on="ticker", how="inner")
    merged["mcap"] = merged["close"].astype(np.float64) * merged["shares"].astype(
        np.float64
    )
    merged = merged[merged["mcap"] > 0].copy()
    return merged[["ticker", "mcap"]]


def _load_classifications(cfg: RelationGraphConfig) -> pd.DataFrame:
    """Merge Yahoo industry with point-in-time market caps for the fold."""
    if cfg.train_end_date is None:
        raise ValueError("RelationGraphConfig.train_end_date must be set.")
    tc = pd.read_parquet(TICKER_COMPANY_PATH)
    tc["ticker"] = tc["ticker"].str.upper()
    tc = tc[tc["ticker"] != "-"].copy().reset_index(drop=True)

    mcap_df = _load_pit_market_caps(cfg.train_end_date)
    out = tc.merge(mcap_df, on="ticker", how="left")
    return out[["ticker", "industry", "mcap"]]


def _compute_size_bins(mcap: np.ndarray, n_bins: int) -> np.ndarray:
    """Quantile-bin log10(market_cap) across tickers with a defined value.

    Returns a (N,) int array; -1 marks tickers missing market cap.
    """
    log_mcap = np.full_like(mcap, np.nan, dtype=np.float64)
    pos = np.where(np.isfinite(mcap) & (mcap > 0))
    log_mcap[pos] = np.log10(mcap[pos])
    defined = log_mcap[~np.isnan(log_mcap)]
    if defined.size == 0:
        return np.full(mcap.shape, -1, dtype=np.int32)
    qs = np.quantile(defined, np.linspace(0.0, 1.0, n_bins + 1))
    n = len(mcap)
    out = np.full(n, -1, dtype=np.int32)
    for i in range(n):
        lm = log_mcap[i]
        if np.isnan(lm):
            continue
        for b in range(n_bins):
            if lm >= qs[b] and lm <= qs[b + 1] + 1e-12:
                out[i] = b
                break
    return out


def build_relation_graph(cfg: RelationGraphConfig) -> dict:
    """Construct the static directed adjacency for the universe (per fold).

    Tickers without a valid point-in-time market cap as of
    ``cfg.train_end_date`` get ``size_bin = -1``. Within
    Biotechnology, these no-mcap tickers receive NO edges among
    themselves (no-clique guard, 2026-05-05) so the giant
    "we-don't-know-size-yet" cohort cannot dominate the graph.
    """
    if cfg.train_end_date is None or cfg.fold is None:
        raise ValueError(
            "RelationGraphConfig.train_end_date and fold are both required."
        )

    df = _load_classifications(cfg)
    tickers = df["ticker"].tolist()
    industries = [
        x if (isinstance(x, str) and x) else None for x in df["industry"].tolist()
    ]
    mcap = df["mcap"].to_numpy(dtype=np.float64)
    size_bin = _compute_size_bins(mcap, cfg.n_size_bins)

    n = len(tickers)
    A = np.zeros((n, n), dtype=np.uint8)
    for i in range(n):
        ii = industries[i]
        if ii is None:
            continue
        for j in range(n):
            if i == j:
                continue
            ij = industries[j]
            if ij is None or ii != ij:
                continue
            if ii == cfg.biotech_industry_label:
                # Within Biotechnology: same size bucket, but NEVER
                # link two no-mcap tickers (size_bin = -1) to each
                # other. This isolates the post-train-end IPO cohort
                # whose shares had not been filed by EDGAR yet, so
                # that the giant no-mcap clique cannot dominate the
                # adjacency. Tickers with size_bin >= 0 still link
                # if they share both industry and size band.
                if size_bin[i] < 0 or size_bin[j] < 0:
                    continue
                if size_bin[i] == size_bin[j]:
                    A[i, j] = 1
            else:
                A[i, j] = 1

    n_edges = int(A.sum())
    density = float(n_edges / (n * (n - 1))) if n > 1 else 0.0
    deg = A.sum(axis=1)
    n_with_mcap = int((size_bin >= 0).sum())

    return {
        "A": torch.from_numpy(A),
        "tickers": tickers,
        "industry": industries,
        "size_bin": size_bin.tolist(),
        "config": {
            "n_size_bins": cfg.n_size_bins,
            "biotech_industry_label": cfg.biotech_industry_label,
            "source": "yahoo_industry+pit_mcap_quintile",
            "fold": cfg.fold,
            "train_end_date": str(cfg.train_end_date.date()),
            "mcap_source": "edgar_shares*price_close",
        },
        "stats": {
            "n_nodes": n,
            "n_with_mcap": n_with_mcap,
            "n_no_mcap_bucket": int((size_bin == -1).sum()),
            "n_edges_directed": n_edges,
            "density": density,
            "mean_out_degree": float(deg.mean()),
            "median_out_degree": float(np.median(deg)),
            "isolated_nodes": int((deg == 0).sum()),
        },
    }


def output_path_for_fold(fold: int) -> Path:
    return REPO_ROOT / f"data/processed/rsr_relation_graph_fold{fold}.pt"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fold",
        type=int,
        choices=[1, 2, 3],
        required=True,
        help="Fold id; controls output filename (rsr_relation_graph_fold{F}.pt).",
    )
    parser.add_argument(
        "--train-end-date",
        type=str,
        required=True,
        help="Inclusive cutoff for shares filed_date and price date (YYYY-MM-DD).",
    )
    parser.add_argument("--n-size-bins", type=int, default=5)
    args = parser.parse_args()

    cfg = RelationGraphConfig(
        n_size_bins=args.n_size_bins,
        train_end_date=pd.Timestamp(args.train_end_date),
        fold=args.fold,
    )
    payload = build_relation_graph(cfg)
    out_path = output_path_for_fold(cfg.fold)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    s = payload["stats"]
    print(f"[rsr-graph] fold={cfg.fold} train_end={cfg.train_end_date.date()}")
    print(f"[rsr-graph] saved -> {out_path}")
    print(f"  N nodes:        {s['n_nodes']}")
    print(f"  With mcap:      {s['n_with_mcap']}")
    print(f"  No-mcap bucket: {s['n_no_mcap_bucket']}")
    print(f"  Edges (dir):    {s['n_edges_directed']}")
    print(f"  Density:        {s['density']:.4f}")
    print(f"  Mean degree:    {s['mean_out_degree']:.1f}")
    print(f"  Isolated:       {s['isolated_nodes']}")


if __name__ == "__main__":
    main()
