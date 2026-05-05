"""Build MTGN edges with mechanistic content (not just price correlation).

Signal audit (2026-04-13) showed that a 60-day Pearson correlation graph
over our biotech universe carries no forward-return signal; neighbor-
return averaging actively hurts predictions. Replacing with edges that
capture real cross-ticker relationships.

Three edge sources, all already on disk:

  1. Trial co-participation (data/processed/catalyst_trials.parquet):
     edge (i, j) if tickers i and j both appear as sponsor / collaborator
     on the same ClinicalTrials.gov NCT. Captures biotech partnerships
     and mechanistic overlap.

  2. StockTwits co-mention (data/raw/stocktwits/symbols.parquet):
     edge (i, j) if tickers i and j co-occur in the same message's
     symbol_list in the train period. Captures market-attention
     coupling that correlation misses.

  3. Sector / industry (data/processed/ticker_company.parquet):
     edge (i, j) if tickers i and j share an industry label (e.g.,
     Biotechnology, Drug Manufacturers - General, Diagnostics &
     Research). Provides a backbone of mechanistic grouping.

Output: edge_index np.ndarray [2, E] and edge_weight np.ndarray [E].
Edge weights are the normalized co-occurrence count (trial / co-mention)
or 1.0 (sector). When multiple sources flag the same pair, weights sum.

This replaces src.mtgn.training.graph_builder.build_correlation_edges
as the default graph source for MTGNFull.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class EdgeBuildConfig:
    trials_parquet: Path = Path("data/processed/catalyst_trials.parquet")
    stocktwits_symbols: Path = Path("data/raw/stocktwits/symbols.parquet")
    stocktwits_msg_info: Path = Path("data/raw/stocktwits/msg_info.parquet")
    ticker_company: Path = Path("data/processed/ticker_company.parquet")
    train_start: str = "2020-01-01"
    train_end: str = "2022-06-30"
    min_comention_count: int = 5
    max_degree: int | None = 40
    trial_weight: float = 1.0
    comention_weight: float = 0.5
    sector_weight: float = 0.25


def _add_edge(edges: dict[tuple[int, int], float], a: int, b: int, w: float) -> None:
    if a == b:
        return
    key = (min(a, b), max(a, b))
    edges[key] = edges.get(key, 0.0) + w


def _trial_edges(
    cfg: EdgeBuildConfig, ticker_to_idx: dict[str, int]
) -> dict[tuple[int, int], float]:
    if not cfg.trials_parquet.exists():
        return {}
    t = pd.read_parquet(cfg.trials_parquet)
    t["ticker"] = t["ticker"].astype(str).str.upper()
    t = t[t["ticker"].isin(ticker_to_idx)]
    # Two tickers sharing any NCT get an edge weighted by shared-trial count.
    pair_counts: dict[tuple[int, int], float] = {}
    by_nct = t.groupby("nct_id")["ticker"].apply(set)
    for tickers in by_nct:
        tickers = [t for t in tickers if t in ticker_to_idx]
        if len(tickers) < 2:
            continue
        idx = sorted({ticker_to_idx[tk] for tk in tickers})
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                _add_edge(pair_counts, idx[i], idx[j], cfg.trial_weight)
    return pair_counts


def _comention_edges(
    cfg: EdgeBuildConfig, ticker_to_idx: dict[str, int]
) -> dict[tuple[int, int], float]:
    """Co-mention edges from StockTwits: two tickers appear together in a message.

    The symbols.parquet file has 1 row per (message_id, symbol) but in this
    dataset each message has exactly ONE symbol (the channel the message was
    posted to). Real co-mentions come from the message TEXT referencing
    other tickers via $cashtags. We reconstruct them from msg_info.parquet's
    `important_words` list (already lowercased, stop-filtered): any token
    that matches a known ticker (case-insensitive) is a cashtag candidate.

    For each message we form the set {channel_ticker} U {cashtags in text}
    and emit pairs. This does NOT miss the channel/cashtag pair (which
    is often the intended co-mention).
    """
    if not cfg.stocktwits_symbols.exists() or not cfg.stocktwits_msg_info.exists():
        return {}
    tickers_upper = set(ticker_to_idx.keys())
    # Exclude text-based cashtag matches for biotech tickers that collide with
    # common English words (EDIT, FOLD, RARE): "rare opportunity" would get
    # misread as ticker RARE. These tickers are still matched when they appear
    # as the message's channel symbol (channel is unambiguous).
    COMMON_WORD_TICKERS = {"EDIT", "FOLD", "RARE"}
    tickers_lower = {t.lower() for t in tickers_upper if t not in COMMON_WORD_TICKERS}

    # Channel per message + timestamp, filtered to train window
    sym = pd.read_parquet(
        cfg.stocktwits_symbols, columns=["message_id", "symbol", "created_at"]
    )
    sym["symbol"] = sym["symbol"].astype(str).str.upper()
    sym["created_at"] = pd.to_datetime(sym["created_at"])
    sym = sym[(sym["created_at"] >= cfg.train_start) & (sym["created_at"] <= cfg.train_end)]
    sym = sym[sym["symbol"].isin(tickers_upper)]
    if sym.empty:
        return {}

    # Important words (pre-tokenized) per message
    info = pd.read_parquet(cfg.stocktwits_msg_info, columns=["message_id", "important_words"])
    merged = sym.merge(info, how="inner", on="message_id")

    # important_words is stored as the string-repr of a Python list
    # (e.g. "['com', 'mnkd', 'gonna']"), not as a real list. Parse it once.
    import ast
    def _parse(s):
        if not isinstance(s, str) or not s:
            return []
        try:
            v = ast.literal_eval(s)
            return v if isinstance(v, list) else []
        except (ValueError, SyntaxError):
            return []

    pairs_agg: dict[tuple[int, int], int] = {}
    for channel, raw in merged[["symbol", "important_words"]].itertuples(index=False, name=None):
        words = _parse(raw)
        if not words:
            continue
        cashtags = {w.upper() for w in words if isinstance(w, str) and w.lower() in tickers_lower}
        cashtags.add(channel)
        if len(cashtags) < 2:
            continue
        idx = sorted({ticker_to_idx[t] for t in cashtags if t in ticker_to_idx})
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                key = (idx[i], idx[j])
                pairs_agg[key] = pairs_agg.get(key, 0) + 1

    out: dict[tuple[int, int], float] = {}
    for key, cnt in pairs_agg.items():
        if cnt < cfg.min_comention_count:
            continue
        out[key] = cfg.comention_weight * np.log1p(cnt)
    return out


def _sector_edges(
    cfg: EdgeBuildConfig, ticker_to_idx: dict[str, int]
) -> dict[tuple[int, int], float]:
    if not cfg.ticker_company.exists():
        return {}
    tc = pd.read_parquet(cfg.ticker_company)
    tc["ticker"] = tc["ticker"].astype(str).str.upper()
    tc = tc[tc["ticker"].isin(ticker_to_idx) & tc["industry"].notna()]
    out: dict[tuple[int, int], float] = {}
    for industry, sub in tc.groupby("industry"):
        idx = sorted({ticker_to_idx[t] for t in sub["ticker"]})
        if len(idx) < 2:
            continue
        # Keep all industry groups; rely on `max_degree` capping at the
        # merge step to prevent per-node edge explosion for the large
        # yfinance Biotechnology industry (~150+ tickers in our universe).
        # Down-weight for large industries to acknowledge lower per-pair
        # information vs a small specialty industry.
        w = cfg.sector_weight / max(1.0, np.log1p(len(idx)))
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                _add_edge(out, idx[i], idx[j], w)
    return out


def build_mechanistic_edges_per_relation(
    tickers: list[str], cfg: EdgeBuildConfig | None = None,
    require_nonempty: bool = True,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return {relation_name: (edge_index [2,E], edge_weight [E])} for each of
    trial / comention / sector. Each edge set is undirected (both directions
    materialized). Degree capping is applied per-relation."""
    cfg = cfg or EdgeBuildConfig()
    ticker_to_idx = {t: i for i, t in enumerate(tickers)}

    missing = []
    for label, path in (
        ("trials_parquet", cfg.trials_parquet),
        ("stocktwits_symbols", cfg.stocktwits_symbols),
        ("ticker_company", cfg.ticker_company),
    ):
        if not path.exists():
            missing.append(f"{label}={path}")
    if missing and require_nonempty:
        raise FileNotFoundError(
            "build_mechanistic_edges_per_relation: missing inputs: "
            + ", ".join(missing)
        )

    raw = {
        "trial":     _trial_edges(cfg, ticker_to_idx),
        "comention": _comention_edges(cfg, ticker_to_idx),
        "sector":    _sector_edges(cfg, ticker_to_idx),
    }
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, pairs in raw.items():
        if not pairs:
            out[name] = (np.zeros((2, 0), dtype=np.int64), np.zeros(0, dtype=np.float32))
            continue
        # Per-relation degree cap
        merged = dict(pairs)
        if cfg.max_degree is not None:
            by_node: dict[int, list[tuple[int, float]]] = {}
            for (a, b), w in merged.items():
                by_node.setdefault(a, []).append((b, w))
                by_node.setdefault(b, []).append((a, w))
            keep: set[tuple[int, int]] = set()
            for n, nbrs in by_node.items():
                nbrs.sort(key=lambda x: -x[1])
                for other, _ in nbrs[: cfg.max_degree]:
                    keep.add((min(n, other), max(n, other)))
            merged = {k: merged[k] for k in keep if k in merged}
        src, dst, w = [], [], []
        for (a, b), wt in merged.items():
            src.extend([a, b]); dst.extend([b, a]); w.extend([wt, wt])
        out[name] = (
            np.asarray([src, dst], dtype=np.int64),
            np.asarray(w, dtype=np.float32),
        )
    total = sum(v[0].shape[1] for v in out.values())
    print("build_mechanistic_edges_per_relation: "
          + ", ".join(f"{k}={v[0].shape[1]}" for k, v in out.items())
          + f"  (total {total})")
    if require_nonempty and total == 0:
        raise ValueError("all three relations produced 0 edges")
    return out


def build_mechanistic_edges(
    tickers: list[str], cfg: EdgeBuildConfig | None = None,
    require_nonempty: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    cfg = cfg or EdgeBuildConfig()
    ticker_to_idx = {t: i for i, t in enumerate(tickers)}

    # Sanity check: all source files should exist. Silent fallback to
    # empty edge sets has burned us once already (2026-04-13 Wulver runs
    # with edges=0 because catalyst_trials / ticker_company were missing).
    missing = []
    for label, path in (
        ("trials_parquet", cfg.trials_parquet),
        ("stocktwits_symbols", cfg.stocktwits_symbols),
        ("ticker_company", cfg.ticker_company),
    ):
        if not path.exists():
            missing.append(f"{label}={path}")
    if missing and require_nonempty:
        raise FileNotFoundError(
            "build_mechanistic_edges: missing input files: "
            + ", ".join(missing)
            + ". Set require_nonempty=False to allow partial inputs."
        )

    # Merge edges from all sources (sum weights)
    merged: dict[tuple[int, int], float] = {}
    sources = {
        "trial": _trial_edges(cfg, ticker_to_idx),
        "comention": _comention_edges(cfg, ticker_to_idx),
        "sector": _sector_edges(cfg, ticker_to_idx),
    }
    print(
        "build_mechanistic_edges sources: "
        + ", ".join(f"{k}={len(v)}" for k, v in sources.items())
    )
    for source in sources.values():
        for k, v in source.items():
            merged[k] = merged.get(k, 0.0) + v
    if require_nonempty and not merged:
        raise ValueError(
            "build_mechanistic_edges produced 0 edges. Check input files and "
            "train_start/train_end window."
        )
    # Cap per-node degree if requested
    if cfg.max_degree is not None:
        by_node: dict[int, list[tuple[int, float]]] = {}
        for (a, b), w in merged.items():
            by_node.setdefault(a, []).append((b, w))
            by_node.setdefault(b, []).append((a, w))
        keep: set[tuple[int, int]] = set()
        for n, nbrs in by_node.items():
            nbrs.sort(key=lambda x: -x[1])
            for other, _ in nbrs[: cfg.max_degree]:
                keep.add((min(n, other), max(n, other)))
        merged = {k: merged[k] for k in keep if k in merged}
    # Materialize to arrays (undirected -> both directions)
    src, dst, w = [], [], []
    for (a, b), wt in merged.items():
        src.extend([a, b])
        dst.extend([b, a])
        w.extend([wt, wt])
    edge_index = np.asarray([src, dst], dtype=np.int64)
    edge_weight = np.asarray(w, dtype=np.float32)
    return edge_index, edge_weight


__all__ = ["build_mechanistic_edges", "EdgeBuildConfig"]
