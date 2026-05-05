"""Export the MTGN data inventory as a multi-sheet Excel workbook.

Source of truth: drafts/memorizing-tgn-research-context.md,
drafts/memorizing-tgn-salience-gating-policy.md,
drafts/memorizing-tgn-social-signal-data-sources.md,
drafts/preliminaries.md, configs/mtgn/phase1.yaml.

Writes docs/data_inventory.xlsx with one sheet per data category plus a
summary sheet.

Usage:
    python3 scripts/export_data_inventory.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


OUT_PATH = Path("docs/data_inventory.xlsx")


UNIVERSE = pd.DataFrame(
    [
        ["Biotech ticker list", "XBI (SPDR S&P Biotech ETF) + NBI (Nasdaq Biotech Index) holdings",
         "Static snapshot; 38 tickers already in data/raw/", "Phase 1",
         "src/mtgn/data/xbi_proxy_tickers.txt", "Survivorship-biased if left static"],
        ["Point-in-time XBI/NBI membership", "WRDS/CRSP (via NJIT)",
         "Pending WRDS access confirmation (see scripts/wrds_access_email.md)",
         "Phase 1", "WRDS/CRSP Stock File + Index Constituents",
         "Required for survivorship-bias-free ranking eval; fallback = static + declared limitation"],
    ],
    columns=["Item", "Source", "Status", "Phase", "Access path", "Notes"],
)


PRICES = pd.DataFrame(
    [
        ["Daily OHLCV per ticker", "Yahoo Finance via yfinance", "Daily",
         "log-return, log-volume, realized volatility, momentum, VWAP deviation",
         "data/raw/*.csv already populated"],
        ["Market capitalization", "Yahoo Finance (optional Compustat)", "Daily/quarterly",
         "market_cap_bucket feature", "Yahoo primary; Compustat via WRDS optional"],
        ["Book-to-market ratio", "Compustat via WRDS", "Quarterly",
         "Fundamental feature", "Optional; blocked on WRDS"],
        ["Pipeline stage composition", "Drugs@FDA + 10-K filings", "Static/annual",
         "Counts of drugs in Phase I/II/III/Approved per company", "Hand-curated"],
    ],
    columns=["Item", "Source", "Frequency", "Features used", "Notes"],
)


VOL = pd.DataFrame(
    [
        ["VIX", "CBOE Volatility Index (SPX 30d implied vol)", "Yahoo Finance (^VIX)",
         "Daily", "Free, no credentials"],
        ["VXN", "CBOE Nasdaq-100 Volatility Index", "Yahoo Finance (^VXN)",
         "Daily", "Free; relevant because XBI and NBI are Nasdaq-listed"],
        ["VVIX", "CBOE VIX-of-VIX", "Yahoo Finance (^VVIX)",
         "Daily", "Free; leading indicator of regime transitions"],
    ],
    columns=["Item", "Source", "Access", "Frequency", "Notes"],
)
VOL_NOTE = "Broadcast as g(t) in R^3 to every node at every timestep. Fetch script: src/mtgn/data/volatility_indices.py"


STOCKTWITS = pd.DataFrame(
    [
        ["Historical posts 2008-2022", "s3://stocktwits-nyu/dataset/v1/data/csv",
         "Public S3, --no-sign-request / anon=True, ~550M posts",
         "Li, Al Ansari, Kaufman (2025), JQD 5",
         "src/mtgn/data/download_stocktwits.py",
         "All six StockTwits features below"],
        ["Fresh posts 2023-present", "StockTwits REST API",
         "Tier pending week-1 verification", "N/A",
         "Not implemented", "Same features for recent period"],
        ["Academic fallback: ACL18", "Hugging Face / arXiv mirrors",
         "Contains StockTwits + Twitter mixed data",
         "Xu & Cohen (2018)", "Not implemented", "Bridge if API tier insufficient"],
        ["Academic fallback: BIGDATA22", "Hugging Face / authors",
         "Tweet-based stock movement benchmark", "Soun et al. (2022)",
         "Not implemented", "Bridge if API tier insufficient"],
        ["Academic fallback: STOCKNET", "GitHub / authors",
         "Includes Twitter sentiment and price data", "Xu & Cohen (2018)",
         "Not implemented", "Bridge if API tier insufficient"],
    ],
    columns=["Item", "Source", "Access", "Citation", "Implementation", "Feature mapping"],
)

ST_FEATURES = pd.DataFrame(
    [
        ["st_volume_24h", "Daily message count for ticker", "Raw attention signal"],
        ["st_volume_change_30d", "v(t) / rolling_mean(v, 30, end=t-1)", "Attention-spike (feeds salience-gating Trigger 2)"],
        ["st_bullish_ratio", "bullish / (bullish + bearish)", "User-declared sentiment direction"],
        ["st_bullish_ratio_weighted", "Weighted by user follower count", "Quality-adjusted sentiment; falls back to unweighted if follower count unavailable"],
        ["st_sentiment_dispersion", "std of sentiment across users in 24h", "Disagreement / uncertainty proxy"],
        ["st_labeled_ratio", "labeled / total posts", "Confidence indicator for the ratio features"],
    ],
    columns=["Feature", "Definition", "Rationale"],
)


REDDIT_SUBS = pd.DataFrame(
    [
        ["r/Biotechnology", "~200K", "Reliable; academic/industry framing",
         "Primary fallback if r/biotechplays blocked", "Primary"],
        ["r/biotechplays", "~20K", "QUARANTINED by Reddit admins",
         "Primary target when PRAW access verified; otherwise dropped", "Primary, verify W1"],
        ["r/biotech", "?", "Identity ambiguous (trading vs career/student)",
         "Include if trading-focused after W1 visit", "Conditional primary"],
        ["r/SqueezePlays", "smaller", "Niche, short-squeeze biotech setups",
         "Low base rate, high signal when triggered", "Secondary"],
        ["r/wallstreetbets", "~17M", "Mostly noise; catches meme-stock episodes",
         "Include for completeness", "Secondary"],
        ["r/stocks", "~7-8M", "General equity; occasional biotech",
         "Coverage check", "Secondary"],
        ["r/investing", "~3M", "Long-term investing, low biotech share",
         "Mostly noise", "Secondary"],
    ],
    columns=["Subreddit", "Members", "Current status", "Role in plan", "Tier"],
)

REDDIT_FEATURES = pd.DataFrame(
    [
        ["reddit_post_count_24h", "Total posts mentioning ticker (across selected subs)", "Raw discussion volume"],
        ["reddit_comment_count_24h", "Total comments on ticker threads", "Engagement"],
        ["reddit_dd_count_24h", "Due-diligence post count (body>500, score>5, primary subs)", "Long-form analysis volume"],
        ["reddit_dd_count_7d", "DD post count, trailing 7 days", "Slow-moving fundamental signal"],
        ["reddit_avg_post_length", "Mean chars of ticker posts", "Substantive vs casual proxy"],
        ["reddit_score_weighted_count", "Posts weighted by upvote score", "Community-validated attention"],
        ["reddit_trusted_user_count_7d", "Posts by users with karma>500, age>1yr in relevant subs", "Quality-floor signal"],
        ["reddit_subreddit_breadth", "Distinct subs discussing ticker in 24h", "Discussion has spread beyond core sub"],
    ],
    columns=["Feature", "Definition", "Rationale"],
)

REDDIT_ACCESS = pd.DataFrame(
    [
        ["Pushshift", "Historical dumps via Pushshift endpoint",
         "Intermittent since 2023", "Week-1 verify"],
        ["Hugging Face Reddit dumps", "Community-curated HF datasets",
         "Partial coverage for biotech-relevant subs", "Week-1 verify"],
        ["Authenticated PRAW forward collection", "Official Reddit API with user token",
         "Only ~4 weeks of fresh data before QE deadline; insufficient alone", "Fallback"],
        ["HODL (Rahman et al., ICDEW 2023) pipeline advice", "NJIT FinTech Lab precedent",
         "Email Muntasir Rahman (now Rutgers postdoc)", "scripts/rahman_outreach_email.md"],
    ],
    columns=["Access path", "Description", "Status", "Action"],
)


SENTIMENT = pd.DataFrame(
    [
        ["FinBERT (yiyanghkust/finbert-tone)", "Hugging Face",
         "Reddit post/comment scoring for reddit_* aggregates",
         "Phase 1", "Drop-in"],
        ["StockTwits user tags (bullish/bearish)", "Native StockTwits field",
         "Direct use; 30-50% of posts tagged at post time", "Phase 1", "No NLP pipeline needed for v1"],
        ["Biotech-domain sentiment fine-tune", "Future work",
         "Specialized vocabulary (FDA, oncology, mechanism)",
         "Phase 2+", "If FinBERT coverage limits are hit"],
    ],
    columns=["Model", "Source", "Use", "Phase", "Notes"],
)


CATALYSTS = pd.DataFrame(
    [
        ["FDA actions, PDUFA dates", "FDA Calendar / Drugs@FDA (open data)", "Free", "Salience-gate Trigger 3 + entry metadata"],
        ["Clinical trial primary completion / results", "ClinicalTrials.gov API", "Free", "Salience-gate Trigger 3 + entry metadata + Phase 2 edges"],
        ["M&A, partnerships", "SEC EDGAR 8-K filings", "Free", "Salience-gate Trigger 3 + entry metadata"],
        ["Earnings calendar", "Yahoo Finance (or Zacks)", "Free", "Salience-gate Trigger 3"],
    ],
    columns=["Event type", "Source", "Cost", "Use"],
)


EDGES = pd.DataFrame(
    [
        ["r_corr (price return correlation)", "Daily log-return Pearson, 60-day window, thresholded",
         "Weekly", "Phase 1"],
        ["r_mention (social co-mention)",
         "StockTwits cashtag co-occurrence + Reddit ticker co-occurrence, 3-day window, source-tagged",
         "Daily", "Phase 1"],
        ["r_target (shared drug target)", "DrugBank Jaccard on target sets", "Quarterly", "Phase 2"],
        ["r_trial (clinical trial co-participation)", "ClinicalTrials.gov shared trials/indications",
         "Quarterly", "Phase 2"],
        ["r_fda (shared FDA regulatory pathway)", "Concurrent PDUFA dates",
         "Event-driven", "Phase 2"],
    ],
    columns=["Edge type", "Construction", "Update frequency", "Phase"],
)


BIOMED_KG = pd.DataFrame(
    [
        ["DrugBank", "Drug-target, mechanism-of-action", "Academic license", "Phase 2+"],
        ["ClinicalTrials.gov", "Trial IDs, sponsors, indications, status", "Free, API", "Phase 2+"],
        ["FDA Drugs@FDA", "Approval history, regulatory pathway", "Free", "Phase 2+"],
        ["ChEMBL", "Bioactivity data, target enrichment", "Free", "Phase 2+ (optional)"],
        ["UniProt", "Target-level protein annotation", "Free", "Phase 2+ (optional)"],
    ],
    columns=["Dataset", "Content", "Access", "Phase"],
)


EVAL = pd.DataFrame(
    [
        ["Cross-sector robustness", "Tech or energy sector equivalent", "Same pipeline with swapped universe", "Phase 3"],
        ["GARCH-EVT baseline", "Parametric VaR / Expected Shortfall", "Computed from price series", "Phase 3"],
        ["Quantile-LSTM baseline", "Neural quantile regression", "Trained on same features", "Phase 3"],
        ["Kupiec / Christoffersen / Diebold-Mariano tests", "Unconditional and conditional coverage", "Standard risk backtests", "Phase 3"],
    ],
    columns=["Item", "Description", "Source/computation", "Phase"],
)


STATUS = pd.DataFrame(
    [
        ["On disk", "data/raw/*.csv (38 biotech ticker OHLCV + close_panel + volume_panel)",
         "Ready"],
        ["On disk", "data/processed/biotech_snapshots.pt (pilot v2 graph snapshots)", "Ready"],
        ["On disk", "src/mtgn/data/xbi_proxy_tickers.txt (38 ticker seed list)", "Ready"],
        ["Script ready", "VIX/VXN/VVIX via src/mtgn/data/volatility_indices.py", "Runnable"],
        ["Script ready", "StockTwits subset via src/mtgn/data/download_stocktwits.py", "Runnable after smoke test"],
        ["Scaffold", "Reddit via src/mtgn/data/reddit_loader.py", "Stubs raise until W1 verification"],
        ["Blocked (external)", "WRDS/CRSP point-in-time membership",
         "Pending NJIT librarian; scripts/wrds_access_email.md drafted"],
        ["Blocked (external)", "Reddit historical", "Pending W1 verification"],
        ["Blocked (external)", "StockTwits API tier", "Pending W1 verification"],
        ["Blocked (external)", "HODL advice", "Pending outreach; scripts/rahman_outreach_email.md drafted"],
    ],
    columns=["Status", "Item", "Action"],
)


SHEETS: list[tuple[str, pd.DataFrame, str]] = [
    ("Universe", UNIVERSE, "Stock universe (nodes)"),
    ("Prices and volume", PRICES, "Price, volume, fundamentals"),
    ("Volatility indices", VOL, VOL_NOTE),
    ("StockTwits sources", STOCKTWITS, "StockTwits data sources (primary trading sentiment)"),
    ("StockTwits features", ST_FEATURES, "Per-stock per-day StockTwits features (memo Section 2.1)"),
    ("Reddit subreddits", REDDIT_SUBS, "Candidate subs; verify statuses in W1"),
    ("Reddit features", REDDIT_FEATURES, "Per-stock per-day Reddit features (memo Section 2.2)"),
    ("Reddit access paths", REDDIT_ACCESS, "Reddit historical-data access options"),
    ("Sentiment models", SENTIMENT, "Sentiment/NLP models"),
    ("Catalyst calendars", CATALYSTS, "Catalyst event sources for salience gating Trigger 3"),
    ("Edge construction", EDGES, "Graph edge types and their construction"),
    ("Biomedical KG", BIOMED_KG, "Phase 2+ biomedical knowledge graph sources"),
    ("Evaluation and baselines", EVAL, "Phase 3+ evaluation datasets and baselines"),
    ("Status summary", STATUS, "What is on disk, what is ready to fetch, what is blocked"),
]


def write(out_path: Path = OUT_PATH) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        # TOC sheet
        toc = pd.DataFrame(
            [(i + 1, name, note) for i, (name, _, note) in enumerate(SHEETS)],
            columns=["#", "Sheet", "Description"],
        )
        toc.to_excel(writer, sheet_name="TOC", index=False)

        for name, df, _note in SHEETS:
            df.to_excel(writer, sheet_name=name[:31], index=False)

        # Basic column widths
        for ws_name in writer.sheets:
            ws = writer.sheets[ws_name]
            for col_idx, col_cells in enumerate(ws.columns, start=1):
                max_len = max((len(str(c.value)) if c.value is not None else 0) for c in col_cells)
                ws.column_dimensions[ws.cell(row=1, column=col_idx).column_letter].width = min(max(max_len + 2, 10), 80)

    return out_path


def main() -> None:
    path = write()
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
