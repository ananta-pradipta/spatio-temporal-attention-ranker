"""Reddit loader for MTGN Phase 1 dual-source social signal pipeline.

This module is a Phase 1 SCAFFOLD. Before using it, complete the week-1
verification checklist at scripts/week1_data_verification.md: PRAW access to
quarantined subs, Pushshift availability, and Hugging Face Reddit dump
coverage for biotech-relevant subreddits. The actual data-access path will
depend on those outcomes (see fallback scenarios in
drafts/memorizing-tgn-social-signal-data-sources.md Section 9).

Reddit is the secondary, fundamental / due-diligence source in the dual-source
plan. StockTwits is the primary trading-sentiment source; features from the
two platforms are kept as SEPARATE feature groups in the model input.

Features produced (per stock, per day) follow memo Section 2.2:
    reddit_post_count_24h, reddit_comment_count_24h,
    reddit_dd_count_24h, reddit_dd_count_7d,
    reddit_avg_post_length, reddit_score_weighted_count,
    reddit_trusted_user_count_7d, reddit_subreddit_breadth.

Due-diligence vs chatter split (memo Section 5.6): body length > 500 AND
score > 5 AND posted in a primary biotech sub.

Trusted user set (memo Section 5.7): karma > 500 in primary biotech subs AND
account age > 1 year AND post history dominated by biotech subs.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


PRIMARY_SUBS = ("Biotechnology", "biotechplays", "biotech")
SECONDARY_SUBS = ("SqueezePlays", "wallstreetbets", "stocks")

DD_MIN_BODY_LENGTH = 500
DD_MIN_SCORE = 5

TRUSTED_MIN_KARMA = 500
TRUSTED_MIN_ACCOUNT_AGE_YEARS = 1


@dataclass
class RedditAccess:
    """Resolved Reddit access mode after week-1 verification.

    Attributes:
        has_pushshift: Pushshift endpoint works for historical queries.
        hf_dump_path: Local path to a Hugging Face Reddit dump, if cached.
        praw_quarantine_ok: Authenticated PRAW can fetch quarantined-sub posts.
        forward_only: Only forward PRAW collection is available.
    """
    has_pushshift: bool = False
    hf_dump_path: Path | None = None
    praw_quarantine_ok: bool = False
    forward_only: bool = False


def is_due_diligence(body: str, score: int, subreddit: str) -> bool:
    """Return True if the post meets the memo Section 5.6 DD criteria."""
    return (
        len(body) > DD_MIN_BODY_LENGTH
        and score > DD_MIN_SCORE
        and subreddit.lower() in {s.lower() for s in PRIMARY_SUBS}
    )


def load_historical(
    access: RedditAccess,
    tickers: set[str],
    start_date: str,
    end_date: str,
    out_dir: Path,
) -> Path:
    """Pull historical Reddit posts, filtered to the ticker universe.

    Dispatches based on `access`:
        - Pushshift path, if functional.
        - HF dump path, if available.
        - PRAW forward-only path, if nothing historical works (limited window).

    Returns the path of a parquet file with columns:
        post_id, created_at, subreddit, author, author_karma, tickers,
        title, body, score, num_comments, is_dd.

    This function is a stub. Implementation waits on week-1 verification.
    """
    raise NotImplementedError(
        "Reddit historical loader is blocked on week-1 verification. "
        "See scripts/week1_data_verification.md before implementing."
    )


def compute_daily_features(
    posts_parquet: Path,
    tickers: set[str],
    out_parquet: Path,
) -> Path:
    """Aggregate raw posts into per-stock per-day Reddit feature rows.

    Output columns match memo Section 2.2.
    """
    raise NotImplementedError(
        "Feature aggregation waits on the loader implementation."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers-file", type=Path, required=True)
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--out-dir", type=Path, default=Path("data/raw/reddit"))
    _ = parser.parse_args()
    print(
        "Reddit loader is blocked on week-1 verification. "
        "Run the checklist in scripts/week1_data_verification.md first."
    )


if __name__ == "__main__":
    main()
