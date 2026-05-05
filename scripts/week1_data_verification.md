# Week 1 Data Verification Checklist (MTGN Phase 1)

Source: `drafts/memorizing-tgn-social-signal-data-sources.md` Sections 5 and 6.

**Rule:** resolve all six items before writing any data-loader code. The pipeline you build depends on what is actually accessible. Document each outcome in the results table at the bottom; the table becomes a citable artifact for the paper's data section and protects against "we built it, then discovered the data was not there."

## Items

### 1. StockTwits API tier and rate limits
- [ ] Sign up for a StockTwits developer account.
- [ ] Document the current quota for ticker-tagged message retrieval.
- [ ] Determine whether an academic tier still exists.
- [ ] Decision: if too restrictive, lean on the public S3 corpus (`s3://stocktwits-nyu/dataset/v1/data/csv`) and academic datasets (ACL18, BIGDATA22, STOCKNET).

### 2. Pushshift status
- [ ] Run a known-good query against the current Pushshift endpoint for a biotech-relevant sub.
- [ ] Document whether it returns results and over what date range.

### 3. Hugging Face Reddit dumps for biotech-relevant subs
- [ ] Search Hugging Face for dumps of r/biotechplays, r/Biotechnology, r/wallstreetbets.
- [ ] For each dump found, record: coverage window, post count, license terms.

### 4. PRAW access to r/biotechplays (quarantined sub)
- [ ] Authenticate PRAW against your Reddit account.
- [ ] Opt in to the r/biotechplays quarantine notice via browser if prompted.
- [ ] Attempt to fetch recent posts via PRAW.
- [ ] Confirm whether quarantined-sub API access works for your account.

### 5. r/biotech identity disambiguation
- [ ] Visit `reddit.com/r/biotech` directly.
- [ ] Determine whether it is the trading-focused sub or the career / student sub (or both).
- [ ] Sample 30 to 50 recent posts and note the dominant content type.

### 6. Existing academic datasets
- [ ] Check ACL18, BIGDATA22, STOCKNET on Hugging Face and academic repositories.
- [ ] For each: current accessibility, biotech ticker coverage, specific time window covered.

## Results table

Fill in after each check. Date and paste outcomes here before running any loader code.

| # | Item | Status | Outcome / notes | Date |
|---|------|--------|------------------|------|
| 1 | StockTwits API tier | | | |
| 2 | Pushshift | | | |
| 3 | HF Reddit dumps | | | |
| 4 | PRAW quarantine access | | | |
| 5 | r/biotech identity | | | |
| 6 | Academic datasets | | | |

## Scenario mapping

Based on outcomes, pick the scenario that matches (memo Section 9):

- **Scenario A (best case):** everything works. Execute dual-source plan as written.
- **Scenario B:** r/biotechplays inaccessible, others work. Promote r/Biotechnology to primary biotech sub, raise DD filter threshold.
- **Scenario C:** Reddit historical access broken entirely. Defer Reddit to Phase 2. Phase 1 = StockTwits-only. Reframe Phase 1 paper as "social-sentiment-augmented MTGN."
- **Scenario D:** StockTwits API too restrictive. Rely on academic datasets plus the public S3 corpus; lean on Reddit for recent period and live signal.

Architectural claims (dual attention, cross-entity retrieval, risk head) are unchanged across scenarios.
