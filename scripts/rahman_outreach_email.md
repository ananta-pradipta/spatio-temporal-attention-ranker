# Outreach Email: Muntasir Rahman (HODL Paper, Reddit Financial Sentiment Pipeline)

**Context for yourself:** Muntasir Rahman co-authored HODL (Rahman, Uddin, Wang, ICDEW 2023) at the NJIT FinTech Lab, which used Reddit-based financial sentiment. He is now a postdoc at Rutgers. A 30-minute conversation could save 5 to 10 days of pipeline-building on the Reddit half of the dual-source plan. Ask Prof. Wei for his current email if not known.

**Before sending:** read the HODL paper carefully (methodology section will document the Reddit sourcing approach used), so the conversation can focus on what is not in the paper (Pushshift status at time of collection, quality-filter thresholds, whether the raw or processed dataset is shareable).

---

**Subject:** Advice on Reddit financial sentiment pipeline for a biotech T-GNN project (NJIT FinTech Lab follow-on)

Dear Muntasir,

I am Ananta Pradipta, a PhD student in Data Science at NJIT, working under Prof. Zhi Wei. My dissertation develops a temporal graph neural network, Memorizing TGN, for biotech stock ranking using social media signals, with biotech (XBI / NBI) as the primary testbed. I am scoping the social-signal pipeline for the qualifying exam paper (Phase 1 of four) and am building the Reddit half of a dual-source (StockTwits + Reddit) plan.

I came across HODL (Rahman, Uddin, Wang, ICDEW 2023), which is the closest NJIT-internal precedent I have found for Reddit-based financial sentiment. If you have 30 minutes, I would very much appreciate your advice on the Reddit pipeline. Specifically:

1. Which subreddits were most productive for financial content, and did you encounter the r/biotechplays quarantine or a similar access restriction?
2. What quality filtering did you use to separate long-form analytical posts ("due diligence") from short comments ("chatter")?
3. Was Pushshift working during your collection window? If not, what historical-data path did you end up using?
4. Where are your processed datasets stored, and would you be willing to share the collection scripts or the processed data itself?
5. Any trusted-user identification heuristics you found reliable (karma thresholds, account-age filters, post-history checks)?

I would be happy to discuss synchronously by Zoom or async by email, whichever works for you. If timing is tight before a qualifying exam deadline, even a brief list of "what I wish I had known" would be very useful.

Thank you, and congratulations on the Rutgers role.

Best regards,
Ananta Pradipta
PhD Student, Data Science, NJIT
Advisor: Prof. Zhi Wei
