"""IPO analogue memory bank for OW-epiSTAR v1.

Thin wrapper around the existing ``IPOAnalogueMemoryBank`` (from
``src.v2.model.ipo_memory``). The v1 spec asks for a separate file
``ipo_analogue_memory.py`` and a richer 22-dim retrieval key that
carries macro context (VIX z-score, XBI realized vol, average pairwise
correlation) alongside the existing per-ticker fundamentals/social/age
fields.

The key column list is exposed as ``IPO_ANALOGUE_KEY_COLS`` so the
trainer can build keys consistently and persist them in the JSON
output. Only the column list and a max-age default change vs the v0
implementation; the retrieval logic, leakage rule, and standardisation
are unchanged.
"""
from __future__ import annotations

from src.v2.model.ipo_memory import IPOAnalogueMemoryBank, IPOMemoryConfig


IPO_ANALOGUE_KEY_COLS = [
    "age_trading_days",
    "log1p_age_trading_days",
    "age_bucket_0_20",
    "age_bucket_21_60",
    "age_bucket_61_252",
    "age_bucket_253_plus",
    "history_valid_ratio_20d",
    "history_valid_ratio_60d",
    "log_market_cap",
    "cash_runway_q",
    "cash_to_mc",
    "rd_intensity",
    "st_volume_24h",
    "st_bullish_ratio",
    "st_labeled_ratio",
    "realized_vol_20d",
    "realized_vol_60d",
    "log_return_5d",
    "log_return_20d",
    "vix_z",
    "xbi_rv_20d",
    "avg_pairwise_corr_60d",
]


__all__ = [
    "IPOAnalogueMemoryBank",
    "IPOMemoryConfig",
    "IPO_ANALOGUE_KEY_COLS",
]
