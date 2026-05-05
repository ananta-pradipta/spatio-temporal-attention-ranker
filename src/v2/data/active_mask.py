"""Active mask helpers with optional 21-day IPO blackout.

The original SBP active mask was derived from the model-ready panel
(``panel_to_tensors`` -> ``mask``), which already requires 20 days of
return history before a ticker becomes active. The spec's revision asks
for an explicit blackout flag so the rule is auditable in configs and
the rebuilt-panel control can be compared against it.

This module exposes:

    - ``apply_ipo_blackout(active_mask, age_trading_days, days)``
      AND-masks ``active_mask`` with ``age_trading_days >= days``.
    - ``build_blackout_active_mask(panel_mask, tradable_mask, days)``
      computes age from ``tradable_mask`` and applies the blackout to
      ``panel_mask``. Returns a boolean array shaped like the inputs.

Theoretical anchoring (per the SBP revision spec):
    - Loughran and Ritter (2004, Financial Management) on day-1
      underpricing pops.
    - Bradley, Jordan, Ritter (2003, Journal of Finance) on
      quiet-period expiration and analyst coverage initiation effects.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.v2.data.minimal_masks import compute_age_from_tradable


@dataclass
class IPOBlackoutConfig:
    """Hyperparameters for the IPO blackout."""

    use_ipo_blackout: bool = True
    ipo_blackout_days: int = 21


def apply_ipo_blackout(
    active_mask: np.ndarray, age_trading_days: np.ndarray, days: int = 21
) -> np.ndarray:
    """Return ``active_mask AND (age >= days)``.

    A ticker enters the blackout-protected mask only after it has
    accumulated ``days`` cumulative tradable days. Ages of 0 (ticker
    never seen as tradable) remain masked out.
    """
    blackout_pass = age_trading_days >= days
    return active_mask & blackout_pass


def build_blackout_active_mask(
    panel_mask: np.ndarray,
    tradable_mask: np.ndarray,
    cfg: IPOBlackoutConfig | None = None,
) -> tuple[np.ndarray, dict]:
    """Compute the blackout-protected active mask.

    Args:
        panel_mask: [T, N] bool, the model-ready panel's active mask
            (output of ``panel_to_tensors``).
        tradable_mask: [T, N] bool, the raw tradability mask.
        cfg: blackout configuration; if None, uses defaults
            (use_ipo_blackout=True, ipo_blackout_days=21).

    Returns:
        active_mask: [T, N] bool, with the blackout AND'd in if enabled.
        diag: dict with cells_before, cells_after, cells_dropped (for
            logging and unit-test asserts).
    """
    cfg = cfg or IPOBlackoutConfig()
    age = compute_age_from_tradable(tradable_mask)
    cells_before = int(panel_mask.sum())
    if cfg.use_ipo_blackout:
        out = apply_ipo_blackout(panel_mask, age, cfg.ipo_blackout_days)
    else:
        out = panel_mask.copy()
    cells_after = int(out.sum())
    return out, {
        "cells_before": cells_before,
        "cells_after": cells_after,
        "cells_dropped": cells_before - cells_after,
        "use_ipo_blackout": bool(cfg.use_ipo_blackout),
        "ipo_blackout_days": int(cfg.ipo_blackout_days),
    }


__all__ = [
    "IPOBlackoutConfig",
    "apply_ipo_blackout",
    "build_blackout_active_mask",
]
