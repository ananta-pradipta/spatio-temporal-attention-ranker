"""Salience gate: write policy for the MTGN episodic store.

Implements the four logical-OR triggers from
`drafts/memorizing-tgn-salience-gating-policy.md` §2:

    Trigger 1: return-magnitude      |r_i(t)| > k_sigma * rolling_std
    Trigger 2: attention-spike       st_volume_24h(t) > k_v * v_bar
    Trigger 3: catalyst-event        FDA, trial readout, M&A, earnings
    Trigger 4: memory-delta          per-stock top 5 percent of |delta|

Each trigger tests a condition that can be computed from pre-prediction
information at time t (shifted features, see memo §4 on causality).

Usage:
    gate = SalienceGate(cfg)
    fire, metadata = gate.should_write(
        ticker=i, date=t,
        return_series=returns_i_prior,
        st_volume_series=stocktwits_prior,
        catalyst_events=calendar_t,
        memory_delta=delta_post_update,
        epoch=epoch_idx,
    )
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class GatingConfig:
    k_sigma: float = 2.0
    use_abnormal_return: bool = True
    cap_pct: float = 0.10
    sigma_window: int = 30

    k_volume: float = 3.0
    v_min: int = 10
    volume_window: int = 30

    catalyst_enabled: bool = True
    catalyst_event_types: tuple[str, ...] = (
        "FDA_action", "trial_readout", "M_and_A", "partnership", "earnings",
    )

    delta_percentile: float = 0.95
    delta_warmup_epochs: int = 3
    per_stock_percentile: bool = True


@dataclass
class TriggerResult:
    """Per-event gate evaluation. Used as metadata on stored entries."""
    return_trigger: bool = False
    attention_trigger: bool = False
    catalyst_trigger: bool = False
    delta_trigger: bool = False
    event_type: str | None = None

    @property
    def any(self) -> bool:
        return (
            self.return_trigger
            or self.attention_trigger
            or self.catalyst_trigger
            or self.delta_trigger
        )

    def to_dict(self) -> dict:
        return {
            "return_trigger": self.return_trigger,
            "attention_trigger": self.attention_trigger,
            "catalyst_trigger": self.catalyst_trigger,
            "delta_trigger": self.delta_trigger,
            "event_type": self.event_type,
        }


class SalienceGate:
    """Stateful gate that tracks per-stock rolling baselines and delta history."""

    def __init__(self, cfg: GatingConfig | None = None):
        self.cfg = cfg or GatingConfig()
        self._delta_history: dict[int, list[float]] = {}

    def evaluate(
        self,
        ticker_id: int,
        return_prior: np.ndarray,
        st_volume_prior: np.ndarray,
        return_today: float,
        st_volume_today: float,
        catalyst_event_type: Optional[str],
        memory_delta: Optional[float],
        epoch: int,
    ) -> TriggerResult:
        """Evaluate all four triggers for a single (ticker, time) step.

        All arrays MUST end at t-1 (no look-ahead). `return_today` and
        `st_volume_today` are the current-day values used to test
        against the baselines.
        """
        cfg = self.cfg
        res = TriggerResult(event_type=catalyst_event_type)

        # Trigger 1: return-magnitude
        if return_prior.size >= 5:
            sigma = float(np.nanstd(return_prior[-cfg.sigma_window:]))
            abs_r = abs(return_today)
            res.return_trigger = (
                (abs_r > cfg.k_sigma * max(sigma, 1e-6))
                or (abs_r > cfg.cap_pct)
            )

        # Trigger 2: attention-spike (StockTwits volume in this scenario)
        if st_volume_prior.size >= 5:
            v_bar = float(np.nanmean(st_volume_prior[-cfg.volume_window:]))
            res.attention_trigger = (
                st_volume_today > cfg.k_volume * max(v_bar, 1e-6)
                and st_volume_today > cfg.v_min
            )

        # Trigger 3: catalyst-event
        if cfg.catalyst_enabled and catalyst_event_type in cfg.catalyst_event_types:
            res.catalyst_trigger = True

        # Trigger 4: memory-delta
        if memory_delta is not None and epoch >= cfg.delta_warmup_epochs:
            hist = self._delta_history.setdefault(ticker_id, [])
            if len(hist) >= 30:
                threshold = float(np.quantile(hist, cfg.delta_percentile))
                res.delta_trigger = memory_delta > threshold
            hist.append(memory_delta)
            if len(hist) > 1000:
                del hist[0 : len(hist) - 1000]

        return res


__all__ = ["GatingConfig", "SalienceGate", "TriggerResult"]
