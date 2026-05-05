"""Rate-shock memory bank for DOW-epiSTAR v2 (spec Section F).

Day-level memory bank keyed on rate, credit, and biotech-duration
shock structure rather than generic volatility. Stores one entry per
training-fold day. At query time, retrieves top M_rate=8 historical
days whose rate-shock key is most similar (cosine over standardised
keys) subject to the leakage rule s + horizon + embargo < t.

Key (17 dims, in spec order):
    delta_10y_5d, delta_10y_20d
    delta_2y_5d, delta_2y_20d
    term_10y_2y, term_10y_3m
    hy_spread (z), delta_hy_spread_5d, delta_hy_spread_20d
    xbi_ret_5d, xbi_ret_20d
    xbi_rv_20d, xbi_rv_60d
    qqq_ret_20d, spy_ret_20d
    avg_pairwise_corr_60d, cross_sectional_dispersion

Value: concat(rate_shock_key, day-level cs summaries). Duration
exposure aggregates per spec are deferred to v2.3 because they
require running the (still-training) DurationExposureEncoder over
the whole panel at memory build time.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn


# v2.3 patch C: explicit 17-dim key list (15 macro + 2 cs scalars,
# no longer appended at runtime). The trainer constructs the key array
# in this exact column order.
RATE_SHOCK_KEY_COLS = [
    "delta_10y_5d", "delta_10y_20d",
    "delta_2y_5d", "delta_2y_20d",
    "term_10y_2y", "term_10y_3m",
    "hy_spread", "delta_hy_spread_5d", "delta_hy_spread_20d",
    "xbi_ret_5d", "xbi_ret_20d",
    "xbi_rv_20d", "xbi_rv_60d",
    "qqq_ret_20d", "spy_ret_20d",
    "avg_pairwise_corr_60d", "cross_sectional_dispersion",
]

# Daily-CS-summary additions to the value (mean, std of features).
RATE_VALUE_CS_SUMMARY_COLS = [
    "log_return_mean", "log_return_std",
    "log_return_5d_mean", "log_return_5d_std",
    "rv_20d_mean", "rv_20d_std",
    "active_count_norm",
]

# v2.3 patch D: duration-distribution summaries appended to rate value.
RATE_VALUE_DURATION_SUMMARY_COLS = [
    "duration_norm_mean", "duration_norm_std",
    "duration_norm_top_decile_mean", "duration_norm_bottom_decile_mean",
    "rate_beta_mean", "rate_beta_std",
    "credit_beta_mean", "credit_beta_std",
    "cash_runway_mean", "cash_runway_bottom_decile_mean",
    "cash_to_mc_mean", "rd_intensity_mean",
]


@dataclass
class RateShockMemoryConfig:
    """Hyperparameters for the rate-shock memory bank."""

    top_m: int = 8
    horizon_days: int = 5
    embargo_days: int = 5
    random_retrieval: bool = False


class RateShockMemoryBank(nn.Module):
    """Day-level memory bank keyed on rate-shock structure."""

    def __init__(
        self, cfg: RateShockMemoryConfig, key_dim: int, value_dim: int,
    ) -> None:
        super().__init__()
        # v2.3 patch C: explicit dimension assertion.
        assert key_dim == len(RATE_SHOCK_KEY_COLS), (
            f"key_dim {key_dim} must equal len(RATE_SHOCK_KEY_COLS)="
            f"{len(RATE_SHOCK_KEY_COLS)}"
        )
        self.cfg = cfg
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.register_buffer("key_mean", torch.zeros(key_dim))
        self.register_buffer("key_std", torch.ones(key_dim))
        self.register_buffer("mem_keys", torch.zeros(0, key_dim))
        self.register_buffer("mem_values", torch.zeros(0, value_dim))
        self.register_buffer("mem_day_idx", torch.zeros(0, dtype=torch.long))

    def populate(
        self,
        keys: np.ndarray,
        values: np.ndarray,
        day_indices: np.ndarray,
        train_day_indices: np.ndarray,
    ) -> None:
        """Load the bank with training-fold day-level entries.

        Args:
            keys: [T, key_dim] raw rate-shock keys for every panel day.
            values: [T, value_dim] daily values.
            day_indices: [T] integer indices.
            train_day_indices: indices used to compute standardisation
                stats AND the only days that enter the bank.
        """
        assert keys.shape[0] == values.shape[0] == day_indices.shape[0]
        assert keys.shape[1] == self.key_dim
        assert values.shape[1] == self.value_dim

        train_keys = keys[train_day_indices]
        mean = train_keys.mean(axis=0)
        std = train_keys.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        self.key_mean.data = torch.from_numpy(mean.astype(np.float32))
        self.key_std.data = torch.from_numpy(std.astype(np.float32))

        keys_std = (keys - mean) / std
        # Restrict bank to training-fold entries (paper-safe).
        train_mask = np.isin(day_indices, train_day_indices)
        sel = train_mask
        self.mem_keys.data = torch.from_numpy(keys_std[sel].astype(np.float32))
        self.mem_values.data = torch.from_numpy(values[sel].astype(np.float32))
        self.mem_day_idx.data = torch.from_numpy(day_indices[sel].astype(np.int64))

    def standardize_query(self, raw_key: Tensor) -> Tensor:
        """Standardise a query key using bank training stats."""
        return (raw_key - self.key_mean) / self.key_std

    def retrieve(
        self, query_raw_key: Tensor, query_day_idx: int,
    ) -> dict[str, Tensor]:
        """Top-M leakage-safe retrieval for one query day."""
        cfg = self.cfg
        cutoff = query_day_idx - cfg.horizon_days - cfg.embargo_days
        device = query_raw_key.device
        mem_day_idx = self.mem_day_idx.to(device)
        eligible = mem_day_idx < cutoff
        n_eligible = int(eligible.sum())
        if n_eligible == 0:
            empty_v = torch.zeros(cfg.top_m, self.value_dim, device=device)
            empty_s = torch.zeros(cfg.top_m, device=device)
            empty_d = torch.full((cfg.top_m,), -1, dtype=torch.long, device=device)
            return {
                "values": empty_v, "similarities": empty_s,
                "day_indices": empty_d,
                "top1_sim": torch.zeros((), device=device),
                "sim_entropy": torch.zeros((), device=device),
            }

        q_std = self.standardize_query(query_raw_key)
        elig_keys = self.mem_keys[eligible]
        q_norm = q_std / (q_std.norm(p=2) + 1e-8)
        k_norm = elig_keys / (elig_keys.norm(p=2, dim=-1, keepdim=True) + 1e-8)
        sims = (k_norm @ q_norm).squeeze(-1)

        m = min(cfg.top_m, sims.shape[0])
        if cfg.random_retrieval:
            idx = torch.randperm(sims.shape[0], device=device)[:m]
        else:
            idx = torch.topk(sims, k=m, largest=True).indices

        eligible_indices = torch.nonzero(eligible).squeeze(-1)
        chosen = eligible_indices[idx]
        values = self.mem_values[chosen]
        similarities = sims[idx]
        day_indices = self.mem_day_idx[chosen]
        if m < cfg.top_m:
            pad = cfg.top_m - m
            values = torch.cat([values, torch.zeros(pad, self.value_dim, device=device)], dim=0)
            similarities = torch.cat([similarities, torch.zeros(pad, device=device)])
            day_indices = torch.cat([day_indices, torch.full((pad,), -1, dtype=torch.long, device=device)])
        soft = torch.softmax(similarities, dim=0)
        entropy = -(soft * torch.log(soft.clamp(min=1e-8))).sum()
        return {
            "values": values, "similarities": similarities,
            "day_indices": day_indices,
            "top1_sim": similarities[0], "sim_entropy": entropy,
        }


__all__ = [
    "RATE_SHOCK_KEY_COLS",
    "RATE_VALUE_CS_SUMMARY_COLS",
    "RATE_VALUE_DURATION_SUMMARY_COLS",
    "RateShockMemoryBank",
    "RateShockMemoryConfig",
]
