"""Ticker-level IPO analogue memory bank for OW-epiSTAR.

Stores per-(day, ticker) memory entries for ticker-day pairs that are
within the spec's max IPO age (default 756 trading days, the first 3
trading years of any ticker). At query time, retrieves the top-M
analogues whose IPO-context key is most similar to the query ticker's
current key.

Differs from the day-level EpisodeMemoryBank in two ways:
    1. Memory items are per-(day, ticker), not per-day.
    2. The retrieval is ticker-specific: each query (t, i) gets its own
       set of M analogues, queried with ticker i's per-(t, i) key.

Leakage rule (Section 10.4 of the spec): for query (t, i), memory
entry (s, j) is valid only if:
    - s + horizon + embargo < t  (with horizon=5, embargo=5, so s < t-10)
    - s is in the training fold (paper-safe setting)
    - the memory entry's age is within max_age_days
    - listed_mask[s, j] = label_mask[s, j] = True
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn


@dataclass
class IPOMemoryConfig:
    """Hyperparameters for the IPO analogue memory bank."""

    top_m: int = 8
    horizon_days: int = 5
    embargo_days: int = 5
    max_age_days: int = 756
    random_retrieval: bool = False


class IPOAnalogueMemoryBank(nn.Module):
    """Ticker-level memory bank with leakage-safe retrieval.

    Holds per-(day, ticker) keys and values for all eligible entries
    in the panel (filtered to training-fold days at populate time).
    Each query is one (day, ticker); the bank returns top-M analogues
    by cosine similarity over standardised keys.
    """

    def __init__(
        self, cfg: IPOMemoryConfig, key_dim: int, value_dim: int
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.key_dim = key_dim
        self.value_dim = value_dim

        self.register_buffer("key_mean", torch.zeros(key_dim))
        self.register_buffer("key_std", torch.ones(key_dim))
        self.register_buffer("mem_keys", torch.zeros(0, key_dim))
        self.register_buffer("mem_values", torch.zeros(0, value_dim))
        self.register_buffer("mem_day_idx", torch.zeros(0, dtype=torch.long))
        self.register_buffer("mem_ticker_idx", torch.zeros(0, dtype=torch.long))

    def populate(
        self,
        keys: np.ndarray,
        values: np.ndarray,
        day_indices: np.ndarray,
        ticker_indices: np.ndarray,
        train_day_indices: np.ndarray,
    ) -> None:
        """Load the bank with eligible (day, ticker) entries.

        Args:
            keys: [M, key_dim] raw IPO keys.
            values: [M, value_dim] IPO values.
            day_indices: [M] integer day indices.
            ticker_indices: [M] integer ticker indices.
            train_day_indices: indices used to compute standardisation.
        """
        assert keys.shape[0] == values.shape[0] == day_indices.shape[0] == ticker_indices.shape[0]
        # Standardise on the subset of memory entries whose day is in
        # the training-day index list.
        train_mask = np.isin(day_indices, train_day_indices)
        train_keys = keys[train_mask] if train_mask.any() else keys
        mean = train_keys.mean(axis=0)
        std = train_keys.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        self.key_mean.data = torch.from_numpy(mean.astype(np.float32))
        self.key_std.data = torch.from_numpy(std.astype(np.float32))

        keys_std = (keys - mean) / std
        # Restrict the stored memory to training-fold entries (paper-safe).
        sel = train_mask
        self.mem_keys.data = torch.from_numpy(keys_std[sel].astype(np.float32))
        self.mem_values.data = torch.from_numpy(values[sel].astype(np.float32))
        self.mem_day_idx.data = torch.from_numpy(day_indices[sel].astype(np.int64))
        self.mem_ticker_idx.data = torch.from_numpy(ticker_indices[sel].astype(np.int64))

    def standardize_query(self, raw_key: Tensor) -> Tensor:
        """Standardise one query key using stored training stats."""
        return (raw_key - self.key_mean) / self.key_std

    def batch_retrieve(
        self, query_raw_keys: Tensor, query_day_idx: int
    ) -> dict[str, Tensor]:
        """Batched retrieval for many query tickers on the same day.

        Args:
            query_raw_keys: [B, key_dim] raw IPO keys for B query tickers.
            query_day_idx: shared integer day index (cutoff applies once).

        Returns:
            Dict with [B, M] tensors (top1_sim, sim_entropy, day_indices,
            ticker_indices, similarities) and a [B, M, value_dim] values
            tensor. The leakage cutoff is applied once for the whole day.
        """
        cfg = self.cfg
        cutoff = query_day_idx - cfg.horizon_days - cfg.embargo_days
        device = query_raw_keys.device
        b = query_raw_keys.shape[0]
        mem_day_idx = self.mem_day_idx.to(device)
        eligible = mem_day_idx < cutoff
        n_eligible = int(eligible.sum())
        if n_eligible == 0:
            empty_v = torch.zeros(b, cfg.top_m, self.value_dim, device=device)
            empty_s = torch.zeros(b, cfg.top_m, device=device)
            empty_d = torch.full((b, cfg.top_m), -1, dtype=torch.long, device=device)
            return {
                "values": empty_v, "similarities": empty_s,
                "day_indices": empty_d, "ticker_indices": empty_d.clone(),
                "top1_sim": torch.zeros(b, device=device),
                "sim_entropy": torch.zeros(b, device=device),
            }

        q_std = (query_raw_keys - self.key_mean) / self.key_std    # [B, K]
        elig_keys = self.mem_keys[eligible]                          # [E, K]
        q_norm = q_std / (q_std.norm(p=2, dim=-1, keepdim=True) + 1e-8)
        k_norm = elig_keys / (elig_keys.norm(p=2, dim=-1, keepdim=True) + 1e-8)
        sims = q_norm @ k_norm.T                                     # [B, E]

        m = min(cfg.top_m, n_eligible)
        if cfg.random_retrieval:
            idx = torch.stack([torch.randperm(n_eligible, device=device)[:m] for _ in range(b)])
        else:
            idx = torch.topk(sims, k=m, dim=-1, largest=True).indices  # [B, M]
        eligible_indices = torch.nonzero(eligible).squeeze(-1)
        chosen = eligible_indices[idx]                                # [B, M]
        values = self.mem_values[chosen]                              # [B, M, V]
        similarities = torch.gather(sims, 1, idx)                     # [B, M]
        day_indices = self.mem_day_idx[chosen]                        # [B, M]
        ticker_indices = self.mem_ticker_idx[chosen]                  # [B, M]

        if m < cfg.top_m:
            pad = cfg.top_m - m
            values = torch.cat([values, torch.zeros(b, pad, self.value_dim, device=device)], dim=1)
            similarities = torch.cat([similarities, torch.zeros(b, pad, device=device)], dim=1)
            day_indices = torch.cat([day_indices, torch.full((b, pad), -1, dtype=torch.long, device=device)], dim=1)
            ticker_indices = torch.cat([ticker_indices, torch.full((b, pad), -1, dtype=torch.long, device=device)], dim=1)

        soft = torch.softmax(similarities, dim=-1)
        entropy = -(soft * torch.log(soft.clamp(min=1e-8))).sum(dim=-1)  # [B]
        return {
            "values": values, "similarities": similarities,
            "day_indices": day_indices, "ticker_indices": ticker_indices,
            "top1_sim": similarities[:, 0], "sim_entropy": entropy,
        }

    def retrieve(
        self, query_raw_key: Tensor, query_day_idx: int
    ) -> dict[str, Tensor]:
        """Retrieve top-M leakage-safe IPO analogues for one (t, i)."""
        cfg = self.cfg
        cutoff = query_day_idx - cfg.horizon_days - cfg.embargo_days
        device = query_raw_key.device
        mem_day_idx = self.mem_day_idx.to(device)
        eligible = mem_day_idx < cutoff
        if eligible.sum() == 0:
            empty_v = torch.zeros(cfg.top_m, self.value_dim, device=device)
            empty_s = torch.zeros(cfg.top_m, device=device)
            empty_d = torch.full((cfg.top_m,), -1, dtype=torch.long, device=device)
            return {
                "values": empty_v,
                "similarities": empty_s,
                "day_indices": empty_d,
                "ticker_indices": empty_d.clone(),
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
        ticker_indices = self.mem_ticker_idx[chosen]

        if m < cfg.top_m:
            pad = cfg.top_m - m
            values = torch.cat([values, torch.zeros(pad, self.value_dim, device=device)], dim=0)
            similarities = torch.cat([similarities, torch.zeros(pad, device=device)])
            day_indices = torch.cat([day_indices, torch.full((pad,), -1, dtype=torch.long, device=device)])
            ticker_indices = torch.cat([ticker_indices, torch.full((pad,), -1, dtype=torch.long, device=device)])

        soft = torch.softmax(similarities, dim=0)
        entropy = -(soft * torch.log(soft.clamp(min=1e-8))).sum()
        return {
            "values": values,
            "similarities": similarities,
            "day_indices": day_indices,
            "ticker_indices": ticker_indices,
            "top1_sim": similarities[0],
            "sim_entropy": entropy,
        }


__all__ = ["IPOAnalogueMemoryBank", "IPOMemoryConfig"]
