"""Leakage-safe episodic memory bank for epiSTAR.

Stores per-day episode keys and values. Retrieval at query day t returns
the top-M most similar memory days subject to the leakage constraint
    s + horizon + embargo < t,
i.e. with horizon=5 and embargo=5, s < t - 10.

The memory bank can be configured to draw only from the training window
(the recommended paper-safe setting), or from training plus validation
days when explicitly authorized.

Two operating modes:
    raw_key: cosine similarity on standardized raw episode keys (Stage 1).
    learned: cosine similarity on a learned linear projection of the keys
        (Stage 2 ablation; not used in the default v1 implementation).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn


@dataclass
class EpisodeMemoryConfig:
    """Hyperparameters for the episode memory bank.

    Attributes:
        top_m: number of memory episodes to retrieve per query day.
        horizon_days: target prediction horizon (used in leakage rule).
        embargo_days: extra embargo before a day becomes retrievable.
        retrieval_mode: "raw_key" or "learned".
        random_retrieval: if True, retrieve uniformly at random rather
            than by similarity (ablation: tests whether retrieval
            quality matters).
    """

    top_m: int = 8
    horizon_days: int = 5
    embargo_days: int = 5
    retrieval_mode: str = "raw_key"
    random_retrieval: bool = False


class EpisodeMemoryBank(nn.Module):
    """In-memory bank of (key, value) episodes with leakage-safe retrieval.

    The bank is filled from numpy arrays at fold setup and queried per day
    during training and evaluation. Query keys are computed from the same
    feature pipeline as memory keys; both are standardized using statistics
    drawn only from the training-day slice to avoid future-data leakage.
    """

    def __init__(
        self,
        cfg: EpisodeMemoryConfig,
        key_dim: int,
        value_dim: int,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.key_dim = key_dim
        self.value_dim = value_dim

        # Standardization statistics (set at populate time).
        self.register_buffer("key_mean", torch.zeros(key_dim))
        self.register_buffer("key_std", torch.ones(key_dim))

        # Memory storage. Initialized to empty; loaded by populate().
        self.register_buffer("mem_keys", torch.zeros(0, key_dim))
        self.register_buffer("mem_values", torch.zeros(0, value_dim))
        self.register_buffer("mem_day_idx", torch.zeros(0, dtype=torch.long))

        # Learned-key projection (Stage 2). Identity at default config.
        if cfg.retrieval_mode == "learned":
            self.key_projector: nn.Module = nn.Sequential(
                nn.Linear(key_dim, key_dim),
                nn.GELU(),
                nn.Linear(key_dim, key_dim),
            )
            self.query_projector: nn.Module = nn.Sequential(
                nn.Linear(key_dim, key_dim),
                nn.GELU(),
                nn.Linear(key_dim, key_dim),
            )
        else:
            self.key_projector = nn.Identity()
            self.query_projector = nn.Identity()

    def populate(
        self,
        keys: np.ndarray,
        values: np.ndarray,
        day_indices: np.ndarray,
        train_day_indices: np.ndarray,
    ) -> None:
        """Load the memory bank with day-level keys and values.

        Args:
            keys: [T, key_dim] raw episode keys for every panel day.
            values: [T, value_dim] episode values for every panel day.
            day_indices: [T] integer day indices matching keys/values.
            train_day_indices: indices used to compute standardization stats.
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
        self.mem_keys.data = torch.from_numpy(keys_std.astype(np.float32))
        self.mem_values.data = torch.from_numpy(values.astype(np.float32))
        self.mem_day_idx.data = torch.from_numpy(day_indices.astype(np.int64))

    def standardize_query(self, raw_key: Tensor) -> Tensor:
        """Standardize a raw query key using the bank's training stats."""
        return (raw_key - self.key_mean) / self.key_std

    def retrieve(
        self,
        query_raw_key: Tensor,
        query_day_idx: int,
        allowed_day_indices: Tensor,
    ) -> dict[str, Tensor]:
        """Retrieve top-M leakage-safe episodes for one query day.

        Args:
            query_raw_key: [key_dim] raw query key.
            query_day_idx: integer day index of the query.
            allowed_day_indices: 1-D long tensor listing memory days that are
                allowed for this query (e.g., training days only). The
                leakage rule s + horizon + embargo < t is applied on top of
                this allowlist as a second filter.

        Returns:
            Dict with:
                values: [M, value_dim] retrieved episode values.
                similarities: [M] cosine similarities (top-M).
                day_indices: [M] long tensor of retrieved day indices.
                top1_sim: scalar top-1 similarity.
                sim_entropy: scalar entropy of the softmax over similarities.
        """
        cfg = self.cfg
        cutoff = query_day_idx - cfg.horizon_days - cfg.embargo_days

        device = query_raw_key.device
        mem_day_idx = self.mem_day_idx.to(device)
        allowed = allowed_day_indices.to(device)

        # Build a boolean mask: memory day s is eligible if
        #   s in allowed AND s < cutoff.
        eligible_mask = torch.zeros_like(mem_day_idx, dtype=torch.bool)
        # Use a scatter approach for membership testing without Python loops.
        max_idx = max(int(mem_day_idx.max().item()) if mem_day_idx.numel() else 0,
                      int(allowed.max().item()) if allowed.numel() else 0,
                      query_day_idx) + 1
        is_allowed = torch.zeros(max_idx + 1, dtype=torch.bool, device=device)
        is_allowed[allowed] = True
        eligible_mask = is_allowed[mem_day_idx] & (mem_day_idx < cutoff)

        if eligible_mask.sum() == 0:
            empty_v = torch.zeros(cfg.top_m, self.value_dim, device=device)
            empty_s = torch.zeros(cfg.top_m, device=device)
            empty_d = torch.full((cfg.top_m,), -1, dtype=torch.long, device=device)
            return {
                "values": empty_v,
                "similarities": empty_s,
                "day_indices": empty_d,
                "top1_sim": torch.zeros((), device=device),
                "sim_entropy": torch.zeros((), device=device),
            }

        q_std = self.standardize_query(query_raw_key)
        q_proj = self.query_projector(q_std)
        k_proj = self.key_projector(self.mem_keys[eligible_mask])

        q_norm = q_proj / (q_proj.norm(p=2) + 1e-8)
        k_norm = k_proj / (k_proj.norm(p=2, dim=-1, keepdim=True) + 1e-8)
        sims = (k_norm @ q_norm).squeeze(-1)  # [num_eligible]

        m = min(cfg.top_m, sims.shape[0])
        if cfg.random_retrieval:
            idx = torch.randperm(sims.shape[0], device=device)[:m]
        else:
            idx = torch.topk(sims, k=m, largest=True).indices

        eligible_indices = torch.nonzero(eligible_mask).squeeze(-1)
        chosen = eligible_indices[idx]
        values = self.mem_values[chosen]
        similarities = sims[idx]
        day_indices = self.mem_day_idx[chosen]

        # Pad to top_m if fewer eligible
        if m < cfg.top_m:
            pad = cfg.top_m - m
            values = torch.cat([values, torch.zeros(pad, self.value_dim, device=device)], dim=0)
            similarities = torch.cat([similarities, torch.zeros(pad, device=device)])
            day_indices = torch.cat(
                [day_indices, torch.full((pad,), -1, dtype=torch.long, device=device)]
            )

        soft = torch.softmax(similarities, dim=0)
        entropy = -(soft * torch.log(soft.clamp(min=1e-8))).sum()

        return {
            "values": values,
            "similarities": similarities,
            "day_indices": day_indices,
            "top1_sim": similarities[0],
            "sim_entropy": entropy,
        }


__all__ = ["EpisodeMemoryBank", "EpisodeMemoryConfig"]
