"""DualRetrieval: regime memory + novelty memory banks.

Per spec section 6.6.

Two retrieval banks:

  Regime bank: keys are 14-d per-training-day regime fingerprints (cs return
    moments, vol moments, pairwise corr, dispersion, active count). Values
    are 32-d learned summaries. Per-day retrieval; multi-head cross-attention
    fuses retrieved values into per-stock embeddings via a per-day gate
    alpha_regime.

  Novelty bank: keys are sector-neutral novelty signatures per (training
    day, training ticker): months_since_ipo, log_market_cap, log_dollar_volume,
    realized_vol_20d, st_volume_abnormal_z60d, st_volume_24h_log,
    idiosyncratic_vol_60d, plus GICS sector embedding. Values are 32-d
    learned summaries. Per-(day, ticker) retrieval with per-cell gate
    alpha_novelty. Restricted to entries with months_since_ipo <= 36.

Both banks: leakage gate `tau <= t - 10` (5-day forward + 5-day embargo).

Banks are populated from training fold only and frozen at training-end
before validation and test.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn


@dataclass
class DualRetrievalConfig:
    regime_key_dim: int = 14
    regime_value_dim: int = 32
    novelty_key_dim: int = 8
    novelty_value_dim: int = 32
    sector_embedding_dim: int = 16
    n_sectors: int = 11
    d_model: int = 128
    top_m: int = 8
    horizon_days: int = 5
    embargo_days: int = 5
    n_attn_heads: int = 4


class RegimeMemory(nn.Module):
    """Regime bank: per-day 14-d key, 32-d value, top-M cross-attended."""

    def __init__(self, cfg: DualRetrievalConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.value_proj = nn.Linear(cfg.regime_key_dim, cfg.regime_value_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=cfg.d_model, num_heads=cfg.n_attn_heads,
            kdim=cfg.regime_value_dim, vdim=cfg.regime_value_dim,
            batch_first=True,
        )
        # Gate from per-day key to scalar in (0, 1). Init bias -1 so
        # alpha_regime ~= sigmoid(-1) ~= 0.27 at training start.
        self.gate_mlp = nn.Sequential(
            nn.Linear(cfg.regime_key_dim, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, 1),
        )
        with torch.no_grad():
            self.gate_mlp[-1].bias.fill_(-1.0)
        # Bank state (registered buffers; populated by populate_bank()).
        self.register_buffer("bank_keys", torch.zeros(1, cfg.regime_key_dim))
        self.register_buffer("bank_values", torch.zeros(1, cfg.regime_value_dim))
        self.register_buffer("bank_day_idx", torch.zeros(1, dtype=torch.long))
        self._bank_populated = False

    def populate_bank(self, keys: Tensor, day_indices: Tensor) -> None:
        """Set the regime bank from training-fold keys.

        Args:
            keys: [n_train_days, regime_key_dim] standardized.
            day_indices: [n_train_days] integer day indices.
        """
        with torch.no_grad():
            values = self.value_proj(keys)
        self.bank_keys = keys.detach()
        self.bank_values = values.detach()
        self.bank_day_idx = day_indices.detach().long()
        self._bank_populated = True

    def forward(
        self, z: Tensor, query_keys: Tensor, query_day_idx: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Retrieve and fuse.

        Args:
            z: [B, N, d_model] per-stock embeddings.
            query_keys: [B, regime_key_dim] per-day query keys.
            query_day_idx: [B] integer day indices for leakage gate.

        Returns:
            (delta_regime, alpha_regime).
            delta_regime: [B, N, d_model] residual to add to z.
            alpha_regime: [B, 1, 1] per-day gate.
        """
        B, N, D = z.shape
        if not self._bank_populated or self.bank_keys.shape[0] < 2:
            zero = torch.zeros_like(z)
            alpha = torch.zeros(B, 1, 1, device=z.device)
            return zero, alpha

        cutoff = query_day_idx - self.cfg.horizon_days - self.cfg.embargo_days  # [B]

        # Build per-batch eligibility mask: bank_day_idx < cutoff[b].
        bank_idx = self.bank_day_idx.to(z.device).unsqueeze(0)  # [1, M_bank]
        eligible = bank_idx < cutoff.unsqueeze(1)               # [B, M_bank]

        # Top-M by L2 distance per batch
        # Compute L2 dists between each batch's query and all bank keys
        keys = self.bank_keys.to(z.device)                       # [M_bank, K]
        diffs = query_keys.unsqueeze(1) - keys.unsqueeze(0)       # [B, M_bank, K]
        dists = (diffs ** 2).sum(dim=-1)                          # [B, M_bank]
        # Mask ineligible to +inf
        dists = dists.masked_fill(~eligible, float("inf"))
        # Top-M (smallest distances)
        top = torch.topk(-dists, k=min(self.cfg.top_m, dists.shape[-1]), dim=-1)
        top_idx = top.indices                                     # [B, M]
        top_valid = torch.isfinite(-top.values)                   # [B, M]

        # Gather values
        values = self.bank_values.to(z.device)                    # [M_bank, V]
        top_values = values[top_idx]                              # [B, M, V]
        # Multi-head cross-attention from z (queries) to top_values (KV)
        # Mask invalid positions
        key_padding = ~top_valid                                  # [B, M]
        attn_out, _ = self.cross_attn(
            z, top_values, top_values,
            key_padding_mask=key_padding, need_weights=False,
        )
        alpha = torch.sigmoid(self.gate_mlp(query_keys)).unsqueeze(1)  # [B, 1, 1]
        return attn_out, alpha


class NoveltyMemory(nn.Module):
    """Novelty bank: per-(day, ticker) signature, 32-d value."""

    def __init__(self, cfg: DualRetrievalConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.sector_embed = nn.Embedding(cfg.n_sectors + 1, cfg.sector_embedding_dim,
                                          padding_idx=0)
        key_total_dim = cfg.novelty_key_dim + cfg.sector_embedding_dim
        self.value_proj = nn.Linear(key_total_dim, cfg.novelty_value_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=cfg.d_model, num_heads=cfg.n_attn_heads,
            kdim=cfg.novelty_value_dim, vdim=cfg.novelty_value_dim,
            batch_first=True,
        )
        # Per-(day, ticker) gate; init bias -1 so alpha_novelty ~= 0.27 at start.
        self.gate_mlp = nn.Sequential(
            nn.Linear(key_total_dim, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, 1),
        )
        with torch.no_grad():
            self.gate_mlp[-1].bias.fill_(-1.0)
        # Bank state
        self.register_buffer("bank_keys", torch.zeros(1, key_total_dim))
        self.register_buffer("bank_values", torch.zeros(1, cfg.novelty_value_dim))
        self.register_buffer("bank_day_idx", torch.zeros(1, dtype=torch.long))
        self._bank_populated = False

    def populate_bank(
        self, keys_numeric: Tensor, sector_ids: Tensor, day_indices: Tensor,
    ) -> None:
        """Populate from training-fold per-(day, ticker) novelty entries.

        Args:
            keys_numeric: [M, novelty_key_dim] numeric features.
            sector_ids: [M] long sector ids in [0, n_sectors-1].
            day_indices: [M] integer day indices.
        """
        with torch.no_grad():
            sec_emb = self.sector_embed(sector_ids.long() + 1)  # +1 for padding_idx=0
            full_keys = torch.cat([keys_numeric, sec_emb], dim=-1)
            values = self.value_proj(full_keys)
        self.bank_keys = full_keys.detach()
        self.bank_values = values.detach()
        self.bank_day_idx = day_indices.detach().long()
        self._bank_populated = True

    def forward(
        self, z: Tensor, query_keys_numeric: Tensor,
        query_sector_ids: Tensor, query_day_idx: Tensor,
        active_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Per-(day, ticker) retrieval.

        Args:
            z: [B, N, d_model].
            query_keys_numeric: [B, N, novelty_key_dim].
            query_sector_ids: [B, N] long sector ids.
            query_day_idx: [B] integer day indices.
            active_mask: [B, N] bool.

        Returns:
            (delta_novelty, alpha_novelty).
            delta_novelty: [B, N, d_model].
            alpha_novelty: [B, N, 1].
        """
        B, N, D = z.shape
        if not self._bank_populated or self.bank_keys.shape[0] < 2:
            zero = torch.zeros_like(z)
            alpha = torch.zeros(B, N, 1, device=z.device)
            return zero, alpha
        sec_emb = self.sector_embed(query_sector_ids + 1)         # [B, N, sec_dim]
        full_query = torch.cat([query_keys_numeric, sec_emb], dim=-1)  # [B, N, K]
        cutoff = query_day_idx - self.cfg.horizon_days - self.cfg.embargo_days
        bank_idx = self.bank_day_idx.to(z.device).unsqueeze(0).unsqueeze(0)  # [1, 1, M]
        eligible = bank_idx < cutoff.view(B, 1, 1)                            # [B, 1, M]

        keys = self.bank_keys.to(z.device)                                     # [M, K]
        # Per-(B, N) query distances to bank
        diffs = full_query.unsqueeze(2) - keys.unsqueeze(0).unsqueeze(0)        # [B, N, M, K]
        dists = (diffs ** 2).sum(dim=-1)                                        # [B, N, M]
        dists = dists.masked_fill(~eligible, float("inf"))
        top = torch.topk(-dists, k=min(self.cfg.top_m, dists.shape[-1]), dim=-1)
        top_idx = top.indices                                                   # [B, N, M]
        top_valid = torch.isfinite(-top.values)                                 # [B, N, M]

        values = self.bank_values.to(z.device)                                  # [M, V]
        top_values = values[top_idx.view(-1)].view(B, N, top_idx.shape[-1], -1) # [B, N, M, V]
        # Cross-attention: each (B, N) row queries its M retrieved values.
        # Reshape to (B*N, 1, D) queries against (B*N, M, V) KV.
        z_flat = z.reshape(B * N, 1, D)
        kv_flat = top_values.reshape(B * N, top_values.shape[-2], top_values.shape[-1])
        kpad = ~top_valid.reshape(B * N, top_valid.shape[-1])
        attn_out, _ = self.cross_attn(z_flat, kv_flat, kv_flat,
                                       key_padding_mask=kpad, need_weights=False)
        attn_out = attn_out.view(B, N, D)
        attn_out = attn_out * active_mask.unsqueeze(-1).float()
        alpha = torch.sigmoid(self.gate_mlp(full_query))                        # [B, N, 1]
        return attn_out, alpha


class DualRetrieval(nn.Module):
    """Composes RegimeMemory and NoveltyMemory."""

    def __init__(self, cfg: DualRetrievalConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or DualRetrievalConfig()
        self.cfg = cfg
        self.regime = RegimeMemory(cfg)
        self.novelty = NoveltyMemory(cfg)


__all__ = [
    "DualRetrievalConfig", "DualRetrieval", "RegimeMemory", "NoveltyMemory",
]
