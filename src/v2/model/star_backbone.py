"""STAR backbone for epiSTAR.

A clean Spatio-Temporal Attention Ranker (STAR) without FiLM, without an
auxiliary volatility head, and without a quantile head. The backbone takes
ticker-day patches built from top-N graph neighbors over a W-day lookback
window and produces a single hidden representation per active ticker.

This module is intentionally minimal so that epiSTAR can compose retrieval
on top of it without inheriting the v1 risk-conditioning machinery that
hurt under leakage-free evaluation.

References:
    Mars/Star implementation memo (archived at archive/v1_models/),
    epiSTAR brief 2026-05-01.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from src.mtgn.model.layers.cs_layer_norm import CrossSectionalLayerNorm


@dataclass
class STARBackboneConfig:
    """Hyperparameters for the STAR backbone.

    Attributes:
        feature_dim: number of input ticker-day features (22 in current panel).
        hidden_dim: Transformer hidden dimension.
        num_neighbors: top-N graph neighbors per ticker (excluding self).
        temporal_window: W-day lookback window length.
        num_heads: Transformer attention heads.
        num_layers: number of Transformer encoder layers.
        ff_dim: feedforward dimension inside each Transformer layer.
        dropout: Transformer dropout probability.
    """

    feature_dim: int = 22
    hidden_dim: int = 128
    num_neighbors: int = 8
    temporal_window: int = 20
    num_heads: int = 4
    num_layers: int = 2
    ff_dim: int = 256
    dropout: float = 0.1


class STARBackbone(nn.Module):
    """STAR encoder producing per-ticker hidden representations.

    The forward pass operates on a single day's worth of (active-ticker,
    neighbor, time, feature) patches. It returns a [num_nodes, hidden_dim]
    tensor with inactive rows zeroed and active rows cross-sectionally
    layer-normalized.
    """

    def __init__(self, cfg: STARBackboneConfig):
        super().__init__()
        self.cfg = cfg
        n_plus_1 = cfg.num_neighbors + 1

        self.input_proj = nn.Linear(cfg.feature_dim, cfg.hidden_dim)
        self.spatial_pe = nn.Embedding(n_plus_1, cfg.hidden_dim)
        self.temporal_pe = nn.Embedding(cfg.temporal_window, cfg.hidden_dim)

        layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.ff_dim,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=cfg.num_layers)
        self.cs_ln = CrossSectionalLayerNorm(cfg.hidden_dim)

    def forward_day(
        self,
        patches: Tensor,
        patch_mask: Tensor,
        active_mask: Tensor,
    ) -> Tensor:
        """Encode one day's patches into per-ticker representations.

        Args:
            patches: [A, N+1, W, F] feature tensor for A active tickers.
            patch_mask: [A, N+1, W] bool, True where the cell is observed.
            active_mask: [num_nodes] bool, True for active tickers.

        Returns:
            z: [num_nodes, hidden_dim] cross-sectionally layer-normalized,
                with inactive rows zeroed.
        """
        cfg = self.cfg
        a, n_plus_1, w, _ = patches.shape
        d = cfg.hidden_dim
        num_nodes = active_mask.shape[0]

        x = self.input_proj(patches)
        sp = self.spatial_pe(torch.arange(n_plus_1, device=x.device))
        tp = self.temporal_pe(torch.arange(w, device=x.device))
        x = x + sp.view(1, n_plus_1, 1, d) + tp.view(1, 1, w, d)

        x_flat = x.reshape(a, n_plus_1 * w, d)
        mask_flat = patch_mask.reshape(a, n_plus_1 * w)
        key_pad = ~mask_flat
        x_enc = self.transformer(x_flat, src_key_padding_mask=key_pad)

        # Extract self position (row 0) at the most recent time step.
        self_today_idx = 0 * w + (w - 1)
        z_active = x_enc[:, self_today_idx, :]

        z_full = torch.zeros(num_nodes, d, device=z_active.device, dtype=z_active.dtype)
        z_full[active_mask] = z_active
        z_norm = self.cs_ln(z_full, active_mask)
        return z_norm


__all__ = ["STARBackbone", "STARBackboneConfig"]
