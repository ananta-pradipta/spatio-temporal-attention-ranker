"""Proposal D1: Set Transformer encoder for universe-agnostic ranking.

Motivation (244-ticker diagnostic):
  R-STAR's per-ticker patch + top-8 neighbor encoder relies on a static
  mechanistic graph built from training-window data. IPO tickers appearing
  in test windows have no edges in that graph, no training history in
  their neighbor slots, and feature distributions out-of-sample relative
  to training statistics. Result: test IC drops from +0.0527 (84-ticker
  closed universe) to +0.0147 (244-ticker open universe), while rank-IC
  holds up better (+0.0327), indicating ordering is preserved but
  magnitude calibration fails.

Set Transformer (Lee et al. 2019) reframes ranking as a function over
the SET of active tickers today, permutation-invariant and universe-size
invariant by construction. Each ticker's temporal history is encoded
independently, then set-attention runs across the full active universe.
New IPOs just appear as additional set members; no retraining, no graph
edge required.

Architecture:
  Stage 1: per-ticker temporal encoder (GRU or 1-layer Transformer) maps
  [W, F] -> [D] per ticker.
  Stage 2: 2-layer self-attention (4 heads) across [N_active, D] with
  optional regime context token prepended.
  Stage 3: cross-sectional LayerNorm, MLP rank head.

Compatibility: same forward_day interface as REMStar (y_hat, z, etc.),
so the training loop in train.py can swap model classes via a flag.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from src.mtgn.model.layers.cs_layer_norm import CrossSectionalLayerNorm


@dataclass
class SetSTARConfig:
    feature_dim: int = 22
    hidden_dim: int = 128
    temporal_window: int = 20
    num_heads: int = 4
    num_layers: int = 2
    ff_dim: int = 256
    transformer_dropout: float = 0.1
    signature_dim: int = 4
    head_hidden: int = 64
    head_dropout: float = 0.2
    use_regime_token: bool = True
    num_prototypes: int = 16
    proto_temperature: float = 1.0
    # Per-ticker temporal encoder: "gru" or "transformer"
    temporal_encoder: str = "gru"


class SetSTAR(nn.Module):
    """Universe-agnostic ranker: set-attention over all active tickers.

    Stage 1 (per-ticker temporal encoder): reduces each ticker's
    [W, F] history to a single [D]-dim vector, using either a GRU or
    a light 1-layer Transformer encoder over the temporal axis.
    Stage 2 (set self-attention): 2-layer Transformer across tickers
    with optional regime token prepended as extra set member.
    Stage 3 (ranking head): cross-sectional LayerNorm + 2-layer MLP.
    """

    def __init__(self, cfg: SetSTARConfig):
        super().__init__()
        self.cfg = cfg
        D = cfg.hidden_dim

        # Stage 1: per-ticker temporal encoder
        self.input_proj = nn.Linear(cfg.feature_dim, D)
        if cfg.temporal_encoder == "gru":
            self.temporal_enc = nn.GRU(
                input_size=D, hidden_size=D, batch_first=True,
            )
        elif cfg.temporal_encoder == "transformer":
            self.temporal_pe = nn.Embedding(cfg.temporal_window, D)
            tlayer = nn.TransformerEncoderLayer(
                d_model=D, nhead=cfg.num_heads,
                dim_feedforward=cfg.ff_dim, dropout=cfg.transformer_dropout,
                activation="gelu", batch_first=True,
            )
            self.temporal_enc = nn.TransformerEncoder(tlayer, num_layers=1)
        else:
            raise ValueError(f"unknown temporal_encoder: {cfg.temporal_encoder}")

        # Stage 2: set self-attention (over tickers)
        slayer = nn.TransformerEncoderLayer(
            d_model=D, nhead=cfg.num_heads,
            dim_feedforward=cfg.ff_dim, dropout=cfg.transformer_dropout,
            activation="gelu", batch_first=True,
        )
        self.set_enc = nn.TransformerEncoder(slayer, num_layers=cfg.num_layers)

        # Regime prototype mechanism (optional, same as REM iter 3B)
        if cfg.num_prototypes > 0:
            self.proto_sig = nn.Parameter(
                torch.randn(cfg.num_prototypes, cfg.signature_dim) * 0.5
            )
            self.proto_summary = nn.Parameter(
                torch.randn(cfg.num_prototypes, D) * 0.02
            )
            self.regime_token_pe = nn.Parameter(torch.zeros(1, 1, D))
            nn.init.normal_(self.regime_token_pe, std=0.02)

        # Stage 3: cross-sectional layer norm + rank head
        self.cs_ln = CrossSectionalLayerNorm(D)
        self.rank_head = nn.Sequential(
            nn.Linear(D, cfg.head_hidden), nn.GELU(),
            nn.Dropout(cfg.head_dropout),
            nn.Linear(cfg.head_hidden, 1),
        )

    def _encode_temporal(self, x_hist: Tensor) -> Tensor:
        """x_hist: [A, W, F] -> [A, D] per-ticker embedding."""
        h = self.input_proj(x_hist)                                # [A, W, D]
        if self.cfg.temporal_encoder == "gru":
            _, h_last = self.temporal_enc(h)                       # h_last: [1, A, D]
            return h_last.squeeze(0)                               # [A, D]
        else:
            W = h.shape[1]
            pe = self.temporal_pe(torch.arange(W, device=h.device))
            h = h + pe.view(1, W, h.shape[2])
            h = self.temporal_enc(h)                               # [A, W, D]
            return h[:, -1, :]                                     # last-timestep pooling -> [A, D]

    def forward_day(self, x_hist: Tensor, x_mask: Tensor,
                    regime_sig: Tensor, active_mask: Tensor) -> dict[str, Tensor]:
        """
        x_hist: [A, W, F] each active ticker's last-W-day feature history
        x_mask: [A, W] True = valid day
        regime_sig: [signature_dim] day-t regime signature (z-scored)
        active_mask: [num_nodes] bool mask selecting active tickers
        """
        cfg = self.cfg
        A, W, F = x_hist.shape
        D = cfg.hidden_dim
        num_nodes = active_mask.shape[0]

        # Stage 1: per-ticker temporal encoding
        h = self._encode_temporal(x_hist)                          # [A, D]

        # Stage 2: build set + optional regime token, self-attend
        soft_weights = None
        if cfg.num_prototypes > 0:
            d2 = ((regime_sig.view(1, -1) - self.proto_sig) ** 2).sum(dim=1)
            soft_weights = torch.softmax(-d2 / cfg.proto_temperature, dim=0)
            regime_summary = (soft_weights.view(-1, 1) * self.proto_summary).sum(dim=0)  # [D]
            regime_tok = regime_summary.view(1, 1, D) + self.regime_token_pe              # [1, 1, D]
            regime_tok = regime_tok.expand(1, 1, D).squeeze(0)                             # [1, D]
        else:
            regime_tok = None

        set_input = h.unsqueeze(0)                                 # [1, A, D]
        if regime_tok is not None:
            regime_elem = regime_tok.view(1, 1, D)                 # [1, 1, D]
            set_input = torch.cat([regime_elem, set_input], dim=1)  # [1, A+1, D]
            prefix = 1
        else:
            prefix = 0

        set_out = self.set_enc(set_input)                          # [1, A+prefix, D]
        z = set_out.squeeze(0)[prefix:]                            # [A, D]  drop regime slot

        # Stage 3: CS layer norm + rank head
        z_full = torch.zeros(num_nodes, D, device=z.device, dtype=z.dtype)
        z_full[active_mask] = z
        z_norm = self.cs_ln(z_full, active_mask)
        y_hat = self.rank_head(z_norm).squeeze(-1)

        return {
            "y_hat": y_hat,
            "z": z_norm,
            "proto_weights": soft_weights,
        }


__all__ = ["SetSTAR", "SetSTARConfig"]
