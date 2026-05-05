"""STAR: Risk-Aware Spatio-Temporal Attention for Cross-Sectional Stock Ranking.

Alternative architecture per `mars-and-star-implementation.md` Section 5.

Four stages per prediction day:
  1. Patch construction: for each active ticker, gather a (N+1, W, F)
     patch from the feature window and the top-N graph neighbors.
     (The graph is used only for neighbor selection; no message passing.)
  2. Transformer encoder over the flattened patch with 2D positional
     encodings (spatial position + temporal offset).
  3. Risk-conditioned Feature-wise Linear Modulation (FiLM) from a 7-dim
     per-day global risk vector.
  4. Cross-sectional LayerNorm + dual Multi-Layer Perceptron (MLP) heads
     (rank score + risk quantiles) + auxiliary next-period Volatility
     head (for the risk-awareness defense loss).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from src.mtgn.model.layers.cs_layer_norm import CrossSectionalLayerNorm
from src.mtgn.model.layers.film import RiskFiLM


@dataclass
class STARConfig:
    feature_dim: int = 22
    hidden_dim: int = 128
    num_neighbors: int = 8                 # N
    temporal_window: int = 20              # W
    num_heads: int = 4
    num_layers: int = 2
    ff_dim: int = 256
    transformer_dropout: float = 0.1
    risk_feature_dim: int = 7
    head_hidden: int = 64
    head_dropout: float = 0.2
    use_risk_head: bool = True
    risk_quantiles: tuple[float, ...] = (0.05, 0.50, 0.95)
    # Ablation switches:
    disable_film: bool = False             # FiLM gamma=1, beta=0
    disable_aux: bool = False              # auxiliary vol-prediction loss off


class STAR(nn.Module):
    def __init__(self, cfg: STARConfig):
        super().__init__()
        self.cfg = cfg
        NP1 = cfg.num_neighbors + 1

        self.input_proj = nn.Linear(cfg.feature_dim, cfg.hidden_dim)
        self.spatial_pe  = nn.Embedding(NP1, cfg.hidden_dim)
        self.temporal_pe = nn.Embedding(cfg.temporal_window, cfg.hidden_dim)

        layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim, nhead=cfg.num_heads,
            dim_feedforward=cfg.ff_dim, dropout=cfg.transformer_dropout,
            activation="gelu", batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=cfg.num_layers)

        self.film = RiskFiLM(cfg.risk_feature_dim, cfg.hidden_dim)

        self.cs_ln = CrossSectionalLayerNorm(cfg.hidden_dim)
        self.rank_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.head_hidden), nn.GELU(),
            nn.Dropout(cfg.head_dropout),
            nn.Linear(cfg.head_hidden, 1),
        )
        if cfg.use_risk_head:
            self.risk_head = nn.Sequential(
                nn.Linear(cfg.hidden_dim, cfg.head_hidden), nn.GELU(),
                nn.Dropout(cfg.head_dropout),
                nn.Linear(cfg.head_hidden, len(cfg.risk_quantiles)),
            )
        # Auxiliary next-period volatility head
        self.aux_vol_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, 32), nn.GELU(), nn.Linear(32, 1),
        )

    def forward_day(self, patches: Tensor, patch_mask: Tensor,
                    risk_features: Tensor, active_mask: Tensor) -> dict[str, Tensor]:
        """
        patches:       [A, N+1, W, F]   patch_mask: [A, N+1, W]
        risk_features: [risk_dim]       active_mask: [num_nodes] bool
        """
        cfg = self.cfg
        A, NP1, W, F = patches.shape
        D = cfg.hidden_dim
        num_nodes = active_mask.shape[0]

        x = self.input_proj(patches)                         # [A, N+1, W, D]
        sp = self.spatial_pe(torch.arange(NP1, device=x.device))   # [N+1, D]
        tp = self.temporal_pe(torch.arange(W, device=x.device))    # [W, D]
        x = x + sp.view(1, NP1, 1, D) + tp.view(1, 1, W, D)

        x_flat = x.reshape(A, NP1 * W, D)
        mask_flat = patch_mask.reshape(A, NP1 * W)
        key_pad = ~mask_flat                                 # True = ignore
        x_enc = self.transformer(x_flat, src_key_padding_mask=key_pad)

        # Extract (self=row 0, today = last temporal position)
        self_today_idx = 0 * W + (W - 1)
        z = x_enc[:, self_today_idx, :]                      # [A, D]

        # Stage 3: FiLM modulation
        if cfg.disable_film:
            z_cond = z
        else:
            z_cond = self.film(z, risk_features)

        # Auxiliary vol prediction (from pooled representation)
        aux_vol = self.aux_vol_head(z_cond.mean(dim=0)).squeeze(-1) if not cfg.disable_aux else None

        # Scatter back to [num_nodes, D] for cs-LN + heads
        z_full = torch.zeros(num_nodes, D, device=z.device, dtype=z.dtype)
        z_full[active_mask] = z_cond
        z_norm = self.cs_ln(z_full, active_mask)

        y_hat = self.rank_head(z_norm).squeeze(-1)
        q_hat = self.risk_head(z_norm) if cfg.use_risk_head else None

        return {"y_hat": y_hat, "q_hat": q_hat, "aux_vol": aux_vol, "z": z_norm}


__all__ = ["STAR", "STARConfig"]
