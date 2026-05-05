"""G²-REM-STAR: REM iter 3B graph path + StockMixer-style no-graph path + gate.

Combines the two best-performing mechanisms discovered in the
investigation:
  - Graph path: REM iter 3B configuration (learnable L=16 prototypes +
    soft-weighted summary token + pure STAR single mechanistic graph).
    Model 1's falsification-bar winner.
  - No-graph path: per-ticker transformer + StockMixer-style cross-stock
    MLP. Same as G²-STAR iter 4/5.
  - Gate: temperature-scaled sigmoid (Fix A) over 7-dim enriched regime
    signature. Same as G²-STAR iter 5.

Output: y_hat = α · y_graph + (1 − α) · y_nograph.

The REM graph path consumes only the first 4 dimensions of the
signature (to remain compatible with REM's 4-dim prototype space).
The gate MLP uses all 7 dimensions.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from src.investigation.g2_star.model import _CrossStockMixer, _PerTickerEncoder
from src.investigation.regime_memory.model import REMConfig, REMStar
from src.mtgn.model.layers.cs_layer_norm import CrossSectionalLayerNorm


@dataclass
class G2REMConfig:
    feature_dim: int = 22
    hidden_dim: int = 128
    num_neighbors: int = 8
    temporal_window: int = 20
    num_heads: int = 4
    num_layers: int = 2
    ff_dim: int = 256
    transformer_dropout: float = 0.1
    head_hidden: int = 64
    head_dropout: float = 0.2
    signature_dim: int = 7
    signature_dim_rem: int = 4           # REM graph path takes first 4 dims only
    gate_hidden: int = 16
    gate_temperature: float = 0.2
    use_xstock_mlp: bool = True
    xstock_hidden: int = 8
    num_stocks: int = 84
    # REM iter 3B config
    num_prototypes: int = 16
    proto_temperature: float = 1.0
    sparsity_weight: float = 0.01
    num_memory_tokens: int = 0
    K: int = 4                            # k-means catalog size (unused but required)


class G2REMStar(nn.Module):
    def __init__(self, cfg: G2REMConfig):
        super().__init__()
        self.cfg = cfg
        D = cfg.hidden_dim

        # Graph path: REM iter 3B
        rem_cfg = REMConfig(
            feature_dim=cfg.feature_dim, hidden_dim=D,
            num_neighbors=cfg.num_neighbors, temporal_window=cfg.temporal_window,
            num_heads=cfg.num_heads, num_layers=cfg.num_layers, ff_dim=cfg.ff_dim,
            transformer_dropout=cfg.transformer_dropout,
            signature_dim=cfg.signature_dim_rem,
            head_hidden=cfg.head_hidden, head_dropout=cfg.head_dropout,
            use_risk_head=False, use_regime_token=False,
            num_memory_tokens=cfg.num_memory_tokens, num_clusters=cfg.K,
            num_prototypes=cfg.num_prototypes, proto_temperature=cfg.proto_temperature,
        )
        self.graph_rem = REMStar(rem_cfg)

        # No-graph path
        self.nograph_enc = _PerTickerEncoder(
            type("C", (), dict(
                hidden_dim=D, feature_dim=cfg.feature_dim,
                temporal_window=cfg.temporal_window, num_heads=cfg.num_heads,
                num_layers=cfg.num_layers, ff_dim=cfg.ff_dim,
                transformer_dropout=cfg.transformer_dropout,
            ))()
        )
        if cfg.use_xstock_mlp:
            self.xstock_mixer = _CrossStockMixer(cfg.num_stocks, cfg.xstock_hidden)
        self.cs_ln_nograph = CrossSectionalLayerNorm(D)
        self.rank_head_nograph = nn.Sequential(
            nn.Linear(D, cfg.head_hidden), nn.GELU(),
            nn.Dropout(cfg.head_dropout),
            nn.Linear(cfg.head_hidden, 1),
        )

        # Gate on full 7-dim signature
        self.gate_mlp = nn.Sequential(
            nn.Linear(cfg.signature_dim, cfg.gate_hidden), nn.GELU(),
            nn.Linear(cfg.gate_hidden, 1),
        )
        nn.init.zeros_(self.gate_mlp[-1].bias)

    def forward_day(self, patches: Tensor, patch_mask: Tensor,
                    self_window: Tensor, regime_sig_full: Tensor,
                    active_mask: Tensor, cluster_id: int = 0) -> dict[str, Tensor]:
        cfg = self.cfg
        D = cfg.hidden_dim
        num_nodes = active_mask.shape[0]

        # Graph path via REM 3B. REM expects the 4-dim base signature.
        rem_out = self.graph_rem.forward_day(
            patches, patch_mask,
            regime_sig_full[: cfg.signature_dim_rem],
            active_mask, cluster_id=cluster_id,
        )
        y_graph = rem_out["y_hat"]                                  # [num_nodes]

        # No-graph path
        z_n = self.nograph_enc(self_window)                         # [A, D]
        z_n_full = torch.zeros(num_nodes, D, device=z_n.device, dtype=z_n.dtype)
        z_n_full[active_mask] = z_n
        if cfg.use_xstock_mlp:
            z_n_full = z_n_full + self.xstock_mixer(z_n_full)
        z_n_norm = self.cs_ln_nograph(z_n_full, active_mask)
        y_nograph = self.rank_head_nograph(z_n_norm).squeeze(-1)    # [num_nodes]

        # Gate
        logit = self.gate_mlp(regime_sig_full)
        alpha = torch.sigmoid(logit / cfg.gate_temperature).squeeze(-1)
        y_hat = alpha * y_graph + (1.0 - alpha) * y_nograph

        return {
            "y_hat": y_hat,
            "y_graph": y_graph,
            "y_nograph": y_nograph,
            "alpha": alpha,
            "proto_weights": rem_out.get("proto_weights"),
        }


__all__ = ["G2REMStar", "G2REMConfig"]
