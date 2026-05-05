"""Memorizing Temporal Graph Network (MTGN) model.

Phase 1 structure:
    [vanilla TGN backbone]  ->  [ranking head + quantile head]

The TGN backbone uses PyG's TGNMemory (via src.mtgn.memory.tgn_memory) for
the Rossi ordering and GRU memory update. The spatial attention over
current temporal neighbors uses TGAT-style graph attention.

The MTGN-specific second attention mechanism (sparse temporal attention
over a salience-gated episodic store of past memory snapshots) is NOT
yet wired here. It lives in `src/mtgn/attention/` and
`src/mtgn/store/`, both to be implemented next; this class exposes hooks
(`self.episodic_store` attribute and an optional `temporal_attention`
module) so the addition is surgical.

Heads:
    * ranking head: two-layer MLP producing `hat_y_i(t)` for ListNet.
    * risk-aware quantile head: two-layer MLP producing
      `hat_q_tau(t)` for tau in {0.05, 0.50, 0.95} via pinball loss.
      In Phase 1 the non-parametric empirical distribution over
      retrieved analogs (memo Section 2.5 quantile head) is absent
      until the store lands; the MLP head is trained directly on
      the embedding z_i(t) and serves as a floor baseline that can
      only improve once retrieval is wired.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import Tensor, nn

from src.mtgn.memory.tgn_memory import (
    MemoryConfig,
    TGNMemory,
    TimeEncoder,
    build_tgn_memory,
    detach_between_batches,
)


try:
    from torch_geometric.nn import TransformerConv
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "torch_geometric is required. Install with `pip install torch-geometric`."
    ) from e


@dataclass
class MTGNConfig:
    num_nodes: int
    raw_msg_dim: int
    node_feature_dim: int
    memory_dim: int = 128
    time_dim: int = 32
    attention_heads: int = 4
    attention_hidden: int = 128
    ranking_hidden: int = 64
    risk_quantiles: tuple[float, ...] = (0.05, 0.50, 0.95)
    risk_hidden: int = 64


class SpatialAttention(nn.Module):
    """TGAT-style graph attention over temporal neighbors.

    Single-layer TransformerConv over a subgraph of (target, neighbor,
    time-encoded edge) tuples. PyG handles the aggregation; we pass
    the time encoding as edge features.
    """

    def __init__(self, cfg: MTGNConfig):
        super().__init__()
        self.time_encoder = TimeEncoder(cfg.time_dim)
        in_dim = cfg.memory_dim + cfg.node_feature_dim
        edge_dim = cfg.time_dim + cfg.raw_msg_dim
        self.conv = TransformerConv(
            in_channels=in_dim,
            out_channels=cfg.attention_hidden // cfg.attention_heads,
            heads=cfg.attention_heads,
            edge_dim=edge_dim,
        )

    def forward(
        self,
        x: Tensor,               # [N_nodes, memory_dim + node_feature_dim]
        edge_index: Tensor,      # [2, E]
        edge_time: Tensor,       # [E]
        edge_attr: Tensor,       # [E, raw_msg_dim]
        ref_time: Tensor,        # [E] query time per edge
    ) -> Tensor:
        rel_t = ref_time - edge_time
        t_enc = self.time_encoder(rel_t.float())
        edge_feat = torch.cat([t_enc, edge_attr], dim=-1)
        return self.conv(x, edge_index, edge_attr=edge_feat)


class RankingHead(nn.Module):
    def __init__(self, cfg: MTGNConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.attention_hidden, cfg.ranking_hidden),
            nn.ReLU(),
            nn.Linear(cfg.ranking_hidden, 1),
        )

    def forward(self, z: Tensor) -> Tensor:
        return self.net(z).squeeze(-1)


class QuantileHead(nn.Module):
    """Foundational Phase 1 quantile head.

    Phase 1 Scenario C: consumes only z_i(t) and predicts the quantile
    directly via a small MLP trained with pinball loss. When the
    episodic store + dual attention land, this head will additionally
    consume the empirical conditional distribution from retrieved
    analog forward returns (memo Section 2.5 / proposal Section 3.1).
    """

    def __init__(self, cfg: MTGNConfig):
        super().__init__()
        self.taus = cfg.risk_quantiles
        self.net = nn.Sequential(
            nn.Linear(cfg.attention_hidden, cfg.risk_hidden),
            nn.ReLU(),
            nn.Linear(cfg.risk_hidden, len(cfg.risk_quantiles)),
        )

    def forward(self, z: Tensor) -> Tensor:
        return self.net(z)   # [N, |quantiles|]


def pinball_loss(y: Tensor, q_hat: Tensor, taus: tuple[float, ...]) -> Tensor:
    """Quantile (pinball) loss aggregated over quantile levels.

    y: [N] realized forward returns
    q_hat: [N, |taus|] predicted quantiles
    """
    diff = y.unsqueeze(-1) - q_hat
    taus_t = torch.tensor(taus, dtype=diff.dtype, device=diff.device)
    return torch.maximum(taus_t * diff, (taus_t - 1.0) * diff).mean()


class MTGN(nn.Module):
    """Memorizing TGN, Phase 1 (vanilla TGN + ranking + quantile heads).

    Episodic store and dual attention are added incrementally. The
    hooks on this class make that addition surgical rather than
    requiring a rewrite.
    """

    def __init__(self, cfg: MTGNConfig):
        super().__init__()
        self.cfg = cfg
        self.memory: TGNMemory = build_tgn_memory(
            MemoryConfig(
                num_nodes=cfg.num_nodes,
                raw_msg_dim=cfg.raw_msg_dim,
                memory_dim=cfg.memory_dim,
                time_dim=cfg.time_dim,
            )
        )
        self.spatial = SpatialAttention(cfg)
        self.ranking_head = RankingHead(cfg)
        self.risk_head = QuantileHead(cfg)

        # Hooks for Phase 1 -> Phase 2 extension (store + dual attention)
        self.episodic_store = None          # type: ignore[assignment]
        self.temporal_attention = None      # type: ignore[assignment]

    def detach_memory(self) -> None:
        detach_between_batches(self.memory)

    def reset_memory(self) -> None:
        self.memory.reset_state()

    def forward(
        self,
        node_ids: Tensor,            # [batch_N] query node ids
        node_features: Tensor,       # [batch_N, node_feature_dim]
        edge_index: Tensor,          # [2, E] subgraph of temporal neighbors
        edge_time: Tensor,           # [E]
        edge_attr: Tensor,           # [E, raw_msg_dim]
        ref_time: Tensor,            # [E] query time for each edge
    ) -> dict[str, Tensor]:
        """Return ranking scores and quantile predictions for the batch."""
        memory_vec, _last_update = self.memory(node_ids)
        x = torch.cat([memory_vec, node_features], dim=-1)
        h_spatial = self.spatial(x, edge_index, edge_time, edge_attr, ref_time)

        # Phase 1: z = h_spatial only; Phase 1b adds + h_temporal residual.
        z = h_spatial
        if self.temporal_attention is not None:
            # Placeholder for Phase 1b dual-attention extension.
            h_temporal = self.temporal_attention(z, self.episodic_store)
            z = torch.nn.functional.layer_norm(z + h_temporal, z.shape[-1:])

        return {
            "z": z,
            "y_hat": self.ranking_head(z),
            "q_hat": self.risk_head(z),
        }


__all__ = [
    "MTGN",
    "MTGNConfig",
    "SpatialAttention",
    "RankingHead",
    "QuantileHead",
    "pinball_loss",
]
