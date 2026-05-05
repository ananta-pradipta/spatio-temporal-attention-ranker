"""Adapter around RSR (Feng et al., TOIS 2019) for our biotech panel.

RSR = a per-ticker LSTM temporal encoder followed by a static
relation-aware graph attention layer. The relation graph is fixed
(same adjacency for every day); the temporal encoder produces
per-ticker embeddings, and the attention layer aggregates each
ticker's neighbours into a relation-aware representation. The
prediction head is a tiny MLP over the concatenation of the raw
embedding and the neighbour-aggregated embedding.

Our adaptation for the v2 biotech-244 panel:

  - Input per active day: ``x_window`` of shape (N_active, T, F) where
    F = 22 (the enriched panel features) and T = 20.
  - LSTM hidden size = 64 (paper default).
  - Static adjacency: precomputed in
    ``data/processed/rsr_relation_graph.pt`` by
    ``src.baselines.build_rsr_relation_graph``. The adapter receives
    only the rows/cols matching the panel ticker order; the trainer
    further restricts to ``active_idx`` per day before forward.
  - Final score: 2-layer MLP over ``[e_i ; g_i]`` -> scalar.
  - Loss in original paper: pairwise margin ranking loss. For our v2
    protocol fairness we use ``cs_mse_loss`` on z-scored 5d forward
    log returns, identical to every other v2 baseline.

The adapter mirrors the structure of ``factorvae_adapter.FactorVAEAdapter``
so the trainer is a near-clone of ``train_factorvae_v2.py``.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.baselines.vendored.rsr import StockLSTM, TemporalGraphAttention


@dataclass
class RSRHyperparams:
    """RSR architecture knobs.

    Defaults match the TOIS 2019 paper's reported configuration where
    stated; otherwise we follow the upstream
    ``Temporal_Relational_Stock_Ranking`` repository defaults.
    """

    d_feat: int = 22                  # F: number of panel features per ticker
    hidden_size: int = 64             # H: LSTM hidden + attention width
    num_layers: int = 1               # LSTM depth (paper default 1)
    dropout: float = 0.0
    head_hidden: int = 64             # MLP head hidden width
    leaky_slope: float = 0.2          # LeakyReLU slope in attention scoring


class RSRAdapter(nn.Module):
    """RSR wrapped for our (N_active, T, F) panel format.

    Public surface used by ``train_rsr_v2.py``:

        forward(x_window) -> (N_active,) scalar score per active ticker.

    Internals:
        e = StockLSTM(x_window)                    # (N_active, H)
        g = TemporalGraphAttention(e, adj_active)  # (N_active, H)
        y_hat = MLP([e ; g])                       # (N_active,)

    Args:
        hp: architecture knobs.
        full_relation_graph: (N_panel, N_panel) uint8/long tensor of
            the static adjacency over the full panel ticker order.
            The adapter retains a CPU copy (``register_buffer`` on
            float-cast adjacency) so it follows .to(device) calls.
    """

    def __init__(
        self,
        hp: RSRHyperparams,
        full_relation_graph: torch.Tensor,
    ) -> None:
        super().__init__()
        self.hp = hp
        if full_relation_graph.dim() != 2 or (
            full_relation_graph.size(0) != full_relation_graph.size(1)
        ):
            raise ValueError(
                f"full_relation_graph must be square (N, N); got "
                f"{tuple(full_relation_graph.shape)}"
            )
        # Buffer so device transitions follow the module.
        self.register_buffer(
            "adj_full",
            full_relation_graph.to(dtype=torch.float32).contiguous(),
            persistent=False,
        )
        self.lstm = StockLSTM(
            d_feat=hp.d_feat,
            hidden_size=hp.hidden_size,
            num_layers=hp.num_layers,
            dropout=hp.dropout,
        )
        self.gat = TemporalGraphAttention(
            hidden_size=hp.hidden_size,
            attn_hidden=hp.hidden_size,
            leaky_slope=hp.leaky_slope,
        )
        self.head = nn.Sequential(
            nn.Linear(2 * hp.hidden_size, hp.head_hidden),
            nn.GELU(),
            nn.Dropout(hp.dropout),
            nn.Linear(hp.head_hidden, 1),
        )

    def forward(
        self,
        x_window: torch.Tensor,
        active_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Score per active ticker on a single day.

        Args:
            x_window: (N_active, T, F) standardised lookback features
                for the active subset on the current day.
            active_idx: (N_active,) long tensor of indices into the
                full panel order. Used to slice the static adjacency
                so attention is computed over only the active subset.

        Returns:
            (N_active,) scalar scores.
        """
        e = self.lstm(x_window)                       # (A, H)
        adj_active = self.adj_full[active_idx][:, active_idx]
        g = self.gat(e, adj_active)                   # (A, H)
        cat = torch.cat([e, g], dim=-1)               # (A, 2H)
        y_hat = self.head(cat).squeeze(-1)            # (A,)
        return y_hat


__all__ = ["RSRAdapter", "RSRHyperparams"]
