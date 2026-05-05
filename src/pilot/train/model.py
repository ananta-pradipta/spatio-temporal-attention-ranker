"""Temporal GAT model for stock ranking.

A simple baseline: GAT (Graph Attention Network) applied independently
to each temporal snapshot. This is not a true temporal GNN (like EvolveGCN
or TGN), but serves as a reasonable pilot baseline.

For the qualifying exam, the point is to show:
1. Graph structure captures cross-stock dependencies
2. The ranking formulation works
3. The pipeline is correct end-to-end

A proper temporal architecture (EvolveGCN, TGAT, TGN) can be swapped in later.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class TemporalGAT(nn.Module):
    """GAT-based model for per-snapshot stock ranking.

    Architecture:
        Input features -> GAT layers -> MLP head -> scalar score per node.
    The model produces a score for each stock; the ranking loss operates
    on these scores.

    Args:
        in_channels: Number of input node features.
        hidden_dim: Hidden dimension for GAT layers.
        num_heads: Number of attention heads.
        num_layers: Number of GAT layers.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.num_layers = num_layers

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # First GAT layer
        self.convs.append(
            GATConv(in_channels, hidden_dim, heads=num_heads, dropout=dropout)
        )
        self.norms.append(nn.LayerNorm(hidden_dim * num_heads))

        # Middle GAT layers
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(
                    hidden_dim * num_heads,
                    hidden_dim,
                    heads=num_heads,
                    dropout=dropout,
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim * num_heads))

        # Last GAT layer (single head for output)
        if num_layers > 1:
            self.convs.append(
                GATConv(
                    hidden_dim * num_heads,
                    hidden_dim,
                    heads=1,
                    concat=False,
                    dropout=dropout,
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim))

        # MLP scoring head
        final_dim = hidden_dim
        self.head = nn.Sequential(
            nn.Linear(final_dim, final_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(final_dim // 2, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass producing a score per node.

        Args:
            x: Node features [n_nodes, in_channels].
            edge_index: Edge indices [2, n_edges].

        Returns:
            Scores tensor [n_nodes] (unnormalized ranking scores).
        """
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x = conv(x, edge_index)
            x = norm(x)
            if i < len(self.convs) - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        scores = self.head(x).squeeze(-1)  # [n_nodes]
        return scores


class MLPBaseline(nn.Module):
    """MLP baseline: no graph, no temporal. Each stock scored independently.

    Args:
        in_channels: Number of input features.
        hidden_dim: Hidden layer size.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 64,
        dropout: float = 0.3,
        **kwargs,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass. edge_index is ignored (no graph structure used).

        Args:
            x: Node features [n_nodes, in_channels].
            edge_index: Ignored.

        Returns:
            Scores [n_nodes].
        """
        return self.net(x).squeeze(-1)


class LSTMBaseline(nn.Module):
    """LSTM baseline: temporal but no graph. Processes a window of features per stock.

    Since this pilot uses single-snapshot input (no lookback window in the
    data loader), this acts as a 1-step LSTM, which is essentially equivalent
    to an MLP with hidden state. For a proper temporal baseline, the data
    loader should provide a sequence of snapshots per training step.

    For the pilot, this serves as a slightly different non-graph architecture
    to confirm that GAT's advantage (if any) comes from graph structure.

    Args:
        in_channels: Number of input features per timestep.
        hidden_dim: LSTM hidden size.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 64,
        dropout: float = 0.3,
        **kwargs,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass. edge_index is ignored.

        Args:
            x: Node features [n_nodes, in_channels].
            edge_index: Ignored.

        Returns:
            Scores [n_nodes].
        """
        # Treat as sequence of length 1: [n_nodes, 1, in_channels]
        x_seq = x.unsqueeze(1)
        lstm_out, _ = self.lstm(x_seq)  # [n_nodes, 1, hidden_dim]
        h = lstm_out[:, -1, :]  # [n_nodes, hidden_dim]
        return self.head(h).squeeze(-1)


def build_model(model_name: str, in_channels: int, cfg: dict) -> nn.Module:
    """Factory function to create a model by name.

    Args:
        model_name: One of "TemporalGAT", "MLP", "LSTM".
        in_channels: Number of input features.
        cfg: Model config dict (hidden_dim, num_heads, etc.).

    Returns:
        Instantiated model.
    """
    if model_name == "TemporalGAT":
        return TemporalGAT(
            in_channels=in_channels,
            hidden_dim=cfg["hidden_dim"],
            num_heads=cfg.get("num_heads", 4),
            num_layers=cfg.get("num_layers", 2),
            dropout=cfg.get("dropout", 0.3),
        )
    elif model_name == "MLP":
        return MLPBaseline(
            in_channels=in_channels,
            hidden_dim=cfg["hidden_dim"],
            dropout=cfg.get("dropout", 0.3),
        )
    elif model_name == "LSTM":
        return LSTMBaseline(
            in_channels=in_channels,
            hidden_dim=cfg["hidden_dim"],
            dropout=cfg.get("dropout", 0.3),
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


class ListNetLoss(nn.Module):
    """ListNet ranking loss (top-1 approximation).

    Computes KL divergence between the predicted score distribution
    (softmax of scores) and the target distribution (softmax of relevance
    labels, here normalized rank scores).

    Reference:
        Cao et al. (2007). "Learning to Rank: From Pairwise Approach to
        Listwise Approach." ICML.

    Args:
        temperature: Softmax temperature for smoothing.
    """

    def __init__(self, temperature: float = 1.0) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        scores: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute ListNet loss.

        Args:
            scores: Predicted scores [n_nodes].
            targets: Target relevance (normalized ranks) [n_nodes].
            mask: Optional boolean mask for valid nodes.

        Returns:
            Scalar loss value.
        """
        if mask is not None:
            scores = scores[mask]
            targets = targets[mask]

        if len(scores) < 2:
            return torch.tensor(0.0, device=scores.device, requires_grad=True)

        # Softmax distributions
        pred_dist = F.softmax(scores / self.temperature, dim=0)
        target_dist = F.softmax(targets / self.temperature, dim=0)

        # Cross-entropy (equivalent to KL up to constant)
        loss = -torch.sum(target_dist * torch.log(pred_dist + 1e-10))
        return loss


class PairwiseRankingLoss(nn.Module):
    """Pairwise margin ranking loss for stock ranking.

    For all pairs (i, j) where target_i > target_j, we want score_i > score_j.

    Args:
        margin: Margin for the ranking loss.
    """

    def __init__(self, margin: float = 0.1) -> None:
        super().__init__()
        self.margin = margin

    def forward(
        self,
        scores: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute pairwise ranking loss.

        Args:
            scores: Predicted scores [n_nodes].
            targets: Target relevance [n_nodes].
            mask: Optional boolean mask.

        Returns:
            Scalar loss.
        """
        if mask is not None:
            scores = scores[mask]
            targets = targets[mask]

        n = len(scores)
        if n < 2:
            return torch.tensor(0.0, device=scores.device, requires_grad=True)

        # Efficient pairwise computation
        score_diff = scores.unsqueeze(0) - scores.unsqueeze(1)  # [n, n]
        target_diff = targets.unsqueeze(0) - targets.unsqueeze(1)  # [n, n]
        target_sign = torch.sign(target_diff)

        # Hinge loss: max(0, margin - sign * score_diff)
        losses = F.relu(self.margin - target_sign * score_diff)

        # Only count pairs where targets differ
        pair_mask = (target_diff.abs() > 1e-6).float()
        loss = (losses * pair_mask).sum() / (pair_mask.sum() + 1e-10)
        return loss
