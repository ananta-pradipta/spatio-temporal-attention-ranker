"""G²-STAR: Graph-Gated STAR.

Two parallel forward paths plus a regime-conditional gate:
  graph_path   = pure STAR over (N+1, W, F) patches.
  nograph_path = per-ticker transformer over (W, F) windows; no
                 graph neighbors. (Iter 1 is purely univariate; iter 2
                 may add cross-stock mixing.)
  gate         = sigmoid(MLP(regime_signature)) → α ∈ [0, 1].
  output       = α · y_graph + (1 − α) · y_nograph (per ticker).

Cross-sectional layer norm + cross-sectional MSE loss as in pure
STAR. FiLM and aux loss disabled (falsified in earlier work).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from src.mtgn.model.layers.cs_layer_norm import CrossSectionalLayerNorm


@dataclass
class G2Config:
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
    signature_dim: int = 4
    gate_hidden: int = 16
    gate_entropy_weight: float = 0.001  # encourages α to stay between 0 and 1
    # Iter 2: StockMixer-style cross-stock MLP in the no-graph path
    use_xstock_mlp: bool = False
    xstock_hidden: int = 20
    num_stocks: int = 84
    # Iter 2: distillation aux loss weight (gradient signal for the gate)
    gate_distill_weight: float = 0.0
    # Iter 5 (Fix A): temperature-scaled sigmoid to widen α output range
    gate_temperature: float = 1.0
    # Iter 7: per-ticker gate (α(t, i) instead of α(t))
    per_ticker_gate: bool = False
    per_ticker_gate_hidden: int = 32
    # Iter 8: event-memory token prepended to graph path
    num_event_tokens: int = 0           # K event-cluster embedding bank size; 0 = off


class _CrossStockMixer(nn.Module):
    """StockMixer-style MLP across the stock axis (no learned graph)."""
    def __init__(self, num_stocks: int, hidden: int = 20):
        super().__init__()
        self.ln = nn.LayerNorm(num_stocks)
        self.dense1 = nn.Linear(num_stocks, hidden)
        self.act = nn.Hardswish()
        self.dense2 = nn.Linear(hidden, num_stocks)

    def forward(self, z_full: Tensor) -> Tensor:
        # z_full: [N, D]  →  [N, D] mixed across the N axis.
        x = z_full.permute(1, 0)              # [D, N]
        x = self.ln(x)
        x = self.dense1(x)
        x = self.act(x)
        x = self.dense2(x)                    # [D, N]
        return x.permute(1, 0)                # [N, D]


class _PerTickerEncoder(nn.Module):
    """Univariate transformer over a single ticker's (W, F) window."""
    def __init__(self, cfg: G2Config):
        super().__init__()
        D = cfg.hidden_dim
        self.input_proj = nn.Linear(cfg.feature_dim, D)
        self.temporal_pe = nn.Embedding(cfg.temporal_window, D)
        layer = nn.TransformerEncoderLayer(
            d_model=D, nhead=cfg.num_heads,
            dim_feedforward=cfg.ff_dim, dropout=cfg.transformer_dropout,
            activation="gelu", batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=cfg.num_layers)

    def forward(self, x: Tensor) -> Tensor:
        # x: [A, W, F]  →  pooled per-ticker representation [A, D]
        D = self.input_proj.out_features
        h = self.input_proj(x)                                       # [A, W, D]
        tp = self.temporal_pe(torch.arange(x.shape[1], device=x.device))
        h = h + tp.view(1, -1, D)
        z = self.transformer(h)                                      # [A, W, D]
        return z[:, -1, :]                                           # last-time pooled


class _GraphPathEncoder(nn.Module):
    """Pure-STAR-style encoder over (N+1, W, F) patches."""
    def __init__(self, cfg: G2Config):
        super().__init__()
        D = cfg.hidden_dim
        NP1 = cfg.num_neighbors + 1
        self.input_proj = nn.Linear(cfg.feature_dim, D)
        self.spatial_pe = nn.Embedding(NP1, D)
        self.temporal_pe = nn.Embedding(cfg.temporal_window, D)
        layer = nn.TransformerEncoderLayer(
            d_model=D, nhead=cfg.num_heads,
            dim_feedforward=cfg.ff_dim, dropout=cfg.transformer_dropout,
            activation="gelu", batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=cfg.num_layers)

    def forward(self, patches: Tensor, patch_mask: Tensor,
                extra_tokens: Tensor | None = None) -> Tensor:
        # patches: [A, N+1, W, F], patch_mask: [A, N+1, W] bool valid
        # extra_tokens: optional [K, D] context tokens prepended to the sequence
        A, NP1, W, F = patches.shape
        D = self.input_proj.out_features
        x = self.input_proj(patches)
        sp = self.spatial_pe(torch.arange(NP1, device=x.device))
        tp = self.temporal_pe(torch.arange(W, device=x.device))
        x = x + sp.view(1, NP1, 1, D) + tp.view(1, 1, W, D)
        x_flat = x.reshape(A, NP1 * W, D)
        mask_flat = patch_mask.reshape(A, NP1 * W)
        prefix_len = 0
        if extra_tokens is not None and extra_tokens.shape[0] > 0:
            K = extra_tokens.shape[0]
            extras = extra_tokens.unsqueeze(0).expand(A, K, D)
            x_flat = torch.cat([extras, x_flat], dim=1)
            ext_valid = torch.ones(A, K, dtype=mask_flat.dtype, device=mask_flat.device)
            mask_flat = torch.cat([ext_valid, mask_flat], dim=1)
            prefix_len = K
        key_pad = ~mask_flat
        z = self.transformer(x_flat, src_key_padding_mask=key_pad)
        # Self-ticker today position: row 0, time W-1 → flat index (prefix_len + W - 1)
        return z[:, prefix_len + (W - 1), :]                         # [A, D]


class GraphGatedSTAR(nn.Module):
    def __init__(self, cfg: G2Config):
        super().__init__()
        self.cfg = cfg
        D = cfg.hidden_dim
        self.graph_enc = _GraphPathEncoder(cfg)
        self.nograph_enc = _PerTickerEncoder(cfg)
        if cfg.use_xstock_mlp:
            self.xstock_mixer = _CrossStockMixer(cfg.num_stocks, cfg.xstock_hidden)
        if cfg.num_event_tokens > 0:
            # Iter 8: learnable event-cluster embedding bank.
            # On day t, weighted sum (by event-similarity weights) is a single
            # context token prepended to the graph path's transformer input.
            self.event_bank = nn.Parameter(
                torch.randn(cfg.num_event_tokens, cfg.hidden_dim) * 0.02
            )
        # Gate
        if cfg.per_ticker_gate:
            # Iter 7: per-ticker gate. Input = regime signature + per-ticker
            # embedding (from no-graph path's pooled z_n). Output = α per ticker.
            self.gate_mlp = nn.Sequential(
                nn.Linear(cfg.signature_dim + cfg.hidden_dim, cfg.per_ticker_gate_hidden),
                nn.GELU(),
                nn.Linear(cfg.per_ticker_gate_hidden, 1),
            )
        else:
            self.gate_mlp = nn.Sequential(
                nn.Linear(cfg.signature_dim, cfg.gate_hidden), nn.GELU(),
                nn.Linear(cfg.gate_hidden, 1),
            )
        # Bias init at 0 → sigmoid(0) = 0.5 (use both paths at start)
        nn.init.zeros_(self.gate_mlp[-1].bias)
        # Heads (separate per path → cross-sectional layer norm shared)
        self.cs_ln = CrossSectionalLayerNorm(D)
        self.rank_head_graph = nn.Sequential(
            nn.Linear(D, cfg.head_hidden), nn.GELU(),
            nn.Dropout(cfg.head_dropout),
            nn.Linear(cfg.head_hidden, 1),
        )
        self.rank_head_nograph = nn.Sequential(
            nn.Linear(D, cfg.head_hidden), nn.GELU(),
            nn.Dropout(cfg.head_dropout),
            nn.Linear(cfg.head_hidden, 1),
        )

    def forward_day(self, patches: Tensor, patch_mask: Tensor,
                    self_window: Tensor, regime_sig: Tensor,
                    active_mask: Tensor,
                    event_weights: Tensor | None = None) -> dict[str, Tensor]:
        """
        patches:     [A, N+1, W, F]   patch_mask: [A, N+1, W]  graph path input
        self_window: [A, W, F]         no-graph path input (= patches[:, 0, :, :])
        regime_sig:  [signature_dim]   gate input (causal)
        active_mask: [num_nodes]
        """
        cfg = self.cfg
        D = cfg.hidden_dim
        num_nodes = active_mask.shape[0]

        # Graph path (optionally with prepended event-context token)
        extra_tokens = None
        if cfg.num_event_tokens > 0 and event_weights is not None:
            # Weighted sum of event bank: [K, D] × [K] → [D] → [1, D]
            evt = (event_weights.view(-1, 1) * self.event_bank).sum(dim=0)  # [D]
            extra_tokens = evt.unsqueeze(0)                                  # [1, D]
        z_g = self.graph_enc(patches, patch_mask, extra_tokens=extra_tokens)  # [A, D]
        z_n = self.nograph_enc(self_window)                          # [A, D]

        # Scatter to full-N for both paths
        z_g_full = torch.zeros(num_nodes, D, device=z_g.device, dtype=z_g.dtype)
        z_n_full = torch.zeros(num_nodes, D, device=z_n.device, dtype=z_n.dtype)
        z_g_full[active_mask] = z_g
        z_n_full[active_mask] = z_n

        # Iter 2: cross-stock MLP mixes information across active tickers in
        # the no-graph path WITHOUT a learned graph (StockMixer-style).
        if cfg.use_xstock_mlp:
            z_n_full = z_n_full + self.xstock_mixer(z_n_full)

        z_g_norm = self.cs_ln(z_g_full, active_mask)
        z_n_norm = self.cs_ln(z_n_full, active_mask)

        y_graph = self.rank_head_graph(z_g_norm).squeeze(-1)          # [num_nodes]
        y_nograph = self.rank_head_nograph(z_n_norm).squeeze(-1)      # [num_nodes]

        # Gate (per-day market-level OR per-ticker, iter 7)
        if cfg.per_ticker_gate:
            S = regime_sig.shape[0]
            sig_b = regime_sig.view(1, S).expand(num_nodes, S)             # [N, S]
            gate_in = torch.cat([sig_b, z_n_full], dim=1)                  # [N, S+D]
            logits = self.gate_mlp(gate_in).squeeze(-1)                     # [N]
            alpha_per = torch.sigmoid(logits / cfg.gate_temperature)        # [N]
            y_hat = alpha_per * y_graph + (1.0 - alpha_per) * y_nograph
            alpha_summary = alpha_per[active_mask].mean()
            return {
                "y_hat": y_hat,
                "y_graph": y_graph,
                "y_nograph": y_nograph,
                "alpha": alpha_summary,
                "alpha_per_ticker": alpha_per,
            }
        else:
            alpha = torch.sigmoid(self.gate_mlp(regime_sig) / cfg.gate_temperature).squeeze(-1)
            y_hat = alpha * y_graph + (1.0 - alpha) * y_nograph
            return {
                "y_hat": y_hat,
                "y_graph": y_graph,
                "y_nograph": y_nograph,
                "alpha": alpha,
            }

    def gate_entropy(self, alpha: Tensor) -> Tensor:
        """Binary entropy of the gate output: -α log α - (1-α) log(1-α).
        Maximized when α = 0.5; encourages the gate to stay away from
        collapse to 0 or 1 during training."""
        eps = 1e-6
        a = alpha.clamp(eps, 1 - eps)
        return -(a * a.log() + (1 - a) * (1 - a).log())


__all__ = ["GraphGatedSTAR", "G2Config"]
