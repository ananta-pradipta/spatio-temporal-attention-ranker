"""Core MERA architectural components, vendored and stripped.

Translated from ``chenchen1104/MERA/MERA/src/model_moe_attn.py`` with
the FastMoE / Qlib / TRA dependencies removed. The translation is
conservative: layer counts, hidden dims, attention shape, and gate
formulation match the upstream ``Transformer`` class and the WWW 2025
paper text.

Module map (input -> output):

    MERABackbone(x)                  : (B, T, F)         -> (B, d_model)
        BatchNorm1d on flattened F
        Linear F -> d_model
        PositionalEncoding (sinusoidal)
        2-layer TransformerEncoder, 4 heads, pre-LN, GELU
        Take the last timestep as the per-sample embedding e^s_t

    RetrievalAttentionAggregator(query, kv_features, kv_label_class)
        -> (aggregated_feature_emb r^s_t, aggregated_label_emb l^s_t)
        Matches upstream
            attn = softmax(query @ kv_features^T)
            l^s_t = attn @ embedding(kv_label_class)
        We additionally aggregate kv_features themselves with the same
        attention to form r^s_t; the upstream code uses the raw last
        feature instead, but the paper text the user supplied
        explicitly mentions "aggregated feature embedding r^s_t" so we
        produce it.

    SparseGRUMoE(query_embed, gate_embed)
        -> (B, d_model)
        Top-K=1 sparse routing of GateNet(label_emb)
        Each expert is a 1-layer GRU run on the per-sample sequence
        seeded with query_embed. The paper says small GRU per expert;
        we keep the GRU width = d_model // 2 and re-project to d_model.

    MERAMaskedAEHead(z_seq, mask)
        -> reconstruction_loss
        Linear(d_model -> F) decoder on each timestep.

The full forward of the adapter combines:

    e^s_t   = MERABackbone(x)
    r^s_t, l^s_t = RetrievalAttentionAggregator(e^s_t, retrieved_feats,
                                                 retrieved_label_class)
    h       = SparseGRUMoE(query_embed=cat(e^s_t, r^s_t),
                            gate_embed=l^s_t)
    y_hat   = MLP(h + e^s_t)        # residual

This file holds only the nn.Modules; the retrieval pool builder, the
two-phase training loop, and the (day, ticker)-batching live in
``src.baselines.train_mera_v2``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Positional encoding (verbatim shape match with upstream Transformer block).
# ---------------------------------------------------------------------------


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding over a (T, B, d) sequence.

    Mirrors upstream ``model_moe_attn.PositionalEncoding`` (which itself
    is the standard PyTorch tutorial implementation). Buffered so the
    encoding moves to the same device as the parent module.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_len, 1, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Backbone Transformer encoder.
# ---------------------------------------------------------------------------


class MERABackbone(nn.Module):
    """2-layer Transformer encoder over (B, T, F) -> (B, d_model).

    Args:
        input_size: F, panel feature count (22 for the v2 panel).
        hidden_size: d_model (paper default: 128).
        num_layers: number of TransformerEncoderLayer blocks (paper: 2).
        num_heads: attention heads (paper: 4 at d_model=128).
        dropout: residual + attn dropout.

    Output:
        Per-sample embedding ``e^s_t`` taken at the last timestep.
        Also exposes the full sequence via ``forward_seq`` for the
        masked-AE head.
    """

    def __init__(
        self,
        input_size: int = 22,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bn = nn.BatchNorm1d(input_size)
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.pe = PositionalEncoding(hidden_size, dropout=dropout)
        layers = []
        for _ in range(num_layers):
            layers.append(
                nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    nhead=num_heads,
                    dim_feedforward=hidden_size * 4,
                    dropout=dropout,
                    activation="gelu",
                    norm_first=True,
                    batch_first=False,
                )
            )
        self.encoder = nn.Sequential(*layers)

    def forward_seq(self, x: torch.Tensor) -> torch.Tensor:
        """Return the full per-timestep encoder sequence, (B, T, d_model).

        Used by the masked-AE pre-training head.
        """
        # x: (B, T, F); BatchNorm1d expects (N, F).
        bsz, T, F_ = x.shape
        x = x.reshape(-1, F_)
        x = self.bn(x)
        x = x.reshape(bsz, T, F_)
        x = self.input_proj(x)         # (B, T, d_model)
        x = x.transpose(0, 1)          # (T, B, d_model) for nn.Transformer
        x = self.pe(x)
        x = self.encoder(x)            # (T, B, d_model)
        x = x.transpose(0, 1)          # (B, T, d_model)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample embedding from the last timestep, (B, d_model)."""
        z_seq = self.forward_seq(x)
        return z_seq[:, -1, :]


# ---------------------------------------------------------------------------
# Masked-autoencoder pre-training head.
# ---------------------------------------------------------------------------


class MERAMaskedAEHead(nn.Module):
    """Masked-AE reconstruction head for Phase-1 self-supervised pre-training.

    The paper specifies "masked autoencoder on raw time-series (random
    mask ratio of input tokens, MSE reconstruction loss)". Following
    BERT/MAE conventions:

      - randomly mask a fraction ``mask_ratio`` of the input timesteps;
      - replace masked timesteps with a learnable mask token;
      - run the MERABackbone on the masked input;
      - decode every timestep back to the original ``input_size`` dim
        with a single linear layer;
      - MSE only on the masked positions.

    Args:
        backbone: a MERABackbone instance whose weights we co-train.
        mask_ratio: fraction of timesteps masked per sample (paper says
            random; we default to 0.5 per the user's task brief).
    """

    def __init__(self, backbone: MERABackbone, mask_ratio: float = 0.5) -> None:
        super().__init__()
        self.backbone = backbone
        self.mask_ratio = float(mask_ratio)
        # Mask token lives in the *projected* d_model space, applied
        # after input_proj. Using a projected-space mask token avoids
        # double-counting the input BatchNorm statistics on masked rows.
        self.mask_token = nn.Parameter(torch.zeros(backbone.hidden_size))
        nn.init.normal_(self.mask_token, mean=0.0, std=0.02)
        self.decoder = nn.Linear(backbone.hidden_size, backbone.input_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the masked-AE reconstruction loss.

        Args:
            x: (B, T, F) raw (pre-standardisation already applied
                upstream) feature window.

        Returns:
            (loss, mask_b_t): scalar MSE loss over masked positions and
            the (B, T) bool mask used (1 for masked, 0 for kept).
        """
        bsz, T, F_ = x.shape
        device = x.device
        # 1. Sample a per-sample, per-timestep Bernoulli mask. Match
        #    paper: fraction = mask_ratio independent across timesteps.
        mask_bt = (torch.rand(bsz, T, device=device) < self.mask_ratio)  # bool
        # If a sample happens to have zero masked tokens (low prob) the
        # gradient is fine; we just clamp the loss to zero in that case.

        # 2. Run the BatchNorm + input projection ourselves so we can
        #    splice in the mask token in d_model space.
        x_flat = x.reshape(-1, F_)
        x_flat = self.backbone.bn(x_flat)
        x_proj = self.backbone.input_proj(x_flat).reshape(bsz, T, self.backbone.hidden_size)

        mask_token = self.mask_token.to(x_proj.dtype).view(1, 1, -1).expand(bsz, T, -1)
        x_proj = torch.where(mask_bt.unsqueeze(-1), mask_token, x_proj)

        # 3. Positional encoding + encoder.
        x_seq = x_proj.transpose(0, 1)         # (T, B, d_model)
        x_seq = self.backbone.pe(x_seq)
        x_seq = self.backbone.encoder(x_seq)   # (T, B, d_model)
        x_seq = x_seq.transpose(0, 1)          # (B, T, d_model)

        # 4. Per-timestep linear decode back to F.
        x_rec = self.decoder(x_seq)            # (B, T, F)

        # 5. MSE only on masked positions.
        if mask_bt.any():
            diff = (x_rec - x) ** 2            # (B, T, F)
            # Reduce over F first, then average over masked (B, T) cells.
            mse_per_token = diff.mean(dim=-1)  # (B, T)
            loss = mse_per_token[mask_bt].mean()
        else:
            loss = torch.zeros((), device=device, dtype=x_rec.dtype)
        return loss, mask_bt


# ---------------------------------------------------------------------------
# Retrieval-attention aggregator (target-aware).
# ---------------------------------------------------------------------------


class RetrievalAttentionAggregator(nn.Module):
    """Aggregate retrieved neighbours via target-aware (query) attention.

    Mirrors the upstream block:

        attn = F.softmax(query @ kv_features^T, dim=-1)        # (B, 1, K)
        l    = attn @ label_class_embedding                    # (B, 1, d)

    where ``query`` is the per-sample Transformer last-step embedding
    ``e^s_t``, ``kv_features`` are the retrieved neighbours' high-level
    embeddings, and ``label_class_embedding`` looks up a learned
    embedding table indexed by the discretised label class.

    Args:
        d_model: backbone hidden size.
        n_classes: number of label-quantile bins B (paper default 10).
        d_label: label-embedding width (paper uses gate_dim=16; we
            default to 16 to match the upstream code).
    """

    def __init__(self, d_model: int = 128, n_classes: int = 10, d_label: int = 16) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_classes = n_classes
        self.d_label = d_label
        self.label_embedding = nn.Embedding(n_classes, d_label)

    def forward(
        self,
        query: torch.Tensor,
        kv_features: torch.Tensor,
        kv_label_class: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run target-aware attention.

        Args:
            query: (B, d_model). Per-sample target embedding e^s_t.
            kv_features: (B, K, d_model). Retrieved neighbours'
                feature embeddings.
            kv_label_class: (B, K) long. Discretised label class index
                for each retrieved neighbour, in [0, n_classes).

        Returns:
            r_st: (B, d_model). Aggregated feature embedding (paper
                term r^s_t).
            l_st: (B, d_label). Aggregated label embedding (paper term
                l^s_t, fed to GateNet).
        """
        # (B, 1, K) attention weights from query.kv similarity.
        scores = torch.bmm(query.unsqueeze(1), kv_features.transpose(1, 2))
        attn = F.softmax(scores, dim=-1)                      # (B, 1, K)
        r_st = torch.bmm(attn, kv_features).squeeze(1)        # (B, d_model)
        label_emb = self.label_embedding(kv_label_class)      # (B, K, d_label)
        l_st = torch.bmm(attn, label_emb).squeeze(1)          # (B, d_label)
        return r_st, l_st


# ---------------------------------------------------------------------------
# Sparse Mixture-of-(GRU)-Experts.
# ---------------------------------------------------------------------------


class _GRUExpert(nn.Module):
    """Small per-expert GRU. Per the paper text, each expert is a small GRU."""

    def __init__(self, d_model: int, expert_hidden: int) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=expert_hidden,
            num_layers=1,
            batch_first=True,
        )
        self.proj = nn.Linear(expert_hidden, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, S, d_model) -> (B, d_model). Returns the last hidden state."""
        out, _ = self.gru(x)
        last = out[:, -1, :]               # (B, expert_hidden)
        return self.proj(last)             # (B, d_model)


class SparseGRUMoE(nn.Module):
    """SMoE with M experts (each a small GRU) and Top-K=1 routing.

    GateNet (per paper):
        gate_logits = Linear(d_label, M)(label_emb)
        weights, idx = TopK(softmax(gate_logits), k=top_k)

    The paper specifies K=1 activation. We implement it with a simple
    per-sample dispatch: for each sample, run the selected expert on
    that sample's input. This is O(B * expert_cost) and keeps the code
    portable (no FastMoE dependency).

    Args:
        d_model: backbone hidden size.
        d_label: width of the gate input (label embedding).
        num_experts: M (paper default 4).
        top_k: K (paper default 1).
        expert_hidden: per-expert GRU hidden width. Defaults to
            d_model // 2 to match "small GRU".
    """

    def __init__(
        self,
        d_model: int = 128,
        d_label: int = 16,
        num_experts: int = 4,
        top_k: int = 1,
        expert_hidden: int | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        eh = int(expert_hidden) if expert_hidden is not None else max(8, d_model // 2)
        self.experts = nn.ModuleList(
            [_GRUExpert(d_model, eh) for _ in range(self.num_experts)]
        )
        self.gate = nn.Linear(d_label, self.num_experts)

    def forward(self, x_seq: torch.Tensor, gate_inp: torch.Tensor) -> torch.Tensor:
        """Run the SMoE.

        Args:
            x_seq: (B, S, d_model). Per-sample sequence fed into the
                selected expert. We use the encoder's last few states
                (the trainer passes the full encoder output of length T
                here); the GRU takes its own last hidden state as the
                expert output.
            gate_inp: (B, d_label). Aggregated label embedding l^s_t.

        Returns:
            (B, d_model) expert-mixed output.
        """
        bsz = x_seq.size(0)
        device = x_seq.device

        gate_logits = self.gate(gate_inp)               # (B, M)
        gate_probs = F.softmax(gate_logits, dim=-1)     # (B, M)
        top_w, top_idx = gate_probs.topk(self.top_k, dim=-1)  # (B, K)
        # Renormalise over the chosen k.
        top_w = top_w / (top_w.sum(dim=-1, keepdim=True) + 1e-9)

        out = torch.zeros(bsz, self.d_model, device=device, dtype=x_seq.dtype)
        # Per-expert pass: select the rows routed to this expert in
        # any of their top-K slots, run a single forward, scatter back.
        for e in range(self.num_experts):
            # which (b, k) slots picked expert e
            mask = (top_idx == e)                        # (B, K) bool
            if not mask.any():
                continue
            row_mask = mask.any(dim=-1)                  # (B,) which samples used it at all
            rows = torch.nonzero(row_mask, as_tuple=False).squeeze(-1)
            if rows.numel() == 0:
                continue
            x_sub = x_seq.index_select(0, rows)          # (b', S, d_model)
            y_sub = self.experts[e](x_sub)               # (b', d_model)
            # Sum the per-slot weights this expert got for each row.
            w_sub = (top_w * mask.float()).sum(dim=-1)   # (B,)
            w_sub = w_sub.index_select(0, rows).unsqueeze(-1)  # (b', 1)
            out.index_add_(0, rows, y_sub * w_sub)

        return out


__all__ = [
    "MERABackbone",
    "MERAMaskedAEHead",
    "PositionalEncoding",
    "RetrievalAttentionAggregator",
    "SparseGRUMoE",
]
