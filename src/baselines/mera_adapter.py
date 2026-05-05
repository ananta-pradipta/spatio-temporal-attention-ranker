"""Adapter around MERA (Liu et al., WWW Companion 2025) for our biotech panel.

MERA = a Transformer backbone, self-supervised masked-AE pre-training,
a per-sample retrieval module that fetches Top-N nearest neighbours
from a training-set pool of pre-trained embeddings, target-aware
attention aggregation of those neighbours' feature and label-class
embeddings, and a Sparse Mixture-of-Experts (M=4, K=1) where the gate
input is the aggregated label embedding.

Our adaptation for the v2 biotech-244 panel:

  - Each (day, ticker) pair is one MERA "sample".
  - Per-sample input window: ``(B, T=20, F=22)`` of standardised
    enriched panel features (built by ``v2_runner.standardize_features``).
  - Backbone: 2-layer Transformer encoder, ``d_model=128``, 4 heads,
    GELU, pre-LN. Last timestep is the per-sample embedding e^s_t.
  - Retrieval pool stores (training-day, training-ticker) pairs from
    the train fold. Each entry has the pre-trained backbone embedding
    plus a discretised (B=10 quantiles) label class. The pool is built
    once after Phase-1 pre-training and respects the v2 5-day embargo
    (entries from days within 5 days of any test/val day are excluded
    when serving that split's queries; in practice we just exclude
    entries whose source day is inside the train-fold range).
  - Retrieval is K-nearest-neighbour by Euclidean distance (paper says
    MSE distance; equivalent for unit-norm features, cheaper without
    normalisation, so we use raw Euclidean). We use FAISS on CPU for
    scalability; the pool size is ~244 * 990 train days ~= 240K which
    a flat L2 FAISS index handles in < 50ms per (day, batch) query.
  - SMoE: M=4 experts, K=1 active per sample, each expert a small GRU
    over the per-sample encoder sequence (paper text says "small GRU
    per expert"). Gate input = aggregated label embedding l^s_t.
  - Final scalar score:
        h = SMoE(encoder_seq, gate_inp=l^s_t)
        y_hat = MLP([h ; e^s_t ; r^s_t])
    where the final MLP is a 2-layer GELU MLP -> scalar (residual on
    e^s_t through concat).

The adapter mirrors ``factorvae_adapter.FactorVAEAdapter`` in shape so
the trainer is a near-clone of ``train_factorvae_v2.py``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from src.baselines.vendored.mera import (
    MERABackbone,
    MERAMaskedAEHead,
    RetrievalAttentionAggregator,
    SparseGRUMoE,
)


@dataclass
class MERAHyperparams:
    """MERA architecture knobs.

    Defaults match the WWW 2025 paper's CSI 300/500/1000 reported
    configuration where stated, otherwise match the upstream
    ``model_moe_attn.Transformer`` defaults.
    """
    d_feat: int = 22                  # F: number of panel features per ticker
    context_window: int = 20          # T: lookback length
    d_model: int = 128                # paper default
    num_layers: int = 2               # paper default
    num_heads: int = 4                # paper default at d_model=128
    dropout: float = 0.1
    # Retrieval.
    n_classes: int = 10               # B: label-quantile bin count (paper)
    top_n: int = 10                   # paper default
    d_label: int = 16                 # gate-input width (upstream gate_dim)
    # SMoE.
    num_experts: int = 4              # M (paper default)
    top_k: int = 1                    # K (paper default)
    expert_hidden: int = 64           # GRU width (~d_model/2)
    # Masked-AE.
    mask_ratio: float = 0.5           # phase-1 random-mask fraction


class MERAAdapter(nn.Module):
    """MERA wrapped for our (N_active, T, F) panel format.

    Public surface used by ``train_mera_v2.py``:

        forward(x_window, retrieved_feat, retrieved_class)
            -> (N_active,) scalar score
        masked_ae_loss(x_window)
            -> (loss, mask_bt) for Phase-1 pre-training
        backbone_embed(x_window)
            -> (N_active, d_model) frozen e^s_t for retrieval queries
            (and, during pool construction, for keys).

    Internals mirror upstream MERA + the paper text:
        e^s_t = backbone(x)[:, -1]
        r^s_t, l^s_t = aggregator(e^s_t, retrieved_feat, retrieved_class)
        h = smoe(encoder_seq, gate_inp=l^s_t)
        y_hat = predict_mlp(cat(h, e^s_t, r^s_t))
    """

    def __init__(self, hp: MERAHyperparams):
        super().__init__()
        self.hp = hp
        self.backbone = MERABackbone(
            input_size=hp.d_feat,
            hidden_size=hp.d_model,
            num_layers=hp.num_layers,
            num_heads=hp.num_heads,
            dropout=hp.dropout,
        )
        self.mae_head = MERAMaskedAEHead(self.backbone, mask_ratio=hp.mask_ratio)
        self.aggregator = RetrievalAttentionAggregator(
            d_model=hp.d_model,
            n_classes=hp.n_classes,
            d_label=hp.d_label,
        )
        self.smoe = SparseGRUMoE(
            d_model=hp.d_model,
            d_label=hp.d_label,
            num_experts=hp.num_experts,
            top_k=hp.top_k,
            expert_hidden=hp.expert_hidden,
        )
        # Predict head: residual concat -> 2-layer MLP -> scalar.
        head_in = hp.d_model * 3
        self.predict_head = nn.Sequential(
            nn.Linear(head_in, hp.d_model),
            nn.GELU(),
            nn.Dropout(hp.dropout),
            nn.Linear(hp.d_model, 1),
        )

    # -- Phase-1 pre-training --------------------------------------------------

    def masked_ae_loss(self, x_window: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Phase-1 self-supervised reconstruction loss.

        Args:
            x_window: (B, T, F).
        Returns:
            (loss, mask_bt) where loss is a scalar MSE over masked
            positions and mask_bt is the (B, T) bool mask used.
        """
        return self.mae_head(x_window)

    # -- Embedding extraction (for retrieval pool) -----------------------------

    @torch.no_grad()
    def backbone_embed(self, x_window: torch.Tensor) -> torch.Tensor:
        """Per-sample target embedding e^s_t under the current backbone.

        Args:
            x_window: (B, T, F).
        Returns:
            (B, d_model) embedding from the encoder's last timestep.

        Used both to build the retrieval pool keys and to extract the
        query embedding at inference. Wrapped in ``no_grad`` because
        Phase 2 freezes the backbone.
        """
        was_training = self.backbone.training
        self.backbone.eval()
        z = self.backbone(x_window)
        self.backbone.train(was_training)
        return z

    # -- Phase-2 forward (retrieval-aware score) -------------------------------

    def forward(
        self,
        x_window: torch.Tensor,
        retrieved_feat: torch.Tensor,
        retrieved_class: torch.Tensor,
    ) -> torch.Tensor:
        """Score per active ticker given a precomputed retrieval batch.

        Args:
            x_window: (B, T, F).
            retrieved_feat: (B, K, d_model). Feature embeddings of the
                Top-N nearest neighbours from the retrieval pool, in
                d_model space (i.e. backbone output dim).
            retrieved_class: (B, K) long. Discretised label-class
                indices in [0, n_classes).

        Returns:
            y_hat: (B,) scalar scores (raw, not z-scored).
        """
        # Encoder sequence under the (typically frozen) backbone.
        z_seq = self.backbone.forward_seq(x_window)        # (B, T, d_model)
        e_st = z_seq[:, -1, :]                             # (B, d_model)

        r_st, l_st = self.aggregator(e_st, retrieved_feat, retrieved_class)

        h = self.smoe(z_seq, gate_inp=l_st)                # (B, d_model)

        # Residual concat: SMoE output + raw query + retrieval-aggregated.
        cat = torch.cat([h, e_st, r_st], dim=-1)           # (B, 3 * d_model)
        y_hat = self.predict_head(cat).squeeze(-1)         # (B,)
        return y_hat

    # -- Convenience: which params are trainable in Phase 2 --------------------

    def phase2_parameters(self):
        """Return parameters that should remain trainable in Phase 2.

        The paper says: freeze the pre-trained Transformer backbone,
        train the SMoE + retrieval-attention + predict head only.
        We also leave the aggregator's learned label-embedding table
        trainable (it is not part of the backbone).
        """
        for p in self.aggregator.parameters():
            yield p
        for p in self.smoe.parameters():
            yield p
        for p in self.predict_head.parameters():
            yield p


__all__ = ["MERAAdapter", "MERAHyperparams"]
