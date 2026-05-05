"""REM-STAR: pure STAR + regime-context token.

Inherits pure STAR's transformer and heads. At forward time, an
additional input token carrying the day's retrieved regime signature
(projected into D-dim) is concatenated to the flattened patch
sequence. The regime token participates in self-attention but is not
read by the rank head.

FiLM and auxiliary-vol paths are preserved only as ablation switches;
for iteration 1 both are disabled, matching pure STAR baseline.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from src.investigation.regime_memory.corr_attention import (
    CorrBiasedEncoder, CorrBiasedEncoderLayer, expand_bias_for_heads,
)
from src.investigation.regime_memory.dann import RegimeDiscriminator, grad_reverse
from src.mtgn.model.layers.cs_layer_norm import CrossSectionalLayerNorm


@dataclass
class REMConfig:
    feature_dim: int = 22
    hidden_dim: int = 128
    num_neighbors: int = 8
    temporal_window: int = 20
    num_heads: int = 4
    num_layers: int = 2
    ff_dim: int = 256
    transformer_dropout: float = 0.1
    signature_dim: int = 4
    head_hidden: int = 64
    head_dropout: float = 0.2
    use_risk_head: bool = False            # REM disables by default (aux off)
    risk_quantiles: tuple[float, ...] = (0.05, 0.50, 0.95)
    use_regime_token: bool = True           # iter 1 default (one projected sig token)
    num_memory_tokens: int = 0              # iter 3A/C: M learnable tokens per cluster/prototype
    num_clusters: int = 4                   # size of memory bank's first dim
    num_prototypes: int = 0                 # iter 3B/C: L learnable prototypes; 0 = use k-means
    proto_temperature: float = 1.0          # softmax temperature over -‖s - P‖²
    use_rl_proto: bool = False              # iter 9: REINFORCE sampling instead of softmax
    # Proposal A: correlation-aware attention bias. alpha > 0 enables a
    # per-batch attention-logit penalty proportional to pairwise rolling
    # correlation between ego and neighbors, diffusing attention during
    # correlation-spike regimes.
    corr_bias_alpha: float = 0.0
    # Proposal A learnable gate: if > 0, wraps alpha with a sigmoid(MLP
    # over signature) so the model learns when the bias is helpful and
    # when to suppress it. 0 = constant alpha.
    corr_bias_gate_hidden: int = 0
    # Proposal C: domain-adversarial regime invariance. dann_lambda_max > 0
    # enables; dann_hidden is the discriminator MLP hidden size; dann_classes
    # matches the number of regime clusters in the catalog.
    dann_lambda_max: float = 0.0
    dann_hidden: int = 64
    dann_classes: int = 4


class REMStar(nn.Module):
    def __init__(self, cfg: REMConfig):
        super().__init__()
        self.cfg = cfg
        NP1 = cfg.num_neighbors + 1
        D = cfg.hidden_dim

        self.input_proj = nn.Linear(cfg.feature_dim, D)
        self.spatial_pe = nn.Embedding(NP1, D)
        self.temporal_pe = nn.Embedding(cfg.temporal_window, D)

        if cfg.use_regime_token:
            self.regime_proj = nn.Linear(cfg.signature_dim, D)
            self.regime_pe = nn.Parameter(torch.zeros(1, 1, D))
            nn.init.normal_(self.regime_pe, std=0.02)

        if cfg.num_memory_tokens > 0:
            # Learnable memory bank. First dim is num_prototypes if learnable
            # prototypes are used, otherwise num_clusters (k-means mode).
            bank_size = cfg.num_prototypes if cfg.num_prototypes > 0 else cfg.num_clusters
            self.memory_bank = nn.Parameter(
                torch.randn(bank_size, cfg.num_memory_tokens, D) * 0.02
            )
            self.memory_pe = nn.Parameter(torch.zeros(1, cfg.num_memory_tokens, D))
            nn.init.normal_(self.memory_pe, std=0.02)

        if cfg.num_prototypes > 0:
            # Learnable prototype signatures (clustered in signature space)
            self.proto_sig = nn.Parameter(
                torch.randn(cfg.num_prototypes, cfg.signature_dim) * 0.5
            )
            if cfg.num_memory_tokens == 0:
                # 3B mode: single learnable embedding per prototype for a
                # soft-weighted summary token.
                self.proto_summary = nn.Parameter(
                    torch.randn(cfg.num_prototypes, D) * 0.02
                )
                self.proto_summary_pe = nn.Parameter(torch.zeros(1, 1, D))
                nn.init.normal_(self.proto_summary_pe, std=0.02)

        if cfg.corr_bias_alpha > 0:
            layer = CorrBiasedEncoderLayer(
                d_model=D, nhead=cfg.num_heads,
                ff_dim=cfg.ff_dim, dropout=cfg.transformer_dropout,
            )
            self.transformer = CorrBiasedEncoder(layer, num_layers=cfg.num_layers)
            self._corr_biased = True
        else:
            layer = nn.TransformerEncoderLayer(
                d_model=D, nhead=cfg.num_heads,
                dim_feedforward=cfg.ff_dim, dropout=cfg.transformer_dropout,
                activation="gelu", batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(layer, num_layers=cfg.num_layers)
            self._corr_biased = False

        if cfg.corr_bias_alpha > 0 and cfg.corr_bias_gate_hidden > 0:
            self.alpha_gate = nn.Sequential(
                nn.Linear(cfg.signature_dim, cfg.corr_bias_gate_hidden),
                nn.GELU(),
                nn.Linear(cfg.corr_bias_gate_hidden, 1),
            )
        else:
            self.alpha_gate = None

        if cfg.dann_lambda_max > 0:
            self.regime_disc = RegimeDiscriminator(
                d_model=D, hidden=cfg.dann_hidden, num_classes=cfg.dann_classes,
            )
        else:
            self.regime_disc = None

        self.cs_ln = CrossSectionalLayerNorm(D)
        self.rank_head = nn.Sequential(
            nn.Linear(D, cfg.head_hidden), nn.GELU(),
            nn.Dropout(cfg.head_dropout),
            nn.Linear(cfg.head_hidden, 1),
        )
        if cfg.use_risk_head:
            self.risk_head = nn.Sequential(
                nn.Linear(D, cfg.head_hidden), nn.GELU(),
                nn.Dropout(cfg.head_dropout),
                nn.Linear(cfg.head_hidden, len(cfg.risk_quantiles)),
            )

    def forward_day(self, patches: Tensor, patch_mask: Tensor,
                    regime_sig: Tensor, active_mask: Tensor,
                    cluster_id: int = 0,
                    attn_bias_patch: Tensor | None = None,
                    dann_lambda: float = 0.0) -> dict[str, Tensor]:
        """
        patches:    [A, N+1, W, F]
        patch_mask: [A, N+1, W]  True = valid
        regime_sig: [signature_dim]  day-t regime signature (z-scored)
        active_mask: [num_nodes]
        cluster_id: index into memory_bank for iter 3A (hard retrieval)
        attn_bias_patch: optional [A, (N+1)*W, (N+1)*W] attention bias for
            the patch tokens only. Ignored unless cfg.corr_bias_alpha > 0.
            Prefix and suffix tokens are zero-padded internally.
        """
        cfg = self.cfg
        A, NP1, W, F = patches.shape
        D = cfg.hidden_dim
        num_nodes = active_mask.shape[0]

        x = self.input_proj(patches)                                   # [A, N+1, W, D]
        sp = self.spatial_pe(torch.arange(NP1, device=x.device))       # [N+1, D]
        tp = self.temporal_pe(torch.arange(W, device=x.device))        # [W, D]
        x = x + sp.view(1, NP1, 1, D) + tp.view(1, 1, W, D)

        x_flat = x.reshape(A, NP1 * W, D)
        mask_flat = patch_mask.reshape(A, NP1 * W)

        # Prototype assignment weights (iter 3B/C: soft; iter 9: RL hard sample)
        soft_weights = None
        log_prob = None
        sampled_idx = None
        if cfg.num_prototypes > 0:
            d2 = ((regime_sig.view(1, -1) - self.proto_sig) ** 2).sum(dim=1)  # [L]
            logits = -d2 / cfg.proto_temperature                               # [L]
            if cfg.use_rl_proto:
                if self.training:
                    dist = torch.distributions.Categorical(logits=logits)
                    sampled_idx = dist.sample()
                    log_prob = dist.log_prob(sampled_idx)
                else:
                    sampled_idx = logits.argmax()
                # Hard one-hot weights
                soft_weights = torch.zeros_like(logits)
                soft_weights[sampled_idx] = 1.0
            else:
                soft_weights = torch.softmax(logits, dim=0)

        # Prepend memory / summary tokens.
        memory_prefix_len = 0
        if cfg.num_memory_tokens > 0:
            if cfg.num_prototypes > 0:
                # 3C: soft-weighted memory bank across L prototypes.
                # memory_bank: [L, M, D]; soft_weights: [L]
                mem = (soft_weights.view(-1, 1, 1) * self.memory_bank).sum(dim=0)   # [M, D]
            else:
                # 3A: hard retrieval from k-means cluster.
                mem = self.memory_bank[cluster_id]                                  # [M, D]
            mem = mem + self.memory_pe.squeeze(0)
            mem = mem.unsqueeze(0).expand(A, cfg.num_memory_tokens, D)              # [A, M, D]
            x_flat = torch.cat([mem, x_flat], dim=1)
            mem_valid = torch.ones(A, cfg.num_memory_tokens,
                                   dtype=mask_flat.dtype, device=mask_flat.device)
            mask_flat = torch.cat([mem_valid, mask_flat], dim=1)
            memory_prefix_len = cfg.num_memory_tokens
        elif cfg.num_prototypes > 0:
            # 3B: single soft-weighted summary token derived from learnable prototypes.
            summary = (soft_weights.view(-1, 1) * self.proto_summary).sum(dim=0)    # [D]
            summary = summary.view(1, 1, D) + self.proto_summary_pe                 # [1, 1, D]
            summary = summary.expand(A, 1, D)                                        # [A, 1, D]
            x_flat = torch.cat([summary, x_flat], dim=1)
            sum_valid = torch.ones(A, 1, dtype=mask_flat.dtype, device=mask_flat.device)
            mask_flat = torch.cat([sum_valid, mask_flat], dim=1)
            memory_prefix_len = 1

        if cfg.use_regime_token:
            regime_emb = self.regime_proj(regime_sig).view(1, 1, D)     # [1, 1, D]
            regime_emb = regime_emb.expand(A, 1, D) + self.regime_pe    # [A, 1, D]
            x_flat = torch.cat([x_flat, regime_emb], dim=1)             # append at end
            regime_valid = torch.ones(A, 1, dtype=mask_flat.dtype, device=mask_flat.device)
            mask_flat = torch.cat([mask_flat, regime_valid], dim=1)

        key_pad = ~mask_flat
        gate_value: Tensor | None = None
        if self._corr_biased:
            S = x_flat.shape[1]
            if attn_bias_patch is not None:
                if self.alpha_gate is not None:
                    # Learnable gate over regime signature; scales alpha to [0, 1].
                    gate_value = torch.sigmoid(self.alpha_gate(regime_sig)).squeeze()
                    attn_bias_patch = attn_bias_patch * gate_value
                S_patch = NP1 * W
                bias_full = torch.zeros(A, S, S, device=x_flat.device, dtype=x_flat.dtype)
                # Patch positions sit at [memory_prefix_len : memory_prefix_len + NP1*W].
                s0 = memory_prefix_len
                s1 = s0 + S_patch
                bias_full[:, s0:s1, s0:s1] = attn_bias_patch.to(x_flat.dtype)
            else:
                bias_full = torch.zeros(A, S, S, device=x_flat.device, dtype=x_flat.dtype)
            attn_mask = expand_bias_for_heads(bias_full, cfg.num_heads)
            x_enc = self.transformer(x_flat, attn_bias=attn_mask, key_padding_mask=key_pad)
        else:
            x_enc = self.transformer(x_flat, src_key_padding_mask=key_pad)

        # Self-ticker today position shifts by memory_prefix_len
        self_today_idx = memory_prefix_len + (0 * W + (W - 1))
        z = x_enc[:, self_today_idx, :]                                # [A, D]

        # Scatter to [num_nodes, D]
        z_full = torch.zeros(num_nodes, D, device=z.device, dtype=z.dtype)
        z_full[active_mask] = z
        z_norm = self.cs_ln(z_full, active_mask)

        y_hat = self.rank_head(z_norm).squeeze(-1)
        q_hat = self.risk_head(z_norm) if cfg.use_risk_head else None
        regime_logits: Tensor | None = None
        if self.regime_disc is not None and dann_lambda > 0:
            # Pass only the active tickers' hidden states through the GRL +
            # discriminator. z_norm is [num_nodes, D]; active_mask selects the A rows.
            z_active = z_norm[active_mask]
            z_rev = grad_reverse(z_active, dann_lambda)
            regime_logits = self.regime_disc(z_rev)
        return {"y_hat": y_hat, "q_hat": q_hat, "z": z_norm,
                "proto_weights": soft_weights,
                "rl_log_prob": log_prob,
                "rl_action": sampled_idx,
                "corr_gate": gate_value,
                "regime_logits": regime_logits}


__all__ = ["REMStar", "REMConfig"]
