"""RegimeXer-iT: regime-aware iTransformer extension.

A new variate-as-token model that:
  1. Treats each of F = 26 panel features as one variate token AND each of
     F_m = 24 macro features as one variate token; joint encoder attends
     across the 50 tokens.
  2. Pools macro tokens after the first joint encoder block to a per-stock
     context c, used by both FiLM modulation (on stock tokens) and the
     invariance gate.
  3. Runs a thin "invariant twin" pathway over stock-only tokens with no
     macro and no FiLM, weight-shared embedding only.
  4. Mixes the macro-conditioned output and the invariant output by a
     learned per-stock gate alpha in [0, 1], where alpha is regularized
     toward zero so the model only opens the gate where macro helps.
  5. Adds an auxiliary vol head supervised by `fwd_vol_20d`.

Spec: 2026-05-11 user-supplied design (Discord attachment). The CCC
loss component (C6) is dropped per the 2026-05-11 update; the existing
`src.invar.training.loss.hybrid_loss` (Huber + listwise IC + pairwise
margin) is used unchanged for all RegimeXer arms.

The architectural collapse property: when alpha is forced to 0 at every
position, the model output equals the invariant-pathway computation on
the same input (panel-only, single iT block, mean over F, linear to
scalar). This is the key F3 protection guarantee.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

from src.invar.baselines.itransformer import (
    ITransformerConfig,
    _ITransformerBlock,
)
from src.invar.baselines.regimexer_blocks import FiLMBlock, InvarianceGate


RegimeXerMode = Literal[
    "macro_tokens_only",   # A1: no FiLM, no gate, no vol head
    "film",                # A2: + FiLM (still no gate, no vol head)
    "no_gate",             # A3: A4 with alpha hard-forced to 1
    "full",                # A4 (recommended)
    "moe_k8",              # A5: A4 + MoE FFN on block 3
]


def _resolve_mode_flags(mode: RegimeXerMode) -> dict:
    """Mode-to-flags mapping. See design spec Section "CLI surface"."""
    table = {
        "macro_tokens_only": dict(
            use_film=False, use_gate=False, use_vol_head=False, use_moe=False,
        ),
        "film": dict(
            use_film=True, use_gate=False, use_vol_head=False, use_moe=False,
        ),
        "no_gate": dict(
            use_film=True, use_gate=False, use_vol_head=True, use_moe=False,
        ),
        "full": dict(
            use_film=True, use_gate=True, use_vol_head=True, use_moe=False,
        ),
        "moe_k8": dict(
            use_film=True, use_gate=True, use_vol_head=True, use_moe=True,
        ),
    }
    if mode not in table:
        raise ValueError(f"unknown regimexer mode: {mode}")
    return table[mode]


@dataclass
class RegimeXerITConfig:
    n_panel: int = 26
    n_macro: int = 24
    lookback: int = 60
    d_model: int = 128
    n_heads: int = 8
    n_layers: int = 3
    ffn_hidden: int = 512
    dropout: float = 0.1
    lambda_vol: float = 0.1
    lambda_alpha: float = 1.0e-3
    ema_decay: float = 0.99
    mode: RegimeXerMode = "full"


def _make_block(cfg: RegimeXerITConfig) -> _ITransformerBlock:
    """Build a single iTransformer-style block at RegimeXer hyperparameters."""
    block_cfg = ITransformerConfig(
        n_features=cfg.n_panel,
        lookback=cfg.lookback,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        ffn_hidden=cfg.ffn_hidden,
        dropout=cfg.dropout,
    )
    return _ITransformerBlock(block_cfg)


class _VariateEmbed(nn.Module):
    """LayerNorm(Linear(L -> d_model)) per variate token.

    Maps (B, V, L) -> (B, V, d_model). The Linear is applied to the last
    dim (L) and is shared across all V variate tokens.
    """

    def __init__(self, lookback: int, d_model: int) -> None:
        super().__init__()
        self.proj = nn.Linear(lookback, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(self.proj(x))


class RegimeXerIT(nn.Module):
    """RegimeXer-iT full forward (mode="full" / A4).

    The other modes (macro_tokens_only / film / no_gate / moe_k8) wire
    behavioural flags via the `mode` constructor argument and the
    `force_alpha` forward argument. Phase 0 only verifies the full
    forward; Phase 1 onward adds the trainer integration.

    Inputs:
        features: (N, L, F=26) panel features.
        macro:    (L, F_m=24)  shared macro lookback.
        mask:     (N,) bool.

    Outputs (dict):
        y_hat: (N,) ranking score, masked tickers set to 0.
        v_hat: (N,) vol head, masked tickers set to 0.
        alpha: (N,) gate value per stock, for logging.
        regime_logits: None, present for hybrid_loss interface compat.
        c:     (N, d) macro context per stock (returned for diagnostics).
    """

    def __init__(self, cfg: RegimeXerITConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or RegimeXerITConfig()
        c = self.cfg
        flags = _resolve_mode_flags(c.mode)
        self.use_film = flags["use_film"]
        self.use_gate = flags["use_gate"]
        self.use_vol_head = flags["use_vol_head"]
        self.use_moe = flags["use_moe"]
        if self.use_moe:
            raise NotImplementedError(
                "moe_k8 is Phase 3 scope; not built in Phase 0/1.",
            )

        # Variate embeddings: per-feature MLP on the L-length lookback.
        self.embed_stock = _VariateEmbed(c.lookback, c.d_model)
        self.embed_macro = _VariateEmbed(c.lookback, c.d_model)

        # Three joint encoder blocks over (F + F_m) variate tokens.
        self.block1 = _make_block(c)
        self.block2 = _make_block(c)
        self.block3 = _make_block(c)
        # Thin invariant twin: only needed if the gate is learnable. For
        # alpha-forced modes (A1, A2, A3) the thin twin output is multiplied
        # by zero and contributes no signal, so we skip construction to keep
        # the parameter count tight.
        if self.use_gate:
            self.block_thin = _make_block(c)
        else:
            self.block_thin = None

        # FiLM modulation on stock tokens after block 2, conditioned on c.
        if self.use_film:
            self.film = FiLMBlock(c.d_model, c.n_panel)
        else:
            self.film = None

        # Invariance gate (only built when use_gate, to match parameter count).
        if self.use_gate:
            self.gate = InvarianceGate(c.d_model, ema_decay=c.ema_decay)
        else:
            self.gate = None

        # Output heads.
        self.y_head = nn.Linear(c.d_model, 1)
        if self.use_vol_head:
            self.v_head = nn.Linear(c.d_model, 1)
        else:
            self.v_head = None

        # Log parameter count at construction per spec style requirement.
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[regimexer] params={n_params:,}  mode={c.mode}", flush=True)

    def _attend(self, block: _ITransformerBlock, H: Tensor) -> Tensor:
        """Run one iT block with no key padding mask, batch_first layout."""
        return block(H, key_padding_mask=None)

    def _invariant_pathway(self, H0_stock: Tensor) -> Tensor:
        """Compute the thin-twin invariant pathway: (N, F, d)."""
        return self._attend(self.block_thin, H0_stock)

    def forward(
        self,
        features: Tensor,
        macro: Tensor,
        mask: Tensor,
        return_attn: bool = False,
        force_alpha: float | None = None,
    ) -> dict:
        # features: (N, L, F), macro: (L, F_m), mask: (N,) bool.
        N = features.size(0)
        c = self.cfg

        # Embed each variate as one token.
        H0_stock = self.embed_stock(features.transpose(-1, -2))
        H0_macro = self.embed_macro(macro.transpose(-1, -2))
        H0_macro = H0_macro.unsqueeze(0).expand(N, -1, -1)

        # Block 1: joint encoder over (F + F_m) tokens.
        H_joint1 = self._attend(self.block1, torch.cat([H0_stock, H0_macro], dim=1))
        H1_stock = H_joint1[:, : c.n_panel, :]
        H1_macro = H_joint1[:, c.n_panel:, :]

        # Macro context: mean over the F_m macro tokens per stock.
        c_ctx = H1_macro.mean(dim=1)
        # Update running mean of c (training time only; only when gate exists).
        if self.training and self.gate is not None:
            self.gate.update_running_mean(c_ctx)

        # Block 2: joint encoder again (no FiLM yet).
        H_joint2 = self._attend(
            self.block2, torch.cat([H1_stock, H1_macro], dim=1),
        )
        H2_stock = H_joint2[:, : c.n_panel, :]
        H2_macro = H_joint2[:, c.n_panel:, :]

        # FiLM modulation on stock tokens, conditioned on c (skipped in A1).
        if self.use_film:
            H2_film = self.film(H2_stock, c_ctx)
        else:
            H2_film = H2_stock

        # Block 3: joint encoder on FiLM-modulated stock tokens + macro.
        H_joint3 = self._attend(
            self.block3, torch.cat([H2_film, H2_macro], dim=1),
        )
        H3_cond = H_joint3[:, : c.n_panel, :]

        # Invariance gate alpha.
        if force_alpha is not None:
            alpha = torch.full((N, 1), float(force_alpha),
                                device=features.device, dtype=features.dtype)
        elif self.use_gate:
            alpha = self.gate(c_ctx)
        else:
            # A1, A2, A3 force alpha to 1.0 structurally (no gate).
            alpha = torch.ones(N, 1, device=features.device, dtype=features.dtype)

        # Mixed output. When use_gate is False and alpha is forced to 1, we
        # skip the thin twin altogether (its output would be multiplied by
        # zero). When use_gate is True (or force_alpha is set), we compute
        # both pathways and blend.
        if not self.use_gate and force_alpha is None:
            H_mixed = H3_cond
        else:
            if self.block_thin is None:
                # use_gate=False but force_alpha was set; build twin output
                # on the fly using the available block stack. Should not
                # happen in normal training paths; raise for clarity.
                raise RuntimeError(
                    "force_alpha was set on a mode without block_thin; "
                    "the invariant pathway is not available.",
                )
            H3_base = self._invariant_pathway(H0_stock)
            H_mixed = (
                alpha.unsqueeze(-1) * H3_cond
                + (1.0 - alpha.unsqueeze(-1)) * H3_base
            )

        # Heads.
        z = H_mixed.mean(dim=1)
        y_hat = self.y_head(z).squeeze(-1)
        if self.use_vol_head:
            v_hat = self.v_head(z).squeeze(-1)
        else:
            v_hat = torch.zeros(N, device=features.device, dtype=features.dtype)

        # Mask out inactive tickers.
        m_f = mask.float()
        y_hat = y_hat * m_f
        v_hat = v_hat * m_f

        # regime_logits is a dummy zero tensor (8-vector) so hybrid_loss's
        # regime_ce_loss call does not crash on None.attr access. The
        # LossWeights setting regime_ce=0.0 zeros out its contribution to
        # the total loss anyway.
        return {
            "y_hat": y_hat,
            "regime_logits": torch.zeros(
                8, device=features.device, dtype=features.dtype,
            ),
            "vol_hat": v_hat,
            "alpha": alpha.squeeze(-1),
            "c": c_ctx,
        }


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
