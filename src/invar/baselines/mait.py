"""MAiT: Macro-Adaptive iTransformer.

Dual-stream iTransformer extension with a SHARED encoder and a 5-input
regime gate. Stream A reads only the 24 panel variate tokens; Stream B
reads the 24 panel + 17 macro variate tokens through the same encoder
weights, then projects only the panel positions to a per-ticker score.
Predictions are blended by g(t) in [0, 1] gated by a small MLP on
five day-t macro scalars. Trained with composite IC loss + Stream A
standalone IC + L2 on g.

Design reference: docs/mait_design.md.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from src.invar.baselines.itransformer import (
    ITransformerConfig,
    _ITransformerBlock,
)


def ic_loss(y_pred: Tensor, y_true: Tensor) -> Tensor:
    """Cross-sectional 1 - Pearson correlation surrogate for rank IC.

    Args:
        y_pred: (N,) predicted scores.
        y_true: (N,) cross-sectional z-scored forward returns.

    Returns:
        Scalar in approximately [0, 2]. Minimised when corr = 1.
    """
    yp = y_pred - y_pred.mean()
    yt = y_true - y_true.mean()
    denom = yp.norm() * yt.norm() + 1.0e-8
    return 1.0 - (yp * yt).sum() / denom


def mait_loss(y_hat: Tensor, s_panel: Tensor, g: Tensor, y_true: Tensor,
              lambda_A: float = 0.5, lambda_gate: float = 0.05) -> Tensor:
    """Composite MAiT loss.

    L_total = L_primary + lambda_A * L_aux_A + lambda_gate * L_gate
      L_primary = ic_loss(y_hat, y_true)        blended prediction
      L_aux_A   = ic_loss(s_panel, y_true)      Stream A standalone, keeps
                                                Stream A competent so the
                                                gate-closed fallback preserves
                                                F3 performance.
      L_gate    = g ** 2                         encourage closed gate by
                                                default; let macro engage
                                                only when the IC term
                                                rewards it enough.

    Args:
        y_hat:    (N,) blended prediction.
        s_panel:  (N,) Stream A standalone prediction.
        g:        (), scalar gate value in [0, 1] for the day.
        y_true:   (N,) target.
        lambda_A: weight on Stream A standalone IC loss.
        lambda_gate: weight on g**2 penalty.

    Returns:
        Scalar loss.
    """
    return (
        ic_loss(y_hat, y_true)
        + lambda_A * ic_loss(s_panel, y_true)
        + lambda_gate * (g ** 2)
    )


def _make_block_config(d_model: int, n_heads: int, d_ff: int,
                       dropout: float) -> ITransformerConfig:
    """Build a minimal ITransformerConfig for `_ITransformerBlock`.

    The block reads only d_model, n_heads, ffn_hidden, dropout from cfg.
    The other config fields are required by the dataclass but unused.
    """
    return ITransformerConfig(
        n_features=1,
        lookback=60,
        d_model=d_model,
        n_heads=n_heads,
        ffn_hidden=d_ff,
        dropout=dropout,
    )


class MAiT(nn.Module):
    """Macro-Adaptive iTransformer.

    Shared encoder stack applied twice per forward pass (Stream A: panel
    tokens; Stream B: panel + macro tokens). Output is a per-ticker
    scalar score; cross-sectional ranking is computed downstream.
    """

    def __init__(
        self,
        n_panel: int = 24,
        n_macro: int = 17,
        L_lookback: int = 60,
        d_model: int = 128,
        n_heads: int = 4,
        d_ff: int = 256,
        n_layers: int = 3,
        dropout: float = 0.1,
        stream_dropout_p: float = 0.15,
        regime_dim: int = 5,
    ) -> None:
        super().__init__()
        self.n_panel = n_panel
        self.n_macro = n_macro
        self.L_lookback = L_lookback
        self.d_model = d_model
        self.stream_dropout_p = stream_dropout_p

        self.embed_panel = nn.Sequential(
            nn.Linear(L_lookback, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.embed_macro = nn.Sequential(
            nn.Linear(L_lookback, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        block_cfg = _make_block_config(d_model, n_heads, d_ff, dropout)
        self.blocks = nn.ModuleList([
            _ITransformerBlock(block_cfg) for _ in range(n_layers)
        ])

        self.proj = nn.Linear(d_model, 1)

        self.gate_mlp = nn.Sequential(
            nn.Linear(regime_dim, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def _encode_and_score(self, H: Tensor) -> Tensor:
        """Apply shared encoder and read scores from the first n_panel tokens.

        Args:
            H: (B, V, D) where V is the number of variate tokens to attend
               over. V is n_panel for Stream A and n_panel + n_macro for
               Stream B.

        Returns:
            (B,) score per ticker, mean-pooled over the panel positions.
        """
        for block in self.blocks:
            H = block(H, key_padding_mask=None)
        H_panel = H[:, : self.n_panel, :]
        return self.proj(H_panel).mean(dim=1).squeeze(-1)

    def forward(
        self,
        x_panel: Tensor,
        x_macro_lookback: Tensor,
        regime_input: Tensor,
        train_mode: bool = True,
        force_g: float | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Forward pass returning (y_hat, s_panel, s_macro, g).

        Args:
            x_panel:          (B, n_panel, L) panel features over lookback.
            x_macro_lookback: (n_macro, L) macro features over lookback.
            regime_input:     (regime_dim,) day-t macro scalars for the gate.
            train_mode: if True, apply stream-dropout (with probability
                stream_dropout_p, replace g with a Bernoulli(0.5) draw).
            force_g: if not None, override g with this scalar value (in
                [0, 1]). Used by the architectural-collapse unit test.

        Returns:
            y_hat:   (B,) blended prediction.
            s_panel: (B,) Stream A standalone prediction.
            s_macro: (B,) Stream B prediction (panel positions after the
                     macro-aware attention pass).
            g:       (,) scalar gate value (or per-step stream-dropout
                     draw at training time).
        """
        B = x_panel.size(0)

        H_panel = self.embed_panel(x_panel)
        H_macro = self.embed_macro(x_macro_lookback)
        H_macro = H_macro.unsqueeze(0).expand(B, -1, -1)

        # Stream A: encoder on panel tokens only.
        s_panel = self._encode_and_score(H_panel)

        # Stream B: encoder on (panel + macro) tokens, read panel positions.
        H_full = torch.cat([H_panel, H_macro], dim=1)
        s_macro = self._encode_and_score(H_full)

        # Gate.
        if force_g is not None:
            g = torch.tensor(float(force_g), device=x_panel.device,
                             dtype=x_panel.dtype)
        else:
            g = self.gate_mlp(regime_input).squeeze(-1)
            if (
                train_mode
                and self.stream_dropout_p > 0.0
                and torch.rand((), device=g.device).item() < self.stream_dropout_p
            ):
                g = torch.bernoulli(
                    torch.tensor(0.5, device=g.device, dtype=g.dtype),
                )

        y_hat = (1.0 - g) * s_panel + g * s_macro
        return y_hat, s_panel, s_macro, g


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
