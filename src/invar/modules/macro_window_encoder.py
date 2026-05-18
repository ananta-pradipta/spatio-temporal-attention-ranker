"""MacroWindowEncoder for InVAR-v6.

Replaces the v4 single-day macro projection with a richer temporal encoder
that summarises the full L-step macro lookback into a fixed
``out_dim`` state vector. Four modes are supported (Section 3 of the
v6 spec):

  - ``last``         : project ``macro[:, -1, :]`` only. Reproduces v4
                       baseline behavior.
  - ``mlp_flat``     : flatten the (L, F_macro) tensor and pass through
                       a 2-layer MLP. Default for v6.
  - ``temporal_attn``: project to ``hidden_dim``, one
                       ``TransformerEncoderLayer`` over time, last-step
                       pooling, project to ``out_dim``.
  - ``gru``          : GRU over time, take the final hidden, project to
                       ``out_dim``.

The encoder operates on batched (B, L, F_macro) tensors and returns
(B, out_dim). For unbatched callers (single-day pipeline), wrap the
input via ``unsqueeze(0)`` and ``squeeze(0)`` at the call site.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MacroWindowEncoder(nn.Module):
    def __init__(
        self,
        macro_dim: int = 24,
        lookback: int = 60,
        out_dim: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        mode: str = "mlp_flat",
    ) -> None:
        super().__init__()
        if mode not in ("last", "mlp_flat", "temporal_attn", "gru"):
            raise ValueError(f"unknown mode: {mode!r}")
        self.macro_dim = macro_dim
        self.lookback = lookback
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.mode = mode

        if mode == "last":
            self.proj = nn.Linear(macro_dim, out_dim)
        elif mode == "mlp_flat":
            self.proj = nn.Sequential(
                nn.Linear(lookback * macro_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, out_dim),
            )
        elif mode == "temporal_attn":
            self.in_proj = nn.Linear(macro_dim, hidden_dim)
            self.encoder = nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim * 2,
                dropout=dropout, batch_first=True,
            )
            self.out_proj = nn.Linear(hidden_dim, out_dim)
        elif mode == "gru":
            self.gru = nn.GRU(
                input_size=macro_dim, hidden_size=hidden_dim,
                num_layers=1, batch_first=True,
            )
            self.out_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, macro: torch.Tensor) -> torch.Tensor:
        """Args:
            macro : ``(B, L, F_macro)``.

        Returns:
            ``(B, out_dim)``.
        """
        if macro.dim() == 2:
            macro = macro.unsqueeze(0)
        B = macro.shape[0]
        if self.mode == "last":
            return self.proj(macro[:, -1, :])
        if self.mode == "mlp_flat":
            return self.proj(macro.reshape(B, -1))
        if self.mode == "temporal_attn":
            h = self.in_proj(macro)
            h = self.encoder(h)
            return self.out_proj(h[:, -1, :])
        # gru
        _, h_final = self.gru(macro)
        return self.out_proj(h_final.squeeze(0))


__all__ = ["MacroWindowEncoder"]
