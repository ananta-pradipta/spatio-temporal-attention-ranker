"""Adapter around iTransformer (Liu et al., ICLR 2024) for our biotech panel.

iTransformer = inverted transformer for time-series forecasting.
The paper's central idea is to treat each *variate* as one token and
let self-attention run *across variates*. For univariate-style time
series this is equivalent to swapping the spatial/temporal axes of the
tokeniser; for cross-sectional ranking on a multi-asset panel it lines
up exactly with what we want, because every ticker can attend to every
other ticker in the active panel for that day.

Adaptation for our (N_active, T=20, F=22) per-day cross-sectional
ranking task:

  - One token per ticker (= "variate" in the paper).
  - Each token's input vector is the flattened lookback feature window
    of length ``T * F = 440``. The paper embeds each variate's lookback
    via a single ``Linear(seq_len, d_model)`` so we can plug ``T * F``
    in directly without changing the architecture.
  - Encoder: ``e_layers`` layers of self-attention across the
    ``N_active`` ticker tokens. With our 244-ticker biotech universe
    and ~150-200 active tickers per day, this is small enough that we
    use full quadratic attention (the paper's default).
  - Per-variate output head ``Linear(d_model, 1)`` produces one scalar
    score per ticker. No cross-channel mixer is needed because the
    feature dimension was already absorbed into the token embedding.

The adapter mirrors ``patchtst_adapter.PatchTSTAdapter`` in shape so
the trainer is a near-clone of ``train_patchtst_v2.py``.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.baselines.vendored.itransformer import ITransformerModel


@dataclass
class ITransformerHyperparams:
    """iTransformer architecture knobs.

    Defaults correspond to the project-side adaptation laid out in the
    task brief (d_model=128, n_heads=4, d_ff=256, e_layers=2). These
    are smaller than the paper's headline configuration because our
    variate dimension is the active ticker count (~150-200) rather
    than the long-horizon forecasting datasets in the paper.

    Args:
        d_feat: number of panel features F (kept for symmetry with the
            other v2 adapters; only used for the embedding-input
            calculation T * F).
        context_window: lookback length T (default 20, v2 protocol).
        d_model: transformer hidden width.
        n_heads: number of attention heads (must divide d_model).
        d_ff: feed-forward width inside each encoder block.
        e_layers: number of encoder layers stacked across variates.
        dropout: shared dropout for embedding + attention + FFN.
        activation: ``'gelu'`` (default) or ``'relu'``.
        use_norm: paper's non-stationary lookback normalisation. Off
            by default because the v2 protocol already standardises
            features per training fold.
        pred_len: per-variate output dimension. Fixed at 1 for
            cross-sectional ranking.
    """

    d_feat: int = 22
    context_window: int = 20
    d_model: int = 128
    n_heads: int = 4
    d_ff: int = 256
    e_layers: int = 2
    dropout: float = 0.1
    activation: str = "gelu"
    use_norm: bool = False
    pred_len: int = 1


class ITransformerAdapter(nn.Module):
    """iTransformer wrapped for our (N_active, T, F) panel format.

    Public surface used by ``train_itransformer_v2.py``:
        forward(x_window) -> (N_active,) score per active ticker.

    There is no auxiliary loss: iTransformer is a pure regression
    model, and the trainer applies ``cs_mse_loss`` on the scalar
    score, mirroring MASTER, StockMixer, DySTAGE, and PatchTST.

    Internals (per day, single forward pass over the entire active
    panel so cross-ticker attention can fire):
        x_window: (N_active, T, F)
          flatten lookback features  -> (N_active, T*F)
          add batch dim, swap axes   -> (1, T*F, N_active)
              ^ shape iTransformer expects: (B, L, N)
          ITransformerModel          -> (1, pred_len=1, N_active)
          squeeze                    -> (N_active,)
    """

    def __init__(self, hp: ITransformerHyperparams):
        super().__init__()
        self.hp = hp
        # The flattened lookback per ticker is the variate's "time" axis
        # for the inverted transformer. Length = T * F = 20 * 22 = 440.
        seq_len = hp.context_window * hp.d_feat
        self.seq_len = seq_len
        self.model = ITransformerModel(
            seq_len=seq_len,
            pred_len=hp.pred_len,
            d_model=hp.d_model,
            n_heads=hp.n_heads,
            e_layers=hp.e_layers,
            d_ff=hp.d_ff,
            dropout=hp.dropout,
            activation=hp.activation,
            use_norm=hp.use_norm,
        )

    def forward(self, x_window: torch.Tensor) -> torch.Tensor:
        """Score per active ticker.

        Args:
            x_window: (N_active, T, F).

        Returns:
            y_hat: (N_active,) raw scalar scores.
        """
        n_active, _, _ = x_window.shape
        # Flatten the (T, F) lookback per ticker into a length-(T*F) vector.
        # iTransformer expects (B, L, N) where N is the number of
        # variates (= tickers) and L is the per-variate sequence length.
        x_flat = x_window.reshape(n_active, -1)              # (N, T*F)
        x_in = x_flat.transpose(0, 1).unsqueeze(0)           # (1, T*F, N)
        y = self.model(x_in)                                 # (1, pred_len, N)
        # pred_len=1 by construction; squeeze out batch + horizon axes.
        y_hat = y.squeeze(0).squeeze(0)                      # (N,)
        return y_hat


__all__ = ["ITransformerAdapter", "ITransformerHyperparams"]
