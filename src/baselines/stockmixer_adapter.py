"""Adapter around SJTU-DMTai/StockMixer for our biotech panel.

Keeps the published model architecture but aligns the training loop
and evaluation with our conventions:
  - input shape [N_active, W, F] from our enriched panel tensors
  - target: 5d forward log return, cross-sectionally z-scored per day
  - loss: cs_mse_loss (unchanged from our convention) + light ranking
    (StockMixer's original uses L_reg + alpha * L_rank; we use ours)
  - metrics: daily IC and rank-IC, same as STAR/baselines

The StockMixer model file is vendored from ~/baselines/StockMixer/src/model.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

STOCKMIXER_SRC = Path.home() / "baselines" / "StockMixer" / "src"
if str(STOCKMIXER_SRC) not in sys.path:
    sys.path.insert(0, str(STOCKMIXER_SRC))

# Import vendored class. Note: their file is `model.py`.
from model import StockMixer as _StockMixer  # type: ignore


class StockMixerAdapter(nn.Module):
    """Wraps StockMixer. Input: [A, W, F]. Output: [A] rank scores.

    A = number of active tickers on the day, W = temporal window,
    F = feature channels. For StockMixer's internal conv to work,
    W must be even (stride-2 conv halves the time axis).

    `stocks_pad` is the static stock count the vendored `NoGraphMixer`
    expects. We pad/truncate active tickers to `stocks_pad` using a
    mask-aware zero fill and recover the active-ticker outputs after
    the forward pass.
    """

    def __init__(self, stocks_pad: int, time_steps: int, channels: int,
                 market: int = 20):
        super().__init__()
        self.stocks_pad = stocks_pad
        self.time_steps = time_steps
        self.channels = channels
        self.inner = _StockMixer(
            stocks=stocks_pad,
            time_steps=time_steps,
            channels=channels,
            market=market,
            scale=1,
        )

    def forward(self, x: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
        """x: [N_pad, W, F] pre-aligned. active_mask: [N_pad] bool.

        Returns: [N_pad] raw scores. Caller should apply active_mask
        when computing loss / metrics.
        """
        # StockMixer expects input shape [stocks, time_steps, channels]
        y = self.inner(x)          # [N_pad, 1]
        return y.squeeze(-1)       # [N_pad]
