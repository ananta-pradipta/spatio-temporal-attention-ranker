"""Adapter around SJTU-DMTai/MASTER (AAAI 2024) for our biotech panel.

MASTER = Market-guided Stock Transformer. Architecture:
  (a) Feature gate: last-timestep "market features" softmax-scale each
      stock's features.
  (b) Intra-stock temporal attention (T-Attention).
  (c) Inter-stock spatial attention (S-Attention).
  (d) Temporal attention pooling.
  (e) Linear head → scalar per stock.

Our adaptation:
  - d_feat = 22 (the enriched panel features).
  - d_gate_input = 7 (our causal regime signature: XBI vol, dispersion,
    correlation, VIX slope, PC1 share, skewness, kurtosis).
  - Input per day: concat [stock_features (22), market_sig (7)] per time
    step = 29 dims. T = 20.
  - d_model = 128.
  - Training: cs_mse_loss on z-scored 5d forward returns, same walk-forward
    folds, same 5 seeds as all our baselines.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

MASTER_SRC = Path.home() / "baselines" / "MASTER"
if str(MASTER_SRC) not in sys.path:
    sys.path.insert(0, str(MASTER_SRC))

# Import the vendored MASTER nn.Module. We do NOT import MASTERModel (which
# wraps MASTER in a Qlib-tied training loop); we keep only the architecture.
from master import MASTER as _MASTER  # type: ignore


class MASTERAdapter(nn.Module):
    """MASTER nn.Module wrapped for our panel format.

    Input at day t: x shape [N_active, W, d_feat + d_gate]; the last d_gate
    columns are the day-t market regime signature (broadcast across W).

    Output: [N_active] raw scores per ticker.
    """
    def __init__(self, d_feat: int, d_model: int, t_nhead: int, s_nhead: int,
                 gate_input_start_index: int, gate_input_end_index: int,
                 T_dropout_rate: float = 0.1, S_dropout_rate: float = 0.1,
                 beta: float = 5.0):
        super().__init__()
        self.inner = _MASTER(
            d_feat=d_feat, d_model=d_model,
            t_nhead=t_nhead, s_nhead=s_nhead,
            T_dropout_rate=T_dropout_rate, S_dropout_rate=S_dropout_rate,
            gate_input_start_index=gate_input_start_index,
            gate_input_end_index=gate_input_end_index,
            beta=beta,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, W, d_feat + d_gate] where the market-gate slice is the last
        # (gate_input_end_index - gate_input_start_index) columns.
        return self.inner(x)  # [N]
