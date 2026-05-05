"""Vendored PatchTST architecture (ICLR 2023).

Source: https://github.com/yuqinie98/PatchTST (Apache-2.0). Specifically
adapted from
``PatchTST_supervised/layers/PatchTST_backbone.py`` and
``PatchTST_supervised/layers/PatchTST_layers.py`` and
``PatchTST_supervised/layers/RevIN.py``.

We strip the upstream training loop, dataloaders, the optional series
decomposition (moving-average + residual head), and the multi-horizon
forecasting head ``Flatten_Head`` which projects to ``target_window``.
We keep:

    Transpose                  utility
    get_activation_fn          GELU/ReLU helper
    positional_encoding        learned/sinusoidal pe registrations
    RevIN                      reversible instance normalisation
    _ScaledDotProductAttention scaled dot-product with optional residual
    _MultiheadAttention        wrapper around the SDP attention
    TSTEncoderLayer            single transformer encoder block
    TSTEncoder                 stack of N encoder blocks
    TSTiEncoder                channel-independent input encoder
    PatchTST_backbone          patch + transformer + (no head)

Reference: Nie, Y., Nguyen, N. H., Sinthong, P., Kalagnanam, J. (2023).
"A Time Series is Worth 64 Words: Long-term Forecasting with
Transformers." ICLR.
"""
from .module import (
    PatchTST_backbone,
    RevIN,
    TSTEncoder,
    TSTEncoderLayer,
    TSTiEncoder,
)

__all__ = [
    "PatchTST_backbone",
    "RevIN",
    "TSTEncoder",
    "TSTEncoderLayer",
    "TSTiEncoder",
]
