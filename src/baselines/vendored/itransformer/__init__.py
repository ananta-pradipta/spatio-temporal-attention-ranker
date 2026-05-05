"""Vendored iTransformer architecture (ICLR 2024).

Source: https://github.com/thuml/iTransformer (MIT). Adapted from
``model/iTransformer.py`` and the supporting layers in ``layers/``
(``Embed.py``, ``SelfAttention_Family.py``, ``Transformer_EncDec.py``).

We strip the upstream training loop, dataloaders, the decoder path, the
non-iTransformer model variants, the optional output-attention plumbing,
and the unused causal/Prob/Flow attentions. We keep only the
inverted-transformer encoder used in the paper's headline experiments.

Reference:
    Liu, Y., Hu, T., Liu, H., Zhou, J., Li, S., Long, M. (2024).
    "iTransformer: Inverted Transformers Are Effective for Time Series
    Forecasting." ICLR.
"""
from .module import (
    AttentionLayer,
    DataEmbeddingInverted,
    Encoder,
    EncoderLayer,
    FullAttention,
    ITransformerModel,
)

__all__ = [
    "ITransformerModel",
    "DataEmbeddingInverted",
    "FullAttention",
    "AttentionLayer",
    "EncoderLayer",
    "Encoder",
]
