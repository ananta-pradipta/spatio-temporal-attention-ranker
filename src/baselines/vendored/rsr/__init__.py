"""Vendored RSR architecture (Feng et al., TOIS 2019).

Source: https://github.com/fulifeng/Temporal_Relational_Stock_Ranking

The upstream code is TensorFlow 1.x (``rank_lstm.py`` / ``relation_rank_lstm.py``)
with a custom data loader, training loop, and ranking-loss formulation
tightly coupled to the NASDAQ/NYSE Wikidata graphs they shipped. We
strip the data-loading and training-loop code and port only the two
core nn.Modules to PyTorch:

    StockLSTM            Per-ticker LSTM over the (T, F) lookback,
                         returning the final-step hidden state e_i.
    TemporalGraphAttention
                         The "Explicit relation rank" relation-aware
                         aggregation: for each (day, ticker i) compute
                         g_i = sum_{j: A[i,j]=1} alpha_ij * e_j with
                         attention weights alpha_ij produced by the
                         relation-aware leaky-ReLU + softmax operator.

Reference: Feng, F., He, X., Wang, X., Luo, C., Liu, Y., Chua, T.-S.
(2019). "Temporal Relational Ranking for Stock Prediction." ACM TOIS.
https://doi.org/10.1145/3309547
"""
from .module import StockLSTM, TemporalGraphAttention

__all__ = ["StockLSTM", "TemporalGraphAttention"]
