"""Vendored MERA architecture (WWW 2025 Companion).

Source: https://github.com/chenchen1104/MERA (commit on `main` as of
2026-05-04). The upstream repo is built on top of Microsoft's TRA code
(Lin, Wang, Liu, Zhang, Jiang, Yang, Bian, KDD 2021), adds the MERA
modules in ``MERA/src/model_moe_attn.py``, and depends on the heavy
``fmoe`` (FastMoE) distributed library plus Qlib for data loading.

We strip:

  - the Qlib + h5py data pipeline (``main_500.py``, ``dataset.py``);
  - the full TRA training loop (``TRAModel.fit/test_epoch``);
  - the FastMoE distributed MoE machinery (``custom_moe_layer.py``,
    ``noisy_gate.py``, ``noisy_gate_vmoe.py``); FastMoE is not safely
    installable in the v2 baseline conda env and forces a CUDA-only
    build, while we want CPU-portable code for retrieval pool ops;
  - the upstream ``Transformer.forward`` plumbing for the
    pre-aggregated ``similars`` h5 tensor.

We keep, faithful to the paper text the user shared:

  - 2-layer Transformer encoder, ``d_model=128``, 4 heads, with input
    BatchNorm, learned linear projection, sinusoidal positional
    encoding (matches the upstream ``Transformer`` class);
  - a target-aware attention aggregator over retrieved neighbours
    (matches the upstream
    ``F.softmax(torch.bmm(query, similar_features.transpose(1,2))) @
    similar_label_embedding`` block);
  - a Sparse MoE block reimplemented in plain PyTorch with M=4 experts
    and Top-K=1 activation. The paper text specifies a small GRU per
    expert; we follow that spec rather than the upstream FMoE-MLP
    expert (the FMoE expert is an MLP, but the published WWW paper
    description says "small GRU per expert", so we go with what the
    paper says, consistent with the user's task brief);
  - a masked-autoencoder pre-training head: random-mask a fraction of
    the (T, F) input tokens, reconstruct via a small linear decoder,
    MSE loss. The upstream code only loads ``model_init_state`` from
    disk and does not contain the masked-AE pre-training step itself,
    so we implement it cleanly per the paper's description.

Reference: Liu, Y., Song, C.-H., Liu, P., Li, N., Dai, T., Bao, J.,
Jiang, Y., Xia, S.-T. (2025). "MERA: Mixture of Experts with
Retrieval-Augmented Representation for Modeling Diversified Stock
Patterns." Companion Proceedings of the ACM Web Conference 2025.
DOI: 10.1145/3701716.3715513.
"""
from .module import (
    MERABackbone,
    MERAMaskedAEHead,
    PositionalEncoding,
    RetrievalAttentionAggregator,
    SparseGRUMoE,
)

__all__ = [
    "MERABackbone",
    "MERAMaskedAEHead",
    "PositionalEncoding",
    "RetrievalAttentionAggregator",
    "SparseGRUMoE",
]
