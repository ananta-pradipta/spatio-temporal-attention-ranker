"""External-baseline adapters for InVAR Phase 3 comparison.

All baselines reuse ``src.invar.data.dataset.InvarDataset`` so the
training data, fold splits, FoldScaler, and active mask are identical
to the InVAR-C headline. Each adapter provides a ``ModelClass`` and
``ModelConfig`` compatible with ``src.invar.training.train.train_one``
via a thin shim that swaps the backbone.
"""
from src.invar.baselines.regimexer import RegimeXerIT, RegimeXerITConfig
from src.invar.baselines.regimexer_blocks import FiLMBlock, InvarianceGate

__all__ = [
    "RegimeXerIT",
    "RegimeXerITConfig",
    "FiLMBlock",
    "InvarianceGate",
]
