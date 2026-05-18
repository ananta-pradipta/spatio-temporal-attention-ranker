"""InVAR-STAR: Inverted-Variate Attention with Regime-aware Self-Throttling.

A non-surgical extension of iTransformer (Liu et al., ICLR 2024) that promotes
each of the 24 macro indicators to a first-class variate token coexisting with
the 26 stock-feature tokens, gates macro-to-stock attention with a stochastic
Concrete-relaxed self-throttling scalar beta_t, and decodes through a
mixture-of-experts head with noisy top-k routing. Trained with SWA over the
final third of training.

Design document: see `docs/invar_star_design.md` (loaded 2026-05-10).
"""

from src.invar_star.model import (
    set_global_seed,
    MacroVariateBank,
    SelfThrottlingGate,
    ThrottledVariateAttention,
    InVARSTARBlock,
    MoERankingHead,
    InVARSTAR,
)
from src.invar_star.losses import (
    throttle_kl_prior,
    load_balance_loss,
    weighted_pearson_ic_loss,
)

__all__ = [
    "set_global_seed",
    "MacroVariateBank",
    "SelfThrottlingGate",
    "ThrottledVariateAttention",
    "InVARSTARBlock",
    "MoERankingHead",
    "InVARSTAR",
    "throttle_kl_prior",
    "load_balance_loss",
    "weighted_pearson_ic_loss",
]
