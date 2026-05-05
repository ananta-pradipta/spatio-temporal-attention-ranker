"""Vendored FactorVAE architecture (AAAI 2022).

Source: https://github.com/x7jeon8gi/FactorVAE (unofficial PyTorch
implementation; the original authors did not publicly release code).

We strip the upstream training loop / dataloader / Qlib dependency and
keep only the core nn.Modules required for cross-sectional return
prediction:

    FeatureExtractor   GRU over a per-ticker (T, F) window -> stock latent
    FactorEncoder      Posterior over factor mu, sigma given returns
    AlphaLayer         Per-stock idiosyncratic alpha (mu, sigma)
    BetaLayer          Per-stock factor exposures
    FactorDecoder      Reconstruct returns from factors + (alpha, beta)
    AttentionLayer     One-head attention over stocks (factor predictor)
    FactorPredictor    Prior over factors used at inference time
    FactorVAE          End-to-end module bundling the four sub-modules

Reference: Duan, Y., Wang, L., Zhang, Q., Li, J. (2022). "FactorVAE: A
Probabilistic Dynamic Factor Model Based on Variational Autoencoder for
Predicting Cross-Sectional Stock Returns." AAAI.
"""
from .module import (
    FeatureExtractor,
    FactorEncoder,
    AlphaLayer,
    BetaLayer,
    FactorDecoder,
    AttentionLayer,
    FactorPredictor,
    FactorVAE,
)

__all__ = [
    "FeatureExtractor",
    "FactorEncoder",
    "AlphaLayer",
    "BetaLayer",
    "FactorDecoder",
    "AttentionLayer",
    "FactorPredictor",
    "FactorVAE",
]
