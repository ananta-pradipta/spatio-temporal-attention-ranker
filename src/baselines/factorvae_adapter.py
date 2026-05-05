"""Adapter around FactorVAE (Duan et al., AAAI 2022) for our biotech panel.

FactorVAE = probabilistic dynamic factor model: a GRU feature extractor
emits a per-stock latent, a posterior factor encoder maps that latent
plus realised returns to (mu_post, sigma_post) over K latent factors, an
attention-based factor predictor emits the prior (mu_prior, sigma_prior)
used at inference, and an alpha+beta decoder reconstructs y_hat.

Our adaptation:
  - Input per active day: ``x_window`` of shape (N_active, T, F) where
    F = 22 (the enriched panel features) and T = 20.
  - Score per ticker: deterministic posterior-mean reconstruction
    ``alpha_mu + beta @ factor_mu``. We expose this both via
    ``forward(x)`` (uses the prior factor predictor, what FactorVAE uses
    at test time) and via ``training_loss(x, returns)`` which runs the
    full encoder branch and returns the negative ELBO.
  - K = 8 latent factors (the FactorVAE paper's CSI 300 setting).
  - hidden_size = 64, num_portfolio = min(64, N_active) by default.
  - We strip the upstream Qlib dataloader entirely; the v2 baseline
    runner provides the panel.

The adapter mirrors the structure of ``master_adapter.MASTERAdapter`` so
the trainer can be a near-copy of ``train_master_v2.py``.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.baselines.vendored.factorvae import (
    AlphaLayer, BetaLayer, FactorDecoder, FactorEncoder,
    FactorPredictor, FactorVAE, FeatureExtractor,
)


@dataclass
class FactorVAEHyperparams:
    """FactorVAE architecture knobs.

    Defaults match the AAAI 2022 paper's CSI 300 reported configuration
    (K = 8 factors, GRU hidden 64) where stated; otherwise we follow the
    well-cited unofficial PyTorch implementation (x7jeon8gi/FactorVAE).
    """
    d_feat: int = 22                # F: number of panel features per ticker
    hidden_size: int = 64           # H: GRU hidden + alpha/beta head width
    num_factors: int = 8            # K: latent factor count (paper CSI 300)
    num_portfolio: int = 64         # M: soft portfolio set in factor encoder
    num_layers: int = 1             # GRU depth (paper uses 1 layer)


class FactorVAEAdapter(nn.Module):
    """FactorVAE wrapped for our (N_active, T, F) panel format.

    Public surface used by ``train_factorvae_v2.py``:
        forward(x_window)               -> (N,) prior-based score (test path)
        training_loss(x_window, target) -> (loss, y_hat) using posterior path

    The two paths share the feature extractor + alpha/beta head so the
    fp16 autocast and gradient clipping behaviour is identical to MASTER.
    """

    def __init__(self, hp: FactorVAEHyperparams):
        super().__init__()
        self.hp = hp
        feature_extractor = FeatureExtractor(
            num_latent=hp.d_feat,
            hidden_size=hp.hidden_size,
            num_layers=hp.num_layers,
        )
        factor_encoder = FactorEncoder(
            num_factors=hp.num_factors,
            num_portfolio=hp.num_portfolio,
            hidden_size=hp.hidden_size,
        )
        alpha_layer = AlphaLayer(hp.hidden_size)
        beta_layer = BetaLayer(hp.hidden_size, hp.num_factors)
        factor_decoder = FactorDecoder(alpha_layer, beta_layer)
        factor_predictor = FactorPredictor(hp.hidden_size, hp.num_factors)
        self.inner = FactorVAE(feature_extractor, factor_encoder,
                               factor_decoder, factor_predictor)

    def forward(self, x_window: torch.Tensor) -> torch.Tensor:
        """Test-time score per active ticker.

        Input  x_window: (N_active, T, F)
        Output y_hat:    (N_active,) deterministic prior-mean prediction.
        """
        stock_latent = self.inner.feature_extractor(x_window)
        pred_mu, pred_sigma = self.inner.factor_predictor(stock_latent)
        return self.inner.factor_decoder.predict_mu(stock_latent, pred_mu, pred_sigma)

    def training_loss(self, x_window: torch.Tensor,
                      target_z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the encoder branch to get the FactorVAE ELBO.

        Args:
            x_window: (N_active, T, F).
            target_z: (N_active,) per-day cross-sectionally z-scored target.

        Returns:
            (vae_loss, y_hat_score): vae_loss is a scalar tensor; y_hat_score
            is an (N_active,) deterministic score from the posterior mean
            reconstruction. The trainer can combine vae_loss with the cross
            sectional MSE (cs_mse_loss) on y_hat_score to keep the same loss
            surface as MASTER while still getting FactorVAE's ELBO regulariser.
        """
        stock_latent = self.inner.feature_extractor(x_window)
        factor_mu, factor_sigma = self.inner.factor_encoder(stock_latent, target_z)
        y_hat_post = self.inner.factor_decoder.predict_mu(
            stock_latent, factor_mu, factor_sigma,
        )  # (N,)
        # ELBO: reconstruction MSE + KL(posterior || prior).
        import torch.nn.functional as F
        target_b = target_z.view(-1)
        recon_loss = F.mse_loss(y_hat_post, target_b)
        pred_mu, pred_sigma = self.inner.factor_predictor(stock_latent)
        kl = self.inner.kl_divergence(factor_mu, factor_sigma, pred_mu, pred_sigma)
        vae_loss = recon_loss + kl
        return vae_loss, y_hat_post


__all__ = ["FactorVAEAdapter", "FactorVAEHyperparams"]
