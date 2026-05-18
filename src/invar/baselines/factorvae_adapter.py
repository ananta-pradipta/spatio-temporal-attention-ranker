"""FactorVAE adapter for the LATTICE per-day batch contract.

Reference
---------
Duan, Y., Wang, L., Zhang, Q., Li, J. (2022). FactorVAE: A
Probabilistic Dynamic Factor Model Based on Variational Autoencoder
for Predicting Cross-Sectional Stock Returns. AAAI 2022.

Implementation source
---------------------
Vendored at ``src/baselines/vendored/factorvae`` (the authors did not
release code; this is the unofficial reference at
https://github.com/x7jeon8gi/FactorVAE with sharp-edge fixes).

Adaptation
----------
- Input:  ``features (N, L=60, F=26)`` directly fed to the GRU
  feature extractor (the upstream took T=20; we use the LATTICE
  60-day lookback with no truncation).
- Output: ``y_hat`` of shape ``(N,)`` plus zero placeholders for the
  unused ``regime_logits`` and ``vol_hat`` slots so the model fits
  the InVAR baseline harness's forward dict contract.
- Macro input is ignored (the paper does not condition on macro;
  including it would deviate from "use their original code").
- Hyperparameters match the AAAI 2022 CSI 300 setting:
  K = 8 latent factors, hidden = 64, num_portfolio = 64.

Training-time behavior
----------------------
``forward`` returns the inference-path mean ``y_hat`` and additionally
stores the ELBO components on the module (``self._last_vae_loss``,
``self._last_factor_mu``, etc.) so the trainer can read the VAE loss
without running a second forward pass. The trainer's total objective is

    L_total = L_rank(y_hat, y_cs)  +  lambda_vae * L_vae

where ``L_rank`` is the InVAR cross-sectional ranking loss and
``L_vae`` is the FactorVAE ELBO. ``lambda_vae`` defaults to 0.10 to
keep the score-head dominant for ranking while still training the
factor encoder; this matches the v2 baseline protocol.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from src.baselines.vendored.factorvae import (
    AlphaLayer, BetaLayer, FactorDecoder, FactorEncoder,
    FactorPredictor, FactorVAE, FeatureExtractor,
)


@dataclass
class FactorVAEConfig:
    """LATTICE-tuned FactorVAE knobs (AAAI 2022 CSI 300 defaults)."""

    n_features: int = 26
    lookback: int = 60
    hidden_size: int = 64
    num_factors: int = 8
    num_portfolio: int = 64
    gru_layers: int = 1
    lambda_vae: float = 0.10
    n_offline_regimes: int = 8


class FactorVAEAdapter(nn.Module):
    """FactorVAE wrapper conforming to the InVAR baseline harness."""

    def __init__(self, cfg: FactorVAEConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or FactorVAEConfig()
        feat = FeatureExtractor(
            num_latent=self.cfg.n_features,
            hidden_size=self.cfg.hidden_size,
            num_layers=self.cfg.gru_layers,
        )
        enc = FactorEncoder(
            num_factors=self.cfg.num_factors,
            num_portfolio=self.cfg.num_portfolio,
            hidden_size=self.cfg.hidden_size,
        )
        alpha = AlphaLayer(self.cfg.hidden_size)
        beta = BetaLayer(self.cfg.hidden_size, self.cfg.num_factors)
        dec = FactorDecoder(alpha_layer=alpha, beta_layer=beta)
        pred = FactorPredictor(
            hidden_size=self.cfg.hidden_size,
            num_factor=self.cfg.num_factors,
        )
        self.factorvae = FactorVAE(
            feature_extractor=feat, factor_encoder=enc,
            factor_decoder=dec, factor_predictor=pred,
        )
        # Aux placeholders to satisfy the harness's loss contract.
        # regime_logits and vol_hat are unused; we just emit zeros.
        self._last_vae_loss: Tensor | None = None
        self._last_factor_aux: dict | None = None
        # Day-level regime classifier head over the mean stock latent
        # (so the harness's hybrid_loss has a finite-shape regime
        # logits tensor to ignore via weight=0; not used in training).
        self.regime_classifier = nn.Linear(
            self.cfg.hidden_size, self.cfg.n_offline_regimes,
        )

    def forward(
        self, features: Tensor, macro: Tensor, mask: Tensor,
        return_attn: bool = False, y_cs: Tensor | None = None,
        **_unused: object,
    ) -> dict:
        """Args:
            features : ``(N, L, F)``.
            macro    : ``(L, F_macro)`` (ignored; paper does not use macro).
            mask     : ``(N,)`` bool.
            y_cs     : optional ``(N,)`` cross-sectional z-scored target.
                        When supplied (training mode), the model runs the
                        full ELBO and stores ``vae_loss``. When None
                        (eval), only the prior path is run.

        Returns dict with ``y_hat``, ``regime_logits``, ``vol_hat``.
        ``vae_loss`` is stored on the module as ``self._last_vae_loss``.
        """
        N = features.shape[0]
        m = mask.float()

        # Run feature extractor once (used by both training and eval).
        stock_latent = self.factorvae.feature_extractor(features)  # (N, H)

        if y_cs is not None and self.training:
            # Training: posterior + KL.
            factor_mu, factor_sigma = self.factorvae.factor_encoder(
                stock_latent, y_cs,
            )
            y_hat_dec = self.factorvae.factor_decoder(
                stock_latent, factor_mu, factor_sigma,
            )                                                      # (N, 1)
            recon = torch.nn.functional.mse_loss(
                y_hat_dec, y_cs.view(-1, 1),
            )
            pred_mu, pred_sigma = self.factorvae.factor_predictor(stock_latent)
            kl = FactorVAE.kl_divergence(
                factor_mu, factor_sigma, pred_mu, pred_sigma,
            )
            self._last_vae_loss = recon + kl
            self._last_factor_aux = {
                "factor_mu": factor_mu.detach(),
                "factor_sigma": factor_sigma.detach(),
                "pred_mu": pred_mu.detach(),
                "pred_sigma": pred_sigma.detach(),
            }
            # The score for the ranking loss is the deterministic
            # prior-based prediction (what we use at test time too),
            # not the noisy decoder sample. This keeps train and test
            # path consistent for the cross-sectional ranking signal.
            y_hat = self.factorvae.factor_decoder.predict_mu(
                stock_latent, pred_mu, pred_sigma,
            )                                                      # (N,)
        else:
            self._last_vae_loss = None
            self._last_factor_aux = None
            pred_mu, pred_sigma = self.factorvae.factor_predictor(stock_latent)
            y_hat = self.factorvae.factor_decoder.predict_mu(
                stock_latent, pred_mu, pred_sigma,
            )                                                      # (N,)

        y_hat = y_hat * m
        # Day-level regime logits: average pooled stock latent.
        denom = m.sum().clamp(min=1.0)
        latent_mean = (stock_latent * m.unsqueeze(-1)).sum(dim=0) / denom
        regime_logits = self.regime_classifier(latent_mean)        # (n_offline_regimes,)
        vol_hat = torch.zeros_like(y_hat)
        return {
            "y_hat": y_hat, "regime_logits": regime_logits,
            "vol_hat": vol_hat,
        }


__all__ = ["FactorVAEAdapter", "FactorVAEConfig"]
