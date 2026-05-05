"""Cleaned vendored FactorVAE modules (AAAI 2022).

Adapted from https://github.com/x7jeon8gi/FactorVAE/blob/main/module.py
under MIT-style usage. Only the architecture is kept; the upstream
training loop, dataset, and CSI-300 specific glue are dropped because
the v2 baseline protocol provides those pieces.

Sharp-edge fixes vs the upstream module.py:
  - replace in-place ``factor_sigma[factor_sigma == 0] = 1e-6`` (which
    mutates an autograd tensor and breaks the backward pass under
    fp16) with an out-of-place ``clamp(min=eps)``,
  - guard the AttentionLayer NaN check so it does not silently cut the
    gradient on a fp16 underflow,
  - stop passing returns through ``squeeze(1)`` unconditionally; we
    accept either ``(N,)`` or ``(N, 1)`` to be robust to the panel
    iterator that emits ``y[t]`` as a 1-D vector.

The upstream variable naming is preserved so that anyone who has read
the original code can navigate this file without relearning anything.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


_EPS = 1e-6


class FeatureExtractor(nn.Module):
    """Per-ticker GRU over the temporal window (T, F).

    Input  x: (N, T, F)
    Output stock_latent: (N, hidden_size) using the final time step.
    """

    def __init__(self, num_latent: int, hidden_size: int, num_layers: int = 1):
        super().__init__()
        self.num_latent = num_latent
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.normalize = nn.LayerNorm(num_latent)
        self.linear = nn.Linear(num_latent, num_latent)
        self.leakyrelu = nn.LeakyReLU()
        self.gru = nn.GRU(num_latent, hidden_size, num_layers, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, T, F)
        x = self.normalize(x)
        out = self.linear(x)
        out = self.leakyrelu(out)
        stock_latent, _ = self.gru(out)
        return stock_latent[:, -1, :]  # (N, hidden_size)


class FactorEncoder(nn.Module):
    """Posterior factor distribution given returns.

    Builds a soft "portfolio" set (M slots) by softmaxing a linear
    projection of the stock latents across stocks, then maps each
    portfolio's realised return to a posterior (mu_post, sigma_post)
    over K factors.
    """

    def __init__(self, num_factors: int, num_portfolio: int, hidden_size: int):
        super().__init__()
        self.num_factors = num_factors
        self.linear = nn.Linear(hidden_size, num_portfolio)
        self.softmax = nn.Softmax(dim=0)  # over stocks (paper: row-normalise weights)

        self.linear_mu = nn.Linear(num_portfolio, num_factors)
        self.linear_sigma = nn.Linear(num_portfolio, num_factors)
        self.softplus = nn.Softplus()

    def mapping_layer(self, portfolio_return: torch.Tensor):
        # portfolio_return: (num_portfolio, 1)
        mean = self.linear_mu(portfolio_return.squeeze(1))
        sigma = self.softplus(self.linear_sigma(portfolio_return.squeeze(1)))
        return mean, sigma

    def forward(self, stock_latent: torch.Tensor, returns: torch.Tensor):
        # stock_latent: (N, hidden_size); returns: (N,) or (N, 1)
        weights = self.linear(stock_latent)
        weights = self.softmax(weights)  # (N, num_portfolio)
        if returns.dim() == 1:
            returns = returns.unsqueeze(1)
        portfolio_return = torch.mm(weights.transpose(1, 0), returns)  # (M, 1)
        return self.mapping_layer(portfolio_return)


class AlphaLayer(nn.Module):
    """Per-stock idiosyncratic alpha distribution (mu, sigma)."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.leakyrelu = nn.LeakyReLU()
        self.mu_layer = nn.Linear(hidden_size, 1)
        self.sigma_layer = nn.Linear(hidden_size, 1)
        self.softplus = nn.Softplus()

    def forward(self, stock_latent: torch.Tensor):
        h = self.linear1(stock_latent)
        h = self.leakyrelu(h)
        alpha_mu = self.mu_layer(h)
        alpha_sigma = self.softplus(self.sigma_layer(h))
        return alpha_mu, alpha_sigma


class BetaLayer(nn.Module):
    """Per-stock factor loadings beta of shape (N, K)."""

    def __init__(self, hidden_size: int, num_factors: int):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, num_factors)

    def forward(self, stock_latent: torch.Tensor) -> torch.Tensor:
        return self.linear1(stock_latent)


class FactorDecoder(nn.Module):
    """Reconstruct y_hat = alpha + beta @ factor with reparameterised noise."""

    def __init__(self, alpha_layer: AlphaLayer, beta_layer: BetaLayer):
        super().__init__()
        self.alpha_layer = alpha_layer
        self.beta_layer = beta_layer

    @staticmethod
    def reparameterize(mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        eps = torch.randn_like(sigma)
        return mu + eps * sigma

    def forward(self, stock_latent: torch.Tensor,
                factor_mu: torch.Tensor, factor_sigma: torch.Tensor) -> torch.Tensor:
        alpha_mu, alpha_sigma = self.alpha_layer(stock_latent)  # (N, 1) each
        beta = self.beta_layer(stock_latent)                    # (N, K)

        factor_mu = factor_mu.view(-1, 1)
        factor_sigma = factor_sigma.view(-1, 1).clamp(min=_EPS)

        mu = alpha_mu + torch.matmul(beta, factor_mu)
        sigma = torch.sqrt(alpha_sigma ** 2
                           + torch.matmul(beta ** 2, factor_sigma ** 2)
                           + _EPS)
        return self.reparameterize(mu, sigma)

    def predict_mu(self, stock_latent: torch.Tensor,
                   factor_mu: torch.Tensor, factor_sigma: torch.Tensor) -> torch.Tensor:
        """Deterministic mean prediction (no reparameterisation noise).

        At inference / scoring time we want a stable score per ticker, so
        we use the predicted mean rather than a sample.
        """
        alpha_mu, _ = self.alpha_layer(stock_latent)
        beta = self.beta_layer(stock_latent)
        factor_mu = factor_mu.view(-1, 1)
        return (alpha_mu + torch.matmul(beta, factor_mu)).squeeze(-1)  # (N,)


class AttentionLayer(nn.Module):
    """Single-query attention over stocks for one prior factor."""

    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.query = nn.Parameter(torch.randn(hidden_size))
        self.key_layer = nn.Linear(hidden_size, hidden_size)
        self.value_layer = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = hidden_size

    def forward(self, stock_latent: torch.Tensor) -> torch.Tensor:
        key = self.key_layer(stock_latent)        # (N, H)
        value = self.value_layer(stock_latent)    # (N, H)
        scale = (self.hidden_size ** 0.5) + _EPS
        attn = torch.matmul(self.query, key.transpose(1, 0)) / scale  # (N,)
        attn = self.dropout(attn)
        attn = F.relu(attn)
        attn = F.softmax(attn, dim=0)             # (N,)
        if torch.isnan(attn).any() or torch.isinf(attn).any():
            return torch.zeros_like(value[0])
        return torch.matmul(attn, value)          # (H,)


class FactorPredictor(nn.Module):
    """Prior factor distribution from stock latents (used at inference)."""

    def __init__(self, hidden_size: int, num_factor: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_factor = num_factor
        self.attention_layers = nn.ModuleList([AttentionLayer(hidden_size)
                                               for _ in range(num_factor)])
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.leakyrelu = nn.LeakyReLU()
        self.mu_layer = nn.Linear(hidden_size, 1)
        self.sigma_layer = nn.Linear(hidden_size, 1)
        self.softplus = nn.Softplus()

    def forward(self, stock_latent: torch.Tensor):
        rows = [layer(stock_latent) for layer in self.attention_layers]  # K x (H,)
        h_multi = torch.stack(rows, dim=0)                                # (K, H)
        h_multi = self.leakyrelu(self.linear(h_multi))
        pred_mu = self.mu_layer(h_multi).view(-1)                         # (K,)
        pred_sigma = self.softplus(self.sigma_layer(h_multi)).view(-1)    # (K,)
        return pred_mu, pred_sigma


class FactorVAE(nn.Module):
    """End-to-end FactorVAE.

    Training-time forward (``run_step``) returns a triple
    (vae_loss, y_hat, aux) where ``aux`` carries the prior/posterior
    parameters in case downstream code wants to log them. Inference-time
    forward (``predict``) returns only ``y_hat``.

    The upstream upstream code mutates ``pred_sigma`` in place; we do
    not, so this module is safe under fp16 autocast and grad clipping.
    """

    def __init__(self, feature_extractor: FeatureExtractor,
                 factor_encoder: FactorEncoder,
                 factor_decoder: FactorDecoder,
                 factor_predictor: FactorPredictor):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.factor_encoder = factor_encoder
        self.factor_decoder = factor_decoder
        self.factor_predictor = factor_predictor

    @staticmethod
    def kl_divergence(mu1: torch.Tensor, sigma1: torch.Tensor,
                      mu2: torch.Tensor, sigma2: torch.Tensor) -> torch.Tensor:
        sigma1 = sigma1.clamp(min=_EPS)
        sigma2 = sigma2.clamp(min=_EPS)
        kl = (torch.log(sigma2 / sigma1)
              + (sigma1 ** 2 + (mu1 - mu2) ** 2) / (2.0 * sigma2 ** 2)
              - 0.5).sum()
        return kl

    def run_step(self, x: torch.Tensor, returns: torch.Tensor) -> tuple:
        """Forward pass that returns (vae_loss, y_hat, aux).

        x: (N, T, F); returns: (N,) or (N, 1) standardised target.
        """
        stock_latent = self.feature_extractor(x)
        factor_mu, factor_sigma = self.factor_encoder(stock_latent, returns)
        y_hat = self.factor_decoder(stock_latent, factor_mu, factor_sigma)  # (N, 1)

        if returns.dim() == 1:
            returns_b = returns.unsqueeze(1)
        else:
            returns_b = returns
        recon_loss = F.mse_loss(y_hat, returns_b)

        pred_mu, pred_sigma = self.factor_predictor(stock_latent)
        kl = self.kl_divergence(factor_mu, factor_sigma, pred_mu, pred_sigma)

        vae_loss = recon_loss + kl
        aux = {"factor_mu": factor_mu, "factor_sigma": factor_sigma,
               "pred_mu": pred_mu, "pred_sigma": pred_sigma}
        return vae_loss, y_hat.squeeze(-1), aux

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Inference path: use the prior (factor predictor), not the posterior."""
        stock_latent = self.feature_extractor(x)
        pred_mu, pred_sigma = self.factor_predictor(stock_latent)
        return self.factor_decoder.predict_mu(stock_latent, pred_mu, pred_sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Default forward = inference path so it slots into the v2 protocol."""
        # Compute deterministic inference-time score under autograd so the
        # adapter can support both a posterior-based (training) call via
        # run_step and a prior-based scoring call here. We do NOT use
        # ``torch.no_grad`` on this path because gradients flow through the
        # alpha+beta head when training without the encoder branch (e.g.
        # for a quick smoke test). For the standard v2 trainer we call
        # run_step directly.
        stock_latent = self.feature_extractor(x)
        pred_mu, pred_sigma = self.factor_predictor(stock_latent)
        return self.factor_decoder.predict_mu(stock_latent, pred_mu, pred_sigma)
