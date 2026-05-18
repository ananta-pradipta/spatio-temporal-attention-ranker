"""InVAR-STAR model: variate-as-token encoder over 26 stock + 24 macro tokens
with a stochastic self-throttling scalar gate beta_t and a mixture-of-experts
ranking head.

Reproducibility seeds: 42, 43, 44, 45, 46.
Design document: docs/invar_star_design.md (loaded 2026-05-10).
"""
from __future__ import annotations

import math
import random
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F


def set_global_seed(seed: int) -> None:
    """Set seeds across torch, numpy, and python's random for reproducibility.

    Args:
        seed: integer seed in {42, 43, 44, 45, 46} for the QE replicates.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MacroVariateBank(nn.Module):
    """Embeds 26 stock-feature lookbacks and 24 macro lookbacks as variate tokens.

    Identical embedding rule to iTransformer (Liu et al., ICLR 2024): each
    variate's L-length lookback is linearly projected to a single d-dimensional
    token. Stock tokens occupy indices 0..n_stock-1; macro tokens occupy
    indices n_stock..n_stock+n_macro-1.
    """

    def __init__(self, lookback: int = 60, n_stock_feats: int = 26,
                 n_macro_feats: int = 24, d_model: int = 128) -> None:
        super().__init__()
        self.lookback = lookback
        self.n_stock = n_stock_feats
        self.n_macro = n_macro_feats
        self.d_model = d_model
        self.embed = nn.Linear(lookback, d_model)

    def forward(self, x_stock: Tensor, x_macro: Tensor) -> Tensor:
        """Build the (n_stock + n_macro)-token variate bank.

        Args:
            x_stock: shape (B, n_stock, L). Per-stock lookback features.
            x_macro: shape (B, n_macro, L). Daily macro lookback, broadcast over B.

        Returns:
            tokens: shape (B, n_stock + n_macro, d_model).
        """
        h_stock = self.embed(x_stock)
        h_macro = self.embed(x_macro)
        return torch.cat([h_stock, h_macro], dim=1)


class SelfThrottlingGate(nn.Module):
    """Daily scalar gate beta_t in (0, 1) with Concrete (binary Gumbel) relaxation.

    Training time: additive logistic noise before the sigmoid (Maddison et al.,
    2017). Inference time: deterministic sigmoid (no noise). Temperature tau
    is annealed externally by the training loop (1.0 -> 0.1 across epochs).

    Critical parameterization: beta_t reflects the regime (a market-wide
    property), so for our per-day batch where B equals the day's active
    cross-section size, all B samples share the same macro lookback and the
    deterministic part of beta is identical across B. Per-sample logistic
    noise during training adds Concrete stochasticity; at inference noise is
    zero and beta is exactly per-day.
    """

    def __init__(self, phi_dim: int = 64, hidden: int = 64,
                 init_bias: float = 0.5) -> None:
        super().__init__()
        self.phi_dim = phi_dim
        self.mlp = nn.Sequential(
            nn.Linear(phi_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        # Positive bias on the final linear seeds beta on the macro-engaged
        # side at init without saturating the eval-time sigmoid at tau=0.1.
        # init_bias = 0.5 -> sigmoid(0.5 / 1.0) = 0.62 at train init,
        # sigmoid(0.5 / 0.1) = 0.993 at eval (steep but not saturated, so
        # phi-conditional logits can still move beta meaningfully).
        # Counteracts the bimodal-prior pull toward beta=0 observed in the
        # first 6-epoch smoke (2026-05-10) without re-saturating at beta=1.
        with torch.no_grad():
            self.mlp[-1].bias.fill_(init_bias)

    def forward(self, phi_t: Tensor, tau: float, training: bool) -> Tensor:
        """Compute beta_t.

        Args:
            phi_t: shape (B, phi_dim). Daily macro-summary vector.
            tau: temperature, annealed from 1.0 to 0.1.
            training: if True, inject logistic noise (Concrete relaxation).

        Returns:
            beta: shape (B, 1). Values in (0, 1).
        """
        logits = self.mlp(phi_t)
        if training:
            u = torch.rand_like(logits).clamp(1e-6, 1.0 - 1e-6)
            noise = torch.log(u) - torch.log(1.0 - u)
            logits = logits + noise
        return torch.sigmoid(logits / tau)


class ThrottledVariateAttention(nn.Module):
    """Multi-head self-attention over variate tokens with macro-edge throttling.

    Adds log(beta_t) to the pre-softmax logits on the stock-row, macro-column
    block of every attention head. When beta_t equals 0, macro tokens become
    attention-inert from the stock tokens' perspective and the forward pass
    over stock tokens reduces to vanilla iTransformer (Strand A of the design
    doc Section 5).
    """

    def __init__(self, d_model: int, n_heads: int,
                 n_stock: int = 26, n_macro: int = 24) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.n_stock = n_stock
        self.n_macro = n_macro
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, h: Tensor, beta: Tensor) -> Tensor:
        """Forward pass.

        Args:
            h: shape (B, N, d_model) where N equals n_stock + n_macro.
            beta: shape (B, 1) in (0, 1). Daily throttle.

        Returns:
            out: shape (B, N, d_model).
        """
        B, N, D = h.shape
        qkv = self.qkv(h).reshape(B, N, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        log_beta = torch.log(beta.clamp_min(1e-8))
        bias = torch.zeros(B, 1, N, N, device=h.device, dtype=h.dtype)
        bias[:, :, : self.n_stock, self.n_stock:] = log_beta.unsqueeze(-1).unsqueeze(-1)
        scores = scores + bias
        attn = F.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.out(out)


class InVARSTARBlock(nn.Module):
    """One iTransformer-style encoder block with throttled cross-variate attention.

    Pre-norm residual layout: LN -> ThrottledAttention -> add; LN -> FFN -> add.
    FFN width is ffn_mult * d_model (default 4).
    """

    def __init__(self, d_model: int, n_heads: int, ffn_mult: int = 4,
                 n_stock: int = 26, n_macro: int = 24) -> None:
        super().__init__()
        self.attn = ThrottledVariateAttention(d_model, n_heads, n_stock, n_macro)
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.GELU(),
            nn.Linear(ffn_mult * d_model, d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, h: Tensor, beta: Tensor) -> Tensor:
        h = h + self.attn(self.ln1(h), beta)
        h = h + self.ffn(self.ln2(h))
        return h


class MoERankingHead(nn.Module):
    """K small expert MLPs with noisy top-k routing and expert dropout.

    Router input: concat(z, beta_t * macro_proj(phi)). When beta_t equals 0,
    the router input loses macro dependence by construction (the macro context
    is multiplied by zero before concatenation), so the router collapses to a
    stock-conditional softmax over experts.
    """

    def __init__(self, d_model: int, phi_dim: int, n_experts: int = 4,
                 top_k: int = 2, noise_std: float = 0.5,
                 expert_dropout: float = 0.1) -> None:
        super().__init__()
        if top_k > n_experts:
            raise ValueError("top_k must be <= n_experts")
        self.d_model = d_model
        self.phi_dim = phi_dim
        self.n_experts = n_experts
        self.top_k = top_k
        self.noise_std = noise_std
        self.expert_dropout = expert_dropout
        self.macro_proj = nn.Linear(phi_dim, d_model)
        self.router = nn.Linear(2 * d_model, n_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, 1),
            )
            for _ in range(n_experts)
        ])

    def forward(self, z: Tensor, phi: Tensor, beta: Tensor,
                training: bool) -> Tuple[Tensor, Tensor]:
        """Run the MoE head.

        Args:
            z: shape (B, d_model). Pooled stock-token embedding.
            phi: shape (B, phi_dim). Macro summary.
            beta: shape (B, 1). Throttle.
            training: whether to apply noisy routing and expert dropout.

        Returns:
            y_hat: shape (B, 1). Predicted return score.
            route_probs: shape (B, n_experts). For load-balancing loss.
        """
        macro_ctx = beta * self.macro_proj(phi)
        router_in = torch.cat([z, macro_ctx], dim=-1)
        logits = self.router(router_in)
        if training and self.noise_std > 0:
            logits = logits + self.noise_std * torch.randn_like(logits)
        if training and self.expert_dropout > 0:
            # Drop ONE expert across the entire minibatch with probability
            # expert_dropout. Matches "one expert randomly masked per
            # minibatch" in the design doc. Per-sample Bernoulli on
            # individual experts would occasionally mask ALL n_experts at
            # once for a given sample (P = expert_dropout ** n_experts),
            # making the top-k softmax produce NaN.
            if torch.rand((), device=logits.device).item() < self.expert_dropout:
                drop_idx = int(torch.randint(0, self.n_experts, (1,)).item())
                drop_col = torch.full_like(logits[:, drop_idx], float("-inf"))
                logits = logits.clone()
                logits[:, drop_idx] = drop_col
        top_logits, top_idx = logits.topk(self.top_k, dim=-1)
        top_w = F.softmax(top_logits, dim=-1)
        outs = torch.stack([e(z).squeeze(-1) for e in self.experts], dim=-1)
        gathered = torch.gather(outs, 1, top_idx)
        y_hat = (gathered * top_w).sum(dim=-1, keepdim=True)
        safe_logits = logits.masked_fill(logits == float("-inf"), -1.0e9)
        route_probs = F.softmax(safe_logits, dim=-1)
        return y_hat, route_probs


class InVARSTAR(nn.Module):
    """Full InVAR-STAR model.

    Composes the variate bank, the self-throttling gate, the encoder stack,
    and the MoE ranking head. `phi_proj` is pre-declared in __init__
    (n_macro * 4 -> phi_dim), no lazy init.
    """

    def __init__(self, lookback: int = 60, d_model: int = 128, n_heads: int = 4,
                 n_layers: int = 3, n_experts: int = 4, top_k: int = 2,
                 phi_dim: int = 64, n_stock: int = 26, n_macro: int = 24,
                 noise_std: float = 0.5, expert_dropout: float = 0.1) -> None:
        super().__init__()
        self.lookback = lookback
        self.d_model = d_model
        self.n_stock = n_stock
        self.n_macro = n_macro
        self.phi_dim = phi_dim
        self.phi_raw_dim = 4 * n_macro
        self.bank = MacroVariateBank(lookback, n_stock, n_macro, d_model)
        self.phi_proj = nn.Linear(self.phi_raw_dim, phi_dim)
        self.gate = SelfThrottlingGate(phi_dim=phi_dim)
        self.blocks = nn.ModuleList([
            InVARSTARBlock(d_model, n_heads, ffn_mult=4,
                           n_stock=n_stock, n_macro=n_macro)
            for _ in range(n_layers)
        ])
        self.head = MoERankingHead(
            d_model, phi_dim, n_experts=n_experts, top_k=top_k,
            noise_std=noise_std, expert_dropout=expert_dropout,
        )

    def build_phi(self, x_macro: Tensor) -> Tensor:
        """Build phi_dim-dimensional macro summary from (B, n_macro, L) lookback.

        phi_raw = concat(last(m), mean(m), std(m), ewm_std(m, halflife=10))
        of dimension 4 * n_macro = 96 for n_macro = 24. Projected to phi_dim
        by `self.phi_proj`.
        """
        last = x_macro[..., -1]
        mean = x_macro.mean(dim=-1)
        std = x_macro.std(dim=-1)
        L = x_macro.shape[-1]
        decay = 0.5 ** (1.0 / 10.0)
        weights = torch.tensor(
            [decay ** i for i in range(L)],
            device=x_macro.device, dtype=x_macro.dtype,
        ).flip(0)
        weights = weights / weights.sum()
        ewm_var = ((x_macro - mean.unsqueeze(-1)) ** 2 * weights).sum(dim=-1)
        ewm_std = ewm_var.clamp_min(1.0e-12).sqrt()
        phi_raw = torch.cat([last, mean, std, ewm_std], dim=-1)
        return self.phi_proj(phi_raw)

    def forward(self, x_stock: Tensor, x_macro: Tensor,
                tau: float = 1.0,
                fixed_beta: float | None = None) -> dict:
        """Forward pass.

        Args:
            x_stock: shape (B, n_stock, L).
            x_macro: shape (B, n_macro, L).
            tau: gate temperature (ignored when fixed_beta is set).
            fixed_beta: if not None, bypass the self-throttling gate and
                use this constant value of beta_t for all samples. Used by
                the A1 ablation in design doc Section 7.3 (gate disabled,
                macro always engaged when fixed_beta=1.0). The gate module
                is left in place but unused, so its parameters get no
                gradient and effectively stay frozen during training. The
                MoE head's `macro_proj(phi)` term still feeds the router
                (multiplied by the fixed beta), so the macro-conditional
                routing pathway is fully trained.

        Returns:
            dict with keys 'y_hat' (B, 1), 'beta' (B, 1), 'route_probs' (B, n_experts).
        """
        phi = self.build_phi(x_macro)
        if fixed_beta is not None:
            B = x_stock.shape[0]
            beta = torch.full(
                (B, 1), float(fixed_beta),
                device=x_stock.device, dtype=x_stock.dtype,
            )
        else:
            beta = self.gate(phi, tau=tau, training=self.training)
        h = self.bank(x_stock, x_macro)
        for blk in self.blocks:
            h = blk(h, beta)
        z = h[:, : self.n_stock].mean(dim=1)
        y_hat, route_probs = self.head(z, phi, beta, training=self.training)
        return {"y_hat": y_hat, "beta": beta, "route_probs": route_probs}


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
