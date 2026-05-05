"""RT-CSGA head: Regime-Tempered Cross-Sectional Graph Attention.

Per spec Section 5.4 of `docs/specs/rt_csga_spec.md`. Replaces
Sections E (macro head), F (rate memory), and G (deterministic
duration graph) of DOW-epiSTAR v2.3 with a single tempered
graph-attention primitive over hand-engineered duration features.

Forward pass (one trading day, N active tickers):

    pair[i, j] = LinearProj([abs(d_i - d_j); d_i * d_j])     in R^F_pair
    e[i, j]    = sum_k pair[i, j, k] * w_e[k] + b_e          scalar logit
    e[i, i]    = -inf                                          mask diagonal
    tau        = softplus(a + b * r_t) + 1e-3                 from RegimeTemperature
    alpha      = softmax(e / tau, dim=-1)                     [N, N]
    V_proj     = h_t @ V                                       [N, d_attn]
    m          = alpha @ V_proj                                [N, d_attn]
    score      = m @ v_score                                   [N]

The structural prior is encoded in the hand-engineered duration
features d_t (10 dims, same as DOW v2.3 Section G). The
regime-conditioned temperature controls the entropy of the attention
distribution: low rate-vol -> sharp attention, high rate-vol -> diffuse
attention.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from src.v2.model.auxiliary_branch import AuxiliaryBranch
from src.v2.model.regime_temperature import RegimeTemperature


@dataclass
class RTCSGAConfig:
    """Hyperparameters for RT-CSGA head."""

    d: int = 128                                # OW v1 hidden dim
    d_attn: int = 64                            # attention head dim
    f_single: int = 10                          # per-ticker duration features
    f_pair: int = 10                            # pairwise feature dim
    use_regime_temperature: bool = True         # if False, fixed tau scalar
    fixed_tau: float = 1.0
    a_init: float = 0.0
    b_init: float = 1.0
    use_random_edges: bool = False              # ablation A3
    random_edge_seed: int = 0


class RTCSGAHead(nn.Module):
    """Regime-tempered cross-sectional graph-attention head."""

    def __init__(self, cfg: RTCSGAConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Pairwise feature projection: 2 * F_single -> F_pair.
        self.pair_proj = nn.Linear(2 * cfg.f_single, cfg.f_pair)

        # Edge logit weights.
        self.w_e = nn.Parameter(torch.empty(cfg.f_pair))
        self.b_e = nn.Parameter(torch.zeros(()))
        nn.init.kaiming_uniform_(self.w_e.unsqueeze(0))

        # Value projection (D -> d_attn).
        self.V = nn.Linear(cfg.d, cfg.d_attn, bias=False)

        # Structural ranking head (d_attn -> 1).
        self.v_score = nn.Linear(cfg.d_attn, 1, bias=False)

        # Regime temperature.
        if cfg.use_regime_temperature:
            self.regime_temp = RegimeTemperature(
                a_init=cfg.a_init, b_init=cfg.b_init,
            )
            self.register_buffer("fixed_tau_buf", torch.tensor(0.0))  # unused
        else:
            self.regime_temp = None  # type: ignore[assignment]
            self.register_buffer(
                "fixed_tau_buf", torch.tensor(cfg.fixed_tau, dtype=torch.float32),
            )

    def build_pairwise_features(self, d_t: Tensor) -> Tensor:
        """[N, F_single] -> [N, N, F_pair].

        Pairwise construction = LinearProj(concat([|d_i - d_j|, d_i * d_j])).
        """
        diff = (d_t.unsqueeze(1) - d_t.unsqueeze(0)).abs()
        prod = d_t.unsqueeze(1) * d_t.unsqueeze(0)
        pair_raw = torch.cat([diff, prod], dim=-1)
        return self.pair_proj(pair_raw)

    def edge_logits(self, pair: Tensor) -> Tensor:
        """[N, N, F_pair] -> [N, N], diagonal masked to -inf."""
        e = (pair * self.w_e).sum(dim=-1) + self.b_e
        n = e.shape[0]
        eye_mask = torch.eye(n, dtype=torch.bool, device=e.device)
        return e.masked_fill(eye_mask, float("-inf"))

    def random_edge_logits(self, n: int, device: torch.device) -> Tensor:
        """Frozen random Gaussian edge logits (ablation A3)."""
        gen = torch.Generator(device=device)
        gen.manual_seed(self.cfg.random_edge_seed)
        e = torch.randn(n, n, device=device, generator=gen)
        eye_mask = torch.eye(n, dtype=torch.bool, device=device)
        return e.masked_fill(eye_mask, float("-inf"))

    def get_tau(self, r_t: Tensor) -> Tensor:
        """Return scalar tau (>= 1e-3)."""
        if self.cfg.use_regime_temperature:
            return self.regime_temp(r_t)
        return self.fixed_tau_buf

    def forward(
        self,
        h_t: Tensor,                      # [N, D]
        d_t: Tensor,                      # [N, F_single]
        r_t: Tensor,                      # scalar regime indicator
    ) -> tuple[Tensor, dict]:
        """Per-day forward. Returns (score [N], diagnostics)."""
        n = h_t.shape[0]
        if self.cfg.use_random_edges:
            e = self.random_edge_logits(n, h_t.device)
        else:
            pair = self.build_pairwise_features(d_t)
            e = self.edge_logits(pair)
        tau = self.get_tau(r_t)
        alpha = F.softmax(e / tau, dim=-1)
        v_proj = self.V(h_t)
        m = alpha @ v_proj
        score_struct = self.v_score(m).squeeze(-1)
        diag = {
            "alpha": alpha.detach(),
            "tau": tau.detach() if isinstance(tau, Tensor) else torch.tensor(float(tau)),
            "alpha_entropy": -(alpha * (alpha + 1e-12).log()).sum(dim=-1).mean().detach(),
        }
        return score_struct, diag


@dataclass
class RTCSGAModelConfig:
    """Hyperparameters for the full RT-CSGA wrapper."""

    head: RTCSGAConfig = RTCSGAConfig()
    use_auxiliary_branch: bool = True
    aux_hidden_dim: int = 64
    aux_gate_init: float = -5.0


class RTCSGAModel(nn.Module):
    """Full RT-CSGA model: OW backbone + RT-CSGA head + optional aux branch.

    The OW backbone's per-ticker hidden state is consumed externally
    (the trainer extracts ``z_final`` from the OW v1 forward pass).
    This wrapper is the *additive head* applied on top of OW v1's
    score. Use the trainer to wire OW v1 + this head together.
    """

    def __init__(self, cfg: RTCSGAModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.head = RTCSGAHead(cfg.head)
        if cfg.use_auxiliary_branch:
            self.aux_branch = AuxiliaryBranch(
                in_dim=cfg.head.d, hidden_dim=cfg.aux_hidden_dim,
                gate_init=cfg.aux_gate_init,
            )
        else:
            self.aux_branch = None  # type: ignore[assignment]

    def forward(
        self,
        h_t: Tensor,
        d_t: Tensor,
        r_t: Tensor,
    ) -> tuple[Tensor, Tensor, dict]:
        """Return (score_struct [N], score_aux [N], diagnostics)."""
        score_struct, diag = self.head(h_t, d_t, r_t)
        if self.aux_branch is not None:
            score_aux = self.aux_branch(h_t)
            diag["score_aux_max_abs"] = score_aux.abs().max().detach() \
                if score_aux.numel() > 0 else torch.zeros(())
        else:
            score_aux = torch.zeros(h_t.shape[0], device=h_t.device, dtype=h_t.dtype)
        return score_struct, score_aux, diag


__all__ = [
    "RTCSGAConfig",
    "RTCSGAModelConfig",
    "RTCSGAHead",
    "RTCSGAModel",
]
