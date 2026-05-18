"""DynamicBankController for InVAR-v6.

Per-day scalar weight on the regime retrieval bank's contribution.
Replaces a hard on/off bank decision with a learned/deterministic
hybrid that opens the bank during stress regimes and closes it on
high-novelty days.

Inputs (Section 6.4):
  - macro_state          : ``(B, macro_state_dim)``
  - retrieval distance   : day-level scalar (mean of top-K retrieval
                            scores or analogous novelty proxy)
  - retrieval entropy    : day-level scalar
  - bank value norm      : day-level scalar
  - stress features      : ``(B, stats_dim)`` z-scored on train fold
  - active count         : day-level scalar

Modes:
  - ``deterministic`` : closed-form rule from stress and novelty
                         z-scores; no learned parameters in the gate
                         (linear MLP weights still exist for stability
                         but are unused).
  - ``learned``       : MLP over [macro_state, bank_stats, stress]
                         with sigmoid output.
  - ``hybrid``        : deterministic baseline plus learned residual
                         delta (default).

Output: bank_weight in ``[min_weight, max_weight]``, shape ``(B, 1)``.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DynamicBankController(nn.Module):
    def __init__(
        self,
        macro_state_dim: int = 64,
        stats_dim: int = 6,
        hidden_dim: int = 64,
        init_bias: float = 0.0,
        min_weight: float = 0.05,
        max_weight: float = 1.00,
        mode: str = "hybrid",
    ) -> None:
        super().__init__()
        if mode not in ("deterministic", "learned", "hybrid"):
            raise ValueError(f"unknown mode: {mode!r}")
        self.mode = mode
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.init_bias = init_bias
        self.macro_state_dim = macro_state_dim
        self.stats_dim = stats_dim

        # Bank stats are a fixed 4-vector per the spec:
        # [retrieval_distance_z, retrieval_entropy_z,
        #  bank_value_norm_z, active_count_z].
        self.bank_stats_dim = 4
        in_dim = macro_state_dim + self.bank_stats_dim + stats_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.constant_(self.mlp[-1].bias, init_bias)

    def _deterministic_score(
        self, retrieval_distance_z: torch.Tensor,
        retrieval_entropy_z: torch.Tensor,
        stress_features: torch.Tensor,
    ) -> torch.Tensor:
        novelty = retrieval_distance_z + retrieval_entropy_z      # (B,)
        stress = stress_features.mean(dim=-1)                      # (B,)
        return 0.5 * stress - 0.5 * novelty                        # (B,)

    def forward(
        self,
        macro_state: torch.Tensor,
        bank_stats: dict,
        stress_features: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """Args:
            macro_state    : ``(B, macro_state_dim)``.
            bank_stats     : dict with keys
                ``retrieval_distance_z``, ``retrieval_entropy_z``,
                ``bank_value_norm_z``, ``active_count_z``; each ``(B,)``.
            stress_features: ``(B, stats_dim)`` z-scored.

        Returns ``(bank_weight, debug)`` with ``bank_weight : (B, 1)``.
        """
        if macro_state.dim() == 1:
            macro_state = macro_state.unsqueeze(0)
        B = macro_state.shape[0]
        rd = bank_stats["retrieval_distance_z"].view(B)
        re = bank_stats["retrieval_entropy_z"].view(B)
        bn = bank_stats["bank_value_norm_z"].view(B)
        ac = bank_stats["active_count_z"].view(B)
        if stress_features.dim() == 1:
            stress_features = stress_features.unsqueeze(0)

        det = self._deterministic_score(rd, re, stress_features)

        if self.mode == "deterministic":
            raw = det
        elif self.mode == "learned":
            stats_vec = torch.stack([rd, re, bn, ac], dim=-1)         # (B, 4)
            x = torch.cat([macro_state, stats_vec, stress_features], dim=-1)
            raw = self.mlp(x).squeeze(-1)
        else:  # hybrid
            stats_vec = torch.stack([rd, re, bn, ac], dim=-1)
            x = torch.cat([macro_state, stats_vec, stress_features], dim=-1)
            learned_delta = 0.25 * torch.tanh(self.mlp(x).squeeze(-1))
            raw = det + learned_delta

        bank_weight = torch.sigmoid(raw).clamp(
            min=self.min_weight, max=self.max_weight,
        ).unsqueeze(-1)                                                # (B, 1)
        debug = {
            "deterministic_raw": det.detach(),
            "novelty_z": (rd + re).detach(),
            "stress_score": stress_features.mean(dim=-1).detach(),
            "bank_weight": bank_weight.detach(),
        }
        return bank_weight, debug


__all__ = ["DynamicBankController"]
