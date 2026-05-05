"""DOW-epiSTAR v2: Duration-Aware Open-World epiSTAR.

Wraps OW-epiSTAR v1 (`src.v2.model.ow_epistar_v1.OWEpiSTARV1`) with:

    Section E (macro-duration head, v2.0):
        score_total += lambda_macro * score_duration
    Section F (rate-shock memory, v2.1):
        score_total += alpha_rate * score_rate
    Section G fallback (duration-aware graph, v2.2):
        graph topology = top-K of (w_corr*A_corr + w_duration*A_duration)
        with weights drawn from a softmax gate over macro_gate_state.

Final scoring:

    score_total[t, i] = score_idio[t, i]
                      + lambda_macro[t] * score_duration[t, i]
                      + alpha_rate[t]   * score_rate[t, i]

The duration-aware graph affects the OW backbone's neighbour list
upstream of score_idio. When `use_duration_graph` is False, the
backbone uses the existing reliability-shrunk correlation graph.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from src.v2.graph.duration_dynamic_edges import GraphSourceGate
from src.v2.model.duration_exposure import (
    DurationExposureConfig, DurationExposureEncoder,
)
from src.v2.model.macro_state import MacroStateConfig, MacroStateEncoder
from src.v2.model.ow_epistar_v1 import OWEpiSTARV1, OWEpiSTARV1Config
from src.v2.model.rate_shock_memory import (
    RateShockMemoryBank, RateShockMemoryConfig,
)


def _init_sigmoid_gate_low(module: nn.Module, bias: float = -3.0) -> None:
    """Initialise the final Linear in a Sequential gate to a negative bias.

    Used for v2.3 conservative gate initialisation: starts the
    sigmoid output near 0.05 instead of 0.5 so DOW heads begin as
    small residual corrections to the strong OW backbone.
    """
    for layer in reversed(list(module)):
        if isinstance(layer, nn.Linear):
            nn.init.constant_(layer.bias, bias)
            break


@dataclass
class DOWEpiSTARConfig:
    """Hyperparameters for DOW-epiSTAR v2."""

    ow: OWEpiSTARV1Config = OWEpiSTARV1Config()
    duration: DurationExposureConfig = DurationExposureConfig()
    macro: MacroStateConfig = MacroStateConfig()
    rate_memory: RateShockMemoryConfig = RateShockMemoryConfig()
    macro_gate_input_dim: int = 9   # v2.3: 9 inputs (was 7 in v2.2)
    rate_gate_input_dim: int = 5    # 5 macro scalars; +2 retrieval scalars
                                    # appended internally
    rate_value_dim: int = 0         # set at construction
    rate_key_dim: int = 0           # set at construction
    head_hidden_dim: int = 64
    head_dropout: float = 0.1
    cross_attn_heads: int = 4
    use_macro_duration_head: bool = True
    use_rate_memory: bool = True
    use_duration_graph: bool = True
    # v2.3: conservative gate init bias.
    gate_init_bias: float = -3.0
    # Diagnostic flags. When False, force-zero the relevant component
    # so we can run the no-macro / no-duration / shuffled-macro
    # ablations from spec Section K without changing the trainer.
    disable_lambda_macro: bool = False
    disable_score_duration: bool = False
    disable_alpha_rate: bool = False
    disable_score_rate: bool = False


class DOWEpiSTAR(nn.Module):
    """OW-epiSTAR v1 + additive macro-duration residual head."""

    def __init__(
        self, cfg: DOWEpiSTARConfig, day_key_dim: int, ipo_key_dim: int,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.ow = OWEpiSTARV1(cfg.ow, day_key_dim=day_key_dim, ipo_key_dim=ipo_key_dim)
        self.duration_encoder = DurationExposureEncoder(cfg.duration)
        self.macro_encoder = MacroStateEncoder(cfg.macro)

        d = cfg.duration.out_dim
        m = cfg.macro.out_dim
        # Projection if d != m for the elementwise interaction.
        if d != m:
            self.duration_proj = nn.Linear(d, m)
            common = m
        else:
            self.duration_proj = nn.Identity()
            common = d
        self.duration_head = nn.Sequential(
            nn.LayerNorm(2 * common + common),
            nn.Linear(2 * common + common, cfg.head_hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.head_dropout),
            nn.Linear(cfg.head_hidden_dim, 1),
        )
        # Lambda gate over macro scalars (Section E + v2.3 expansion).
        self.lambda_gate = nn.Sequential(
            nn.Linear(cfg.macro_gate_input_dim, cfg.head_hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.head_hidden_dim, 1),
        )
        # v2.3 patch A: init lambda_gate bias to -3.0 (sigmoid ~= 0.05).
        _init_sigmoid_gate_low(self.lambda_gate, bias=cfg.gate_init_bias)

        # Section F: rate-shock memory + alpha_rate gate + score_rate.
        if cfg.use_rate_memory:
            assert cfg.rate_key_dim > 0 and cfg.rate_value_dim > 0
            self.rate_memory = RateShockMemoryBank(
                cfg.rate_memory, key_dim=cfg.rate_key_dim,
                value_dim=cfg.rate_value_dim,
            )
            d_hidden = cfg.ow.backbone.hidden_dim
            self.rate_value_proj = nn.Linear(cfg.rate_value_dim, d_hidden)
            self.rate_cross_attn = nn.MultiheadAttention(
                embed_dim=d_hidden, num_heads=cfg.cross_attn_heads,
                dropout=cfg.ow.backbone.dropout, batch_first=True,
            )
            self.rate_score_head = nn.Sequential(
                nn.LayerNorm(2 * d_hidden),
                nn.Linear(2 * d_hidden, cfg.head_hidden_dim),
                nn.GELU(),
                nn.Dropout(cfg.head_dropout),
                nn.Linear(cfg.head_hidden_dim, 1),
            )
            # Gate input is rate_gate_input_dim macro scalars from the
            # trainer plus 2 retrieval scalars (top1_sim, sim_entropy)
            # appended internally.
            self.rate_gate = nn.Sequential(
                nn.Linear(cfg.rate_gate_input_dim + 2, cfg.head_hidden_dim),
                nn.GELU(),
                nn.Linear(cfg.head_hidden_dim, 1),
            )
            # v2.3 patch A: init rate_gate bias to -3.0 (sigmoid ~= 0.05).
            _init_sigmoid_gate_low(self.rate_gate, bias=cfg.gate_init_bias)
        else:
            self.rate_memory = None

        # Section G fallback: graph source gate over (w_corr, w_duration).
        if cfg.use_duration_graph:
            self.graph_source_gate = GraphSourceGate(
                macro_gate_state_dim=cfg.macro.gate_state_dim,
                hidden_dim=cfg.head_hidden_dim,
            )
        else:
            self.graph_source_gate = None

    def compute_duration_exposure(self, duration_input: Tensor) -> Tensor:
        """Encode per-ticker duration exposure (used by trainer for the
        graph as well as the duration head)."""
        return self.duration_encoder(duration_input)

    def compute_macro(self, macro_input: Tensor) -> tuple[Tensor, Tensor]:
        """Encode the daily macro state, returning (state, gate_state)."""
        return self.macro_encoder(macro_input)

    def compute_graph_weights(self, macro_gate_state: Tensor) -> Tensor:
        """[2] softmax weights (w_corr, w_duration) for the graph gate."""
        if self.graph_source_gate is None:
            return torch.tensor(
                [1.0, 0.0], dtype=macro_gate_state.dtype,
                device=macro_gate_state.device,
            )
        return self.graph_source_gate(macro_gate_state)

    def forward_day(
        self,
        # OW v1 args
        patches: Tensor, patch_mask: Tensor, active_mask: Tensor,
        day_query_key: Tensor, ipo_query_keys: Tensor,
        ipo_gate_features: Tensor, query_day_idx: int,
        allowed_day_indices: Tensor, gate_regime_scalars: Tensor,
        # DOW v2 new args
        duration_input: Tensor,
        macro_input: Tensor,
        macro_gate_input: Tensor,
        # Section F new args (optional)
        rate_query_key: Tensor | None = None,
        rate_gate_input: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Forward pass for one trading day with score decomposition.

        Args:
            patches..gate_regime_scalars: passed through to OW v1.
            duration_input: [num_active, duration_input_dim].
            macro_input: [macro_input_dim] daily macro features.
            macro_gate_input: [macro_gate_input_dim] daily gate scalars.
            rate_query_key: [rate_key_dim] daily rate-shock key
                (Section F). Required when ``cfg.use_rate_memory``.
            rate_gate_input: [rate_gate_input_dim] daily rate gate
                scalars (top1 sim and entropy are appended internally).
        """
        cfg = self.cfg
        ow_out = self.ow.forward_day(
            patches=patches, patch_mask=patch_mask, active_mask=active_mask,
            day_query_key=day_query_key, ipo_query_keys=ipo_query_keys,
            ipo_gate_features=ipo_gate_features,
            query_day_idx=query_day_idx,
            allowed_day_indices=allowed_day_indices,
            gate_regime_scalars=gate_regime_scalars,
        )
        score_idio = ow_out["y_hat"]
        z_final = ow_out["z_final"]                     # [N, hidden_dim]
        device = score_idio.device

        active_idx = active_mask.nonzero(as_tuple=False).squeeze(-1)
        a = active_idx.shape[0]

        # Section E: macro-duration head.
        if cfg.use_macro_duration_head and a > 0:
            d_exp = self.duration_encoder(duration_input)
            d_exp_proj = self.duration_proj(d_exp)

            m_state, m_gate_state = self.macro_encoder(macro_input)
            m_state_b = m_state.unsqueeze(0).expand(a, -1)
            interaction = torch.cat(
                [d_exp_proj, m_state_b, d_exp_proj * m_state_b], dim=-1
            )
            score_dur_active = self.duration_head(interaction).squeeze(-1)

            lambda_logit = self.lambda_gate(macro_gate_input).squeeze()
            lambda_macro = torch.sigmoid(lambda_logit)
            if cfg.disable_lambda_macro:
                lambda_macro = torch.zeros((), device=device)
            if cfg.disable_score_duration:
                score_dur_active = torch.zeros_like(score_dur_active)

            score_duration_full = torch.zeros_like(score_idio, dtype=score_dur_active.dtype)
            score_duration_full[active_idx] = score_dur_active
        else:
            score_duration_full = torch.zeros_like(score_idio)
            lambda_macro = torch.zeros((), device=device)

        # Section F: rate-shock memory + score_rate residual.
        score_rate_full = torch.zeros_like(score_idio)
        alpha_rate = torch.zeros((), device=device)
        rate_top1_sim = torch.zeros((), device=device)
        rate_entropy = torch.zeros((), device=device)
        if (cfg.use_rate_memory and self.rate_memory is not None
                and rate_query_key is not None and a > 0):
            rate_ret = self.rate_memory.retrieve(
                query_raw_key=rate_query_key, query_day_idx=query_day_idx,
            )
            rate_top1_sim = rate_ret["top1_sim"].detach()
            rate_entropy = rate_ret["sim_entropy"].detach()
            rate_proj = self.rate_value_proj(rate_ret["values"]).unsqueeze(0)  # [1, M, D]
            z_active = z_final[active_idx].unsqueeze(0)                          # [1, A, D]
            delta_rate, _ = self.rate_cross_attn(
                query=z_active, key=rate_proj, value=rate_proj,
            )
            delta_rate = delta_rate.squeeze(0)                                   # [A, D]
            score_rate_active = self.rate_score_head(
                torch.cat([z_active.squeeze(0), delta_rate], dim=-1)
            ).squeeze(-1)
            if rate_gate_input is None:
                rate_gate_in = torch.zeros(cfg.rate_gate_input_dim, device=device)
            else:
                rate_gate_in = rate_gate_input
            # Spec gate inputs are: top1_sim, sim_entropy + 5 macro
            # scalars (the trainer must pass the 5 scalars via
            # rate_gate_input; we append top1_sim and entropy here).
            rate_gate_full = torch.cat([
                rate_ret["top1_sim"].unsqueeze(0),
                rate_ret["sim_entropy"].unsqueeze(0),
                rate_gate_in,
            ])
            alpha_logit = self.rate_gate(rate_gate_full).squeeze()
            alpha_rate = torch.sigmoid(alpha_logit)
            if cfg.disable_alpha_rate:
                alpha_rate = torch.zeros((), device=device)
            if cfg.disable_score_rate:
                score_rate_active = torch.zeros_like(score_rate_active)
            score_rate_full = torch.zeros_like(score_idio, dtype=score_rate_active.dtype)
            score_rate_full[active_idx] = score_rate_active

        # Combine.
        score_total = score_idio + (
            lambda_macro.to(score_idio.dtype) * score_duration_full.to(score_idio.dtype)
            + alpha_rate.to(score_idio.dtype) * score_rate_full.to(score_idio.dtype)
        )
        score_total = score_total * active_mask.to(score_idio.dtype)

        # Update outputs.
        ow_out["score_idio"] = score_idio.detach().clone()
        ow_out["score_duration"] = score_duration_full.detach().clone()
        ow_out["score_rate"] = score_rate_full.detach().clone()
        ow_out["alpha_rate"] = alpha_rate.detach().clone() if alpha_rate.dim() == 0 else alpha_rate.detach()
        ow_out["rate_top1_sim"] = rate_top1_sim
        ow_out["rate_sim_entropy"] = rate_entropy
        ow_out["lambda_macro"] = lambda_macro.detach().clone() if lambda_macro.dim() == 0 else lambda_macro.detach()
        ow_out["y_hat"] = score_total
        return ow_out


__all__ = ["DOWEpiSTAR", "DOWEpiSTARConfig"]
