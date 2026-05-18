"""InVAR: INverted VARiate Attention for Spatio-Temporal Equity Ranking.

Two-axis inverted Transformer for cross-sectional equity ranking.

Components:

  - VariateTokenizer  : (N, L, F) -> (N, d). Per-feature MLP across the
                        time dimension, then a linear projection to
                        ``d_model``.
  - MacroEncoder      : (L, F_macro) -> (d). MLP over the flattened macro
                        lookback producing a per-day regime query q_t.
  - RegimeAxis        : abstract base producing K regime tokens (K, d)
                        given the regime query and a memory bank.
                        Three variants: Calendar (Design A), Kmeans
                        (Design B), Retrieval (Design C, headline).
  - InvarBlock x 4    : variate self-attention plus variate-to-regime
                        cross-attention combined in parallel via a
                        per-ticker scalar gate, followed by a
                        position-wise FFN.
  - RankingHead       : linear (d) -> 1 per ticker.
  - RegimeClassifierHead : linear (d) -> K_offline=8 over q_t (auxiliary
                        supervision against offline GaussianMixture
                        labels).
  - VolHead           : per-ticker linear (d) -> 1 (auxiliary supervision
                        against 20-day forward realised vol).

Hyperparameters (locked at this commit):

  - d_model = 128, n_heads = 4, FFN dim = 256, dropout = 0.1
  - 4 InvarBlocks
  - L = 60, F = 26, F_macro = 24
  - Memory bank size 1024 (Retrieval); K_retrieve = 32 at inference,
    annealed from full bank in Phase 2 training.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import Tensor, nn


@dataclass
class InvarConfig:
    """Hyperparameters for the INVAR model."""

    n_features: int = 26
    lookback: int = 60
    macro_dim: int = 24
    d_model: int = 128
    n_heads: int = 4
    ffn_dim: int = 256
    n_layers: int = 4
    dropout: float = 0.1
    n_offline_regimes: int = 8
    regime_axis: str = "retrieval"        # one of {calendar, kmeans, retrieval}
    bank_size: int = 64                   # v3 default: 64 (was 1024 in v1/v2)
    top_k_retrieve: int = 32              # tokens fed to InvarBlock at inference
    n_calendar_windows: int = 64
    n_kmeans_clusters: int = 12
    tokenizer_hidden: int = 64
    macro_hidden: int = 64
    head_hidden: int = 64
    use_scalar_gate: bool = True          # v3: shared scalar gate (was per-ticker MLP)
    zero_init_cross_attn: bool = True     # v3: cross_attn out_proj starts at zero
    # InVAR v4: MASTER-style market-guided softmax gate at the raw-feature input.
    use_market_gate: bool = False
    gate_beta_init: float = 2.0
    gate_learn_beta: bool = True
    gate_hidden_dim: int = 0
    gate_dropout: float = 0.0
    # Phase 4 ablation matrix:
    gate_location: str = "input"          # "input" (raw F) or "post_tokenizer" (latent d)
    gate_form: str = "softmax_F"          # "softmax_F" or "sigmoid"
    disable_bank: bool = False            # cell F: disable retrieval bank entirely
    # Differentiable retrieval ablation:
    #   "hard_topk"        - default; non-differentiable top-K selection (current v6)
    #   "softmax_full"     - softmax over all K=bank_size; weighted bank values
    #                         returned (WAVE v3 style); gradients flow into all
    #                         keys + values + macro encoder
    #   "softmax_topk"     - softmax over all K; top top_k_retrieve by weight
    #                         returned with weight-multiplied values; gradients
    #                         flow into all keys via softmax denominator
    #   "gumbel_topk"      - Gumbel-noise top-K with straight-through estimator
    retrieval_mode: str = "hard_topk"
    gumbel_tau: float = 1.0               # Gumbel softmax temperature
    # InVAR v6 extensions (additive; v4 default behaviour preserved when
    # all v6 flags are False).
    use_market_gate_v2: bool = False
    macro_encoder_mode: str = "mlp_flat"      # last/mlp_flat/temporal_attn/gru
    macro_state_dim: int = 64
    market_gate_v2_form: str = "softmax_F"    # softmax_F/sigmoid_centered/sigmoid_residual
    market_gate_v2_hidden_dim: int = 64
    use_dynamic_bank_controller: bool = False
    bank_controller_mode: str = "hybrid"      # deterministic/learned/hybrid
    bank_controller_min_weight: float = 0.05
    bank_controller_max_weight: float = 1.00
    bank_controller_hidden_dim: int = 64


class VariateTokenizer(nn.Module):
    """(N, L, F) -> (N, d). Per-feature MLP then linear projection.

    Each of the F features is treated as a "variate" (iTransformer-style):
    its L-step time series passes through a shared per-variate MLP, the F
    outputs are flattened, then a linear maps to ``d_model``.
    """

    def __init__(self, cfg: InvarConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.per_var_mlp = nn.Sequential(
            nn.Linear(cfg.lookback, cfg.tokenizer_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.tokenizer_hidden, cfg.tokenizer_hidden),
        )
        self.proj = nn.Linear(cfg.n_features * cfg.tokenizer_hidden, cfg.d_model)

    def forward(self, x: Tensor) -> Tensor:
        """Args: x ``(N, L, F)``. Returns ``(N, d)``."""
        x = x.transpose(1, 2)                       # (N, F, L)
        h = self.per_var_mlp(x)                     # (N, F, hidden)
        h = h.flatten(1)                            # (N, F * hidden)
        return self.proj(h)


class MacroEncoder(nn.Module):
    """(L, F_macro) -> (d). MLP over the flattened macro lookback."""

    def __init__(self, cfg: InvarConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.flat_dim = cfg.lookback * cfg.macro_dim
        self.mlp = nn.Sequential(
            nn.Linear(self.flat_dim, cfg.macro_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.macro_hidden, cfg.d_model),
        )

    def forward(self, m: Tensor) -> Tensor:
        """Args: m ``(L, F_macro)``. Returns ``(d,)``."""
        return self.mlp(m.flatten())


class RegimeAxisCalendar(nn.Module):
    """Non-overlapping 60-day train-history windows projected to (K, d).

    Memory is rebuilt from train-fold history only (set by the trainer
    via ``populate_calendar`` before training). At forward time the
    memory tokens are returned as-is (calendar tokens are static).
    """

    def __init__(self, cfg: InvarConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.proj = nn.Linear(cfg.macro_dim * 2 + 4, cfg.d_model)
        self.register_buffer(
            "memory_raw", torch.zeros(cfg.n_calendar_windows,
                                       cfg.macro_dim * 2 + 4),
            persistent=True,
        )
        self._populated = False

    def populate(self, raw: Tensor) -> None:
        """Set the calendar memory ``(K, macro_dim*2 + 4)``."""
        with torch.no_grad():
            self.memory_raw = raw.detach()
        self._populated = True

    def forward(self, q_t: Tensor) -> Tensor:
        return self.proj(self.memory_raw)


class RegimeAxisKmeans(nn.Module):
    """K=12 cluster centroids from train-fold macro vectors."""

    def __init__(self, cfg: InvarConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.proj = nn.Linear(cfg.macro_dim, cfg.d_model)
        self.register_buffer(
            "centroids", torch.zeros(cfg.n_kmeans_clusters, cfg.macro_dim),
            persistent=True,
        )
        self._populated = False

    def populate(self, centroids: Tensor) -> None:
        with torch.no_grad():
            self.centroids = centroids.detach()
        self._populated = True

    def forward(self, q_t: Tensor) -> Tensor:
        return self.proj(self.centroids)


class RegimeAxisRetrieval(nn.Module):
    """Learned key-value memory bank with top-K retrieval (Design C).

    Bank is parameterised: ``keys`` and ``values`` are nn.Parameters.
    At forward time we score the regime query against all keys, gather
    the top-K values, and return them as the regime tokens fed to the
    InvarBlocks.

    The trainer can call ``set_top_k`` for the K curriculum (K full
    during epoch 1, K=32 thereafter) and ``freeze_values`` for the
    epoch-1 stop-gradient on values stability mitigation.
    """

    def __init__(self, cfg: InvarConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.keys = nn.Parameter(torch.randn(cfg.bank_size, cfg.d_model) * 0.02)
        self.values = nn.Parameter(torch.randn(cfg.bank_size, cfg.d_model) * 0.02)
        self._top_k = cfg.top_k_retrieve
        self._values_frozen = False
        self.last_top_idx: Optional[Tensor] = None
        self.last_top_scores: Optional[Tensor] = None

    def set_top_k(self, k: int) -> None:
        self._top_k = max(1, min(k, self.cfg.bank_size))

    def freeze_values(self, frozen: bool = True) -> None:
        self._values_frozen = frozen

    def forward(self, q_t: Tensor) -> Tensor:
        """Args: q_t ``(d,)``. Returns ``(K, d)``.

        Output shape depends on ``cfg.retrieval_mode``:
            - ``hard_topk``      : ``(top_k_retrieve, d)``
            - ``softmax_topk``   : ``(top_k_retrieve, d)`` weighted by softmax
            - ``softmax_full``   : ``(bank_size, d)`` weighted by softmax
            - ``gumbel_topk``    : ``(top_k_retrieve, d)`` via Gumbel-noise + ST
        """
        scores = self.keys @ q_t                                # (bank_size,)
        k = min(self._top_k, self.cfg.bank_size)
        mode = getattr(self.cfg, "retrieval_mode", "hard_topk")

        values_all = self.values
        if self._values_frozen:
            values_all = values_all.detach()

        if mode == "hard_topk":
            top = torch.topk(scores, k=k, dim=-1)
            top_idx = top.indices
            self.last_top_idx = top_idx
            self.last_top_scores = top.values
            return values_all[top_idx]                           # (K, d)

        if mode == "softmax_full":
            weights = torch.softmax(scores, dim=-1)              # (bank_size,)
            self.last_top_idx = None
            self.last_top_scores = weights
            return values_all * weights.unsqueeze(-1)            # (bank_size, d)

        if mode == "softmax_topk":
            # Softmax over all keys, then keep the top-K by weight. The
            # softmax denominator includes every key, so gradients flow
            # into every key; only the top-K values get gradient via
            # the weight multiplication.
            weights = torch.softmax(scores, dim=-1)              # (bank_size,)
            top_w = torch.topk(weights, k=k, dim=-1)
            top_idx = top_w.indices
            top_weights = top_w.values
            self.last_top_idx = top_idx
            self.last_top_scores = top_weights
            return values_all[top_idx] * top_weights.unsqueeze(-1)  # (K, d)

        if mode == "gumbel_topk":
            # Gumbel(0,1) noise added to scores for stochastic top-K
            # selection; straight-through estimator: forward is the
            # hard-selected top-K, backward uses the softmax-noise
            # weights so gradients flow into all keys.
            tau = float(getattr(self.cfg, "gumbel_tau", 1.0))
            if self.training:
                gumbel = -torch.log(-torch.log(
                    torch.rand_like(scores).clamp(min=1.0e-9, max=1.0 - 1.0e-9),
                ))
                noisy = (scores + gumbel) / max(tau, 1.0e-3)
            else:
                noisy = scores / max(tau, 1.0e-3)
            soft_w = torch.softmax(noisy, dim=-1)                # (bank_size,)
            top = torch.topk(soft_w, k=k, dim=-1)
            top_idx = top.indices
            top_soft = top.values
            self.last_top_idx = top_idx
            self.last_top_scores = top_soft
            # Straight-through: forward uses the soft weights at the
            # selected positions; backward inherits soft_w gradient.
            return values_all[top_idx] * top_soft.unsqueeze(-1)  # (K, d)

        raise ValueError(f"unknown retrieval_mode: {mode!r}")


def _build_regime_axis(cfg: InvarConfig) -> nn.Module:
    if cfg.regime_axis == "calendar":
        return RegimeAxisCalendar(cfg)
    if cfg.regime_axis == "kmeans":
        return RegimeAxisKmeans(cfg)
    if cfg.regime_axis == "retrieval":
        return RegimeAxisRetrieval(cfg)
    raise ValueError(f"unknown regime_axis: {cfg.regime_axis}")


class InvarBlock(nn.Module):
    """Variate self-attn || regime cross-attn || gate || FFN.

    The two attention paths run in parallel on the same input ``v``;
    their outputs are combined via a per-ticker scalar gate g(m_t, v_i)
    in [0, 1]. The gated sum is added back to ``v`` (residual), then
    a position-wise FFN finishes the block.
    """

    def __init__(self, cfg: InvarConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.norm_regime = nn.LayerNorm(cfg.d_model)
        self.self_attn = nn.MultiheadAttention(
            cfg.d_model, cfg.n_heads, dropout=cfg.dropout, batch_first=False,
        )
        self.cross_attn = nn.MultiheadAttention(
            cfg.d_model, cfg.n_heads, dropout=cfg.dropout, batch_first=False,
        )
        if cfg.zero_init_cross_attn:
            with torch.no_grad():
                self.cross_attn.out_proj.weight.zero_()
                self.cross_attn.out_proj.bias.zero_()
        if cfg.use_scalar_gate:
            self.gate_mlp = nn.Linear(cfg.d_model, 1)
        else:
            self.gate_mlp = nn.Sequential(
                nn.Linear(cfg.d_model + cfg.d_model, cfg.d_model // 2),
                nn.GELU(),
                nn.Linear(cfg.d_model // 2, 1),
            )
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.ffn_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.ffn_dim, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(
        self, v: Tensor, regime_tokens: Tensor, q_t: Tensor,
        return_attn: bool = False, disable_bank: bool = False,
        bank_weight: Tensor | None = None,
    ) -> tuple[Tensor, Optional[dict]]:
        """Args:
            v             : ``(N, d)``
            regime_tokens : ``(K, d)``
            q_t           : ``(d,)`` (or broadcast to (N, d) inside the gate)
            disable_bank  : Phase 4 cell F: if True, skip the regime
                            cross-attention path entirely and use only
                            variate self-attention.

        Returns ``(out, attn_dict)`` where attn_dict is None unless
        ``return_attn``.
        """
        N, D = v.shape

        v_norm = self.norm1(v).unsqueeze(1)                       # (N, 1, d)
        sa_out, sa_w = self.self_attn(
            v_norm, v_norm, v_norm, need_weights=return_attn,
        )
        sa_out = sa_out.squeeze(1)                                 # (N, d)

        if disable_bank:
            # Cell F: no cross-attention path; gate becomes irrelevant.
            v = v + sa_out
            v = v + self.ffn(self.norm2(v))
            return v, None

        rk = self.norm_regime(regime_tokens).unsqueeze(1)          # (K, 1, d)
        ca_out, ca_w = self.cross_attn(
            v_norm, rk, rk, need_weights=return_attn,
        )
        ca_out = ca_out.squeeze(1)                                 # (N, d)

        if bank_weight is not None:
            # InVAR v6: per-day scalar weight on the bank contribution.
            ca_out = ca_out * bank_weight.view(1, 1)

        if self.cfg.use_scalar_gate:
            # Shared scalar per day (broadcast across N tickers).
            g_scalar = torch.sigmoid(self.gate_mlp(q_t))             # ()
            g = g_scalar.expand(N, 1)
        else:
            gate_in = torch.cat([v, q_t.expand(N, D)], dim=-1)
            g = torch.sigmoid(self.gate_mlp(gate_in))
        combined = g * sa_out + (1.0 - g) * ca_out                    # (N, d)

        v = v + combined
        v = v + self.ffn(self.norm2(v))

        attn_dict = None
        if return_attn:
            attn_dict = {"sa": sa_w, "ca": ca_w, "gate": g.detach()}
        return v, attn_dict


class Invar(nn.Module):
    """End-to-end INVAR forward pass.

    forward(features, macro, mask) -> dict:
      - y_hat : (N,) ranking score
      - regime_logits : (K_offline,)
      - vol_hat : (N,) realised-vol prediction
      - attn_weights : optional dict of per-block attention weights
    """

    def __init__(self, cfg: InvarConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or InvarConfig()
        self.tokenizer = VariateTokenizer(self.cfg)
        self.macro_encoder = MacroEncoder(self.cfg)
        self.regime_axis = _build_regime_axis(self.cfg)
        if self.cfg.use_market_gate:
            from src.invar.modules.market_gate import MarketGate
            # Output dimension depends on gate location:
            #   "input"           -> F (raw panel features)
            #   "post_tokenizer"  -> d_model (latent ticker tokens)
            gate_out_dim = (self.cfg.n_features
                              if self.cfg.gate_location == "input"
                              else self.cfg.d_model)
            self.market_gate = MarketGate(
                num_features=gate_out_dim,
                market_dim=self.cfg.macro_dim,
                beta_init=self.cfg.gate_beta_init,
                learn_beta=self.cfg.gate_learn_beta,
                hidden_dim=self.cfg.gate_hidden_dim,
                dropout=self.cfg.gate_dropout,
                gate_form=self.cfg.gate_form,
            )
        else:
            self.market_gate = None
        self._last_alpha: Tensor | None = None

        # InVAR v6 modules (additive). Selected via cfg flags.
        if self.cfg.use_market_gate_v2:
            from src.invar.modules.macro_window_encoder import MacroWindowEncoder
            from src.invar.modules.market_gate_v2 import MarketGateV2
            self.macro_window_encoder = MacroWindowEncoder(
                macro_dim=self.cfg.macro_dim,
                lookback=self.cfg.lookback,
                out_dim=self.cfg.macro_state_dim,
                hidden_dim=self.cfg.macro_state_dim * 2,
                dropout=self.cfg.dropout,
                mode=self.cfg.macro_encoder_mode,
            )
            self.market_gate_v2 = MarketGateV2(
                num_features=self.cfg.n_features,
                macro_state_dim=self.cfg.macro_state_dim,
                gate_form=self.cfg.market_gate_v2_form,
                hidden_dim=self.cfg.market_gate_v2_hidden_dim,
                beta_init=self.cfg.gate_beta_init,
                learn_beta=self.cfg.gate_learn_beta,
                dropout=self.cfg.dropout,
                identity_init=True,
            )
        else:
            self.macro_window_encoder = None
            self.market_gate_v2 = None

        if self.cfg.use_dynamic_bank_controller:
            from src.invar.modules.dynamic_bank_controller import DynamicBankController
            self.bank_controller = DynamicBankController(
                macro_state_dim=self.cfg.macro_state_dim,
                stats_dim=6,
                hidden_dim=self.cfg.bank_controller_hidden_dim,
                init_bias=0.0,
                min_weight=self.cfg.bank_controller_min_weight,
                max_weight=self.cfg.bank_controller_max_weight,
                mode=self.cfg.bank_controller_mode,
            )
            self._zscore_buf = nn.ParameterDict()
        else:
            self.bank_controller = None
        self._last_bank_weight: Tensor | None = None
        self._last_bank_debug: dict | None = None

        self.blocks = nn.ModuleList(
            [InvarBlock(self.cfg) for _ in range(self.cfg.n_layers)]
        )
        self.norm_out = nn.LayerNorm(self.cfg.d_model)
        self.ranking_head = nn.Sequential(
            nn.Linear(self.cfg.d_model, self.cfg.head_hidden),
            nn.GELU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(self.cfg.head_hidden, 1),
        )
        self.regime_classifier = nn.Linear(
            self.cfg.d_model, self.cfg.n_offline_regimes,
        )
        self.vol_head = nn.Sequential(
            nn.Linear(self.cfg.d_model, self.cfg.head_hidden),
            nn.GELU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(self.cfg.head_hidden, 1),
        )

    def get_alpha(self) -> Tensor | None:
        return self._last_alpha

    def get_bank_weight(self) -> Tensor | None:
        return self._last_bank_weight

    def get_bank_debug(self) -> dict | None:
        return self._last_bank_debug

    def forward(
        self, features: Tensor, macro: Tensor, mask: Tensor,
        return_attn: bool = False,
    ) -> dict[str, Tensor]:
        """Args:
            features : ``(N, L, F)``
            macro    : ``(L, F_macro)``
            mask     : ``(N,)`` bool
        """
        # InVAR v6 path: precompute the macro_state vector via the
        # window encoder; if v6 MarketGateV2 is enabled, it gates the
        # raw input features. v4 MarketGate path remains the default.
        macro_state: Tensor | None = None
        if self.macro_window_encoder is not None:
            macro_state = self.macro_window_encoder(macro)        # (1, D_macro)
        if self.market_gate_v2 is not None and macro_state is not None:
            features_b = features.unsqueeze(0)
            features_b, alpha = self.market_gate_v2(features_b, macro_state)
            features = features_b.squeeze(0)
            self._last_alpha = alpha.detach()
        elif self.market_gate is not None and self.cfg.gate_location == "input":
            # InVAR v4: market-guided gate. Identity at init (zero proj
            # weights yield alpha = ones for both gate forms), so the
            # rest of the backbone behaves like v3 until the gate
            # accumulates gradient signal.
            features_b = features.unsqueeze(0)
            macro_b = macro.unsqueeze(0)
            features_b, alpha = self.market_gate(features_b, macro_b)
            features = features_b.squeeze(0)
            self._last_alpha = alpha.detach()
        else:
            self._last_alpha = None

        v = self.tokenizer(features)                                # (N, d)

        if (self.market_gate is not None
                and self.cfg.gate_location == "post_tokenizer"
                and self.market_gate_v2 is None):
            v_b = v.unsqueeze(0)
            macro_b = macro.unsqueeze(0)
            v_b, alpha = self.market_gate(v_b, macro_b)
            v = v_b.squeeze(0)
            self._last_alpha = alpha.detach()

        q_t = self.macro_encoder(macro)                              # (d,)
        regime_tokens = self.regime_axis(q_t)                        # (K, d)

        # InVAR v6: dynamic bank weight, computed from retrieval stats
        # and macro_state (with stress_features = last-step macro slice
        # as a leakage-safe proxy in this MVP).
        bank_weight: Tensor | None = None
        if self.bank_controller is not None and not self.cfg.disable_bank:
            scores = self.regime_axis.last_top_scores
            if scores is not None:
                # Build bank_stats. We use simple normalisations rather
                # than train-fold z-scores to avoid coupling with the
                # dataset; the controller's MLP can compensate. This is
                # MVP; a full train-fold z-score pipeline is a follow-up.
                rd = -scores.mean().reshape(1)                         # higher = more distant
                p = torch.softmax(scores, dim=-1)
                re = (-(p * (p + 1.0e-9).log()).sum()).reshape(1)      # entropy
                bank_vals_norm = self.regime_axis.values.detach().norm(dim=-1).mean().reshape(1)
                ac = mask.float().sum().reshape(1)
                # MVP: scale by approximate constants tuned for the K=32
                # bank. A proper train-fold z-score pipeline (Section 6.4
                # of the spec) is a follow-up upgrade.
                rd_z = rd / (rd.abs().detach() + 1.0)
                re_z = (re - 3.0) / 1.0                                # log K=32 ~3.5
                bn_z = (bank_vals_norm - 0.5) / 0.5
                ac_z = (ac - 500.0) / 200.0
                bank_stats = {
                    "retrieval_distance_z": rd_z,
                    "retrieval_entropy_z": re_z,
                    "bank_value_norm_z": bn_z,
                    "active_count_z": ac_z,
                }
                # stress_features: 6 dims from the last macro step (proxy).
                stress = macro[-1, : 6] if macro.shape[-1] >= 6 else torch.zeros(6, device=macro.device)
                stress = stress.reshape(1, 6)
                bw_in_state = (
                    macro_state if macro_state is not None
                    else torch.zeros(1, self.cfg.macro_state_dim, device=v.device)
                )
                bank_weight, debug = self.bank_controller(
                    bw_in_state, bank_stats, stress,
                )
                self._last_bank_weight = bank_weight.detach()
                self._last_bank_debug = {k: w.detach() for k, w in debug.items()}

        attn_per_block = []
        for block in self.blocks:
            v, attn = block(
                v, regime_tokens, q_t, return_attn=return_attn,
                disable_bank=self.cfg.disable_bank,
                bank_weight=bank_weight,
            )
            if return_attn:
                attn_per_block.append(attn)

        v_out = self.norm_out(v)                                    # (N, d)
        y_hat = self.ranking_head(v_out).squeeze(-1)                # (N,)
        regime_logits = self.regime_classifier(q_t)                 # (K_offline,)
        vol_hat = self.vol_head(v_out).squeeze(-1)                  # (N,)

        m = mask.float()
        y_hat = y_hat * m
        vol_hat = vol_hat * m

        out = {
            "y_hat": y_hat,
            "regime_logits": regime_logits,
            "vol_hat": vol_hat,
        }
        if return_attn:
            out["attn_weights"] = attn_per_block
        return out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


__all__ = [
    "InvarConfig", "Invar", "VariateTokenizer", "MacroEncoder",
    "RegimeAxisCalendar", "RegimeAxisKmeans", "RegimeAxisRetrieval",
    "InvarBlock", "count_parameters",
]
