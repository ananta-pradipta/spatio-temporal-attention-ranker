"""InVAR-STX: strictly NON-GRAPH spatio-temporal InVAR (EXPERIMENT-ONLY).

NOT a paper baseline. This trainer is a near-clone of
``src.baselines.train_swa_invar_dgm_v2`` and keeps the entire v2 harness,
the RAG-STAR day-memory block, the RAG-STAR macro rate-sensitivity head,
the FIXED SWA EMA loop, and the JSON-only disk-safe write BYTE-IDENTICAL
to that file. The ONLY thing that changes is the backbone.

Motivation / fix vs SWA-InVAR-DGM
---------------------------------
SWA-InVAR-DGM's ``_backbone_hidden`` flattens each ticker's
``(T, F)`` lookback into ``T * F`` and feeds it through the vendored
iTransformer's single ``Linear(T*F, d_model)`` token embedding. That
collapses all temporal structure into one linear projection. InVAR-STX
replaces that flat embedding with a proper PER-TICKER TEMPORAL ENCODER:

  (a) PER-TICKER TEMPORAL ENCODER (the temporal aspect)
      For each active ticker, the ``(T, F)`` window is encoded by a
      small in-file temporal Transformer encoder: ``Linear(F -> d)``
      input projection + learned positional embedding over the ``T``
      time steps + 2 ``nn.TransformerEncoderLayer`` (n_heads=4,
      d_ff=256, dropout=0.1, GELU, batch_first), then last-step pooling.
      Shapes: ``(N_active, T, F) -> (N_active, d_model)``.

  (b) CROSS-TICKER SPATIAL ATTENTION (the spatial aspect)
      The ``(N_active, d_model)`` per-ticker tokens are fed as
      ``(1, N_active, d_model)`` through a fresh in-file instance of the
      VENDORED iTransformer ``Encoder`` (EncoderLayer / AttentionLayer /
      FullAttention, e_layers=2). Because the token axis is the ticker
      axis, the dense self-attention runs ACROSS tickers. Output
      ``(N_active, d_model)``. This is iTransformer-style cross-variate
      attention, but fed temporal-embedding tokens instead of the
      flat-Linear embedding. NO graph: no adjacency, no A_corr, no
      A_dur, no attention-bias-from-graph; it is dense attention.

  (c) MACRO-CONDITIONED SPATIAL ATTENTION (the genuinely new piece)
      Before the cross-ticker Encoder, every per-ticker token is
      FiLM-modulated by the daily macro-rate state:
        token_t = gamma(m_state) * token + beta(m_state)
      where ``m_state`` is produced by the SAME ``MacroStateEncoder``
      the macro head already uses (no new macro encoder, no new macro
      input). ``gamma`` is initialised to 1 and ``beta`` to 0 (final
      Linear weight+bias zeroed, gamma gets a +1 bias), and a learned
      conservative scalar gate (sigmoid, bias init -3) blends the FiLM
      delta in, so at init InVAR-STX behaves as a plain
      temporal-encoder + cross-ticker iTransformer with no macro
      modulation, and only deviates as it learns. This is
      macro-conditioned DENSE attention, NOT a graph.

Everything downstream of the backbone hiddens (Gumbel-topk retrieval
fusion, RAG-STAR day-memory fusion, RAG-STAR macro rate-sensitivity
head, final score = idio + lambda_macro * s_dur) is reused VERBATIM
from ``train_swa_invar_dgm_v2``. The ``src.baselines.v2_runner`` data /
fold / eval calls (build_panel, build_masks, fold_split,
standardize_features, build_age_features, cs_mse_loss,
evaluate_predictions, set_seeds, warmup_cosine_lr) are invoked with the
SAME arguments in the SAME order. The FIXED SWA EMA loop (skipping
non-float buffers via ``torch.is_floating_point``) is kept verbatim.

Disk: experiment-only, home storage nearly full. Writes ONLY the
fold{F}_seed{S}.json (history entries contain "epoch" so skip-if-done
works). NO predictions npz.

Run:
    python -m src.baselines.train_invar_stx_v2 --fold 1 --seed 42 \
        --panel_kind lattice_native --two_regime_val \
        --output_dir results/invar_stx
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn

from src.baselines.itransformer_adapter import (
    ITransformerHyperparams,
)
from src.baselines.v2_runner import (
    V2BaselineConfig,
    build_age_features,
    build_masks,
    build_panel,
    cs_mse_loss,
    evaluate_predictions,
    fold_split,
    set_seeds,
    standardize_features,
    warmup_cosine_lr,
)
from src.baselines.vendored.itransformer import (
    AttentionLayer,
    Encoder,
    EncoderLayer,
    FullAttention,
)
from src.v2.data.episode_keys import (
    EPISODE_KEY_COLS,
    EpisodeKeyConfig,
    build_episode_keys,
)
from src.v2.data.macro_duration_features import (
    MACRO_GATE_COLS,
    build_macro_duration_features,
    standardize_macro_duration,
)
from src.v2.data.rolling_macro_betas import (
    ROLLING_BETA_COLS,
    betas_to_tensor,
    build_rolling_betas,
)
from src.v2.model.duration_exposure import (
    DurationExposureConfig,
    DurationExposureEncoder,
)
from src.v2.model.episode_memory import EpisodeMemoryBank, EpisodeMemoryConfig
from src.v2.model.macro_state import MacroStateConfig, MacroStateEncoder
from src.v2.training.train_dow_epistar import (
    resolve_duration_indices,
    _gather_or_zero,
)


@dataclass
class InvarSTXV2Config(V2BaselineConfig):
    """v2 protocol + SWA-InVAR knobs + RAG-STAR day-memory / macro-head
    knobs + per-ticker temporal-encoder knobs.

    Every field present in ``SWAInvarDGMV2Config`` is copied verbatim so
    the downstream blocks (retrieval bank, day-memory, macro head, SWA)
    are byte-identical. The only new fields are the temporal-encoder
    layer count / FiLM-gate init bias for the new backbone.
    """

    output_dir: str = "results/invar_stx"
    # iTransformer backbone knobs (verbatim from SWAInvarDGMV2Config).
    d_model: int = 128
    n_heads: int = 4
    d_ff: int = 256
    e_layers: int = 2
    dropout: float = 0.1
    activation: str = "gelu"
    use_norm: bool = False
    # InVAR differentiable regime-retrieval bank (verbatim).
    bank_size: int = 64
    top_k_retrieve: int = 32
    retrieval_mode: str = "gumbel_topk"
    gumbel_tau: float = 1.0
    # Stochastic Weight Averaging (verbatim).
    use_swa: bool = True
    swa_decay: float = 0.999
    swa_warmup_epochs: int = 5
    # RAG-STAR day-memory (verbatim).
    day_value_dim: int = 32
    gate_hidden_dim: int = 64
    cross_attn_heads: int = 4
    # RAG-STAR macro rate-sensitivity head (verbatim).
    duration_hidden_dim: int = 64
    duration_out_dim: int = 32
    macro_hidden_dim: int = 64
    macro_out_dim: int = 32
    macro_gate_state_dim: int = 16
    head_hidden_dim: int = 64
    head_dropout: float = 0.1
    gate_init_bias: float = -3.0
    # NEW: per-ticker temporal encoder + macro-conditioned FiLM knobs.
    temporal_e_layers: int = 2
    film_gate_init_bias: float = -3.0
    # ---- ABLATION KNOBS (EXPERIMENT-ONLY; never in any paper) -------
    # Every flag defaults to False so the full model path is
    # byte-identical to the unmodified InVAR. Each disables EXACTLY ONE
    # component and is independent of the others.
    abl_no_temporal_encoder: bool = False
    abl_no_spatial: bool = False
    abl_no_day_memory: bool = False
    abl_no_macro_film: bool = False
    abl_no_macro_head: bool = False
    abl_no_retrieval_bank: bool = False
    abl_no_swa: bool = False
    abl_random_retrieval: bool = False
    abl_shuffle_macro: bool = False
    # Canonical-mode switch (2026-05-16): the retrieval bank is NOT
    # load-bearing on the broad S&P panel (ablation A6: removing it gives
    # +0.0227 vs +0.0211 with it; random_retrieval = no change). The
    # active/default InVAR is therefore BANKLESS. Pass
    # --enable_retrieval_bank to restore the bank for an experiment.
    enable_retrieval_bank: bool = False
    # Output subdir tag (naming only; no behavioural effect).
    ablation_tag: str = ""
    # Universal-panel macro / betas feeds (verbatim).
    universal_macro_duration_parquet: str = (
        "data/processed/macro_duration_features_sp500.parquet"
    )
    biotech_macro_duration_parquet: str = (
        "data/processed/macro_duration_features.parquet"
    )
    universal_rolling_betas_parquet: str = (
        "data/processed/sp500_rolling_betas.parquet"
    )
    biotech_rolling_betas_parquet: str = (
        "data/processed/rolling_macro_betas.parquet"
    )


class GumbelTopKRetrievalBank(nn.Module):
    """Differentiable regime-retrieval bank ported from InVAR.

    Byte-identical to
    ``train_swa_invar_dgm_v2.GumbelTopKRetrievalBank`` (which is itself
    byte-identical to ``train_swa_invar_v2.GumbelTopKRetrievalBank``):
    a faithful re-implementation of the gumbel_topk path of InVAR's
    ``RegimeAxisRetrieval`` with learned ``keys`` / ``values``
    parameters. No data-leakage surface of its own.
    """

    def __init__(self, d_model: int, bank_size: int, top_k: int,
                 gumbel_tau: float, random_retrieval: bool = False) -> None:
        super().__init__()
        self.bank_size = int(bank_size)
        self.top_k = max(1, min(int(top_k), self.bank_size))
        self.gumbel_tau = float(gumbel_tau)
        # ABLATION (--abl_random_retrieval): when True the bank ignores
        # the query-key similarity and returns a uniformly random subset
        # of value slots (sanity baseline). Default False = original
        # gumbel_topk path, byte-identical to the unmodified InVAR.
        self.random_retrieval = bool(random_retrieval)
        self.keys = nn.Parameter(torch.randn(self.bank_size, d_model) * 0.02)
        self.values = nn.Parameter(torch.randn(self.bank_size, d_model) * 0.02)

    def forward(self, q_t: torch.Tensor) -> torch.Tensor:
        scores = self.keys @ q_t                                  # (bank_size,)
        k = min(self.top_k, self.bank_size)
        tau = max(self.gumbel_tau, 1.0e-3)
        if self.random_retrieval:
            # Sanity: random slot selection, equal weighting. Keeps the
            # parameter graph (values) so shapes/optimiser are unchanged.
            perm = torch.randperm(self.bank_size, device=self.values.device)
            ridx = perm[:k]
            w = q_t.new_full((k, 1), 1.0 / float(k))
            return self.values[ridx] * w                          # (k, d)
        if self.training:
            gumbel = -torch.log(-torch.log(
                torch.rand_like(scores).clamp(min=1.0e-9, max=1.0 - 1.0e-9),
            ))
            noisy = (scores + gumbel) / tau
        else:
            noisy = scores / tau
        soft_w = torch.softmax(noisy, dim=-1)                     # (bank_size,)
        top = torch.topk(soft_w, k=k, dim=-1)
        top_idx = top.indices
        top_soft = top.values
        return self.values[top_idx] * top_soft.unsqueeze(-1)      # (k, d)


def _init_sigmoid_gate_low(module: nn.Module, bias: float = -3.0) -> None:
    """Initialise the final Linear in a Sequential gate to a negative bias.

    Byte-faithful to RAG-STAR's
    ``src.v2.model.dow_epistar._init_sigmoid_gate_low`` and to
    ``train_swa_invar_dgm_v2._init_sigmoid_gate_low``: starts the
    sigmoid output near 0.05 instead of 0.5 so the macro head begins as
    a small residual correction to the strong backbone.
    """
    for layer in reversed(list(module)):
        if isinstance(layer, nn.Linear):
            nn.init.constant_(layer.bias, bias)
            break


class PerTickerTemporalEncoder(nn.Module):
    """Per-ticker temporal Transformer encoder (the temporal aspect).

    Encodes each active ticker's ``(T, F)`` lookback window into a
    single ``d_model`` vector. Clean in-file ``nn.Module`` (the vendored
    iTransformer files are NOT modified). Architecture per the brief:
    ``Linear(F -> d_model)`` input projection, a learned positional
    embedding over the ``T`` time steps, 2 ``TransformerEncoderLayer``
    (n_heads=4, d_ff=256, dropout=0.1, GELU, pre-norm, batch_first),
    then last-step pooling.

    Shapes: input ``(N_active, T, F)`` -> output ``(N_active, d_model)``.
    """

    def __init__(
        self,
        n_features: int,
        temporal_window: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        e_layers: int,
        dropout: float,
        activation: str,
    ) -> None:
        super().__init__()
        self.temporal_window = int(temporal_window)
        self.input_proj = nn.Linear(n_features, d_model)
        # Learned positional embedding over the T time steps.
        self.pos_emb = nn.Parameter(
            torch.randn(1, self.temporal_window, d_model) * 0.02
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=e_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x_window: Tensor) -> Tensor:
        """Encode ``(N_active, T, F)`` -> ``(N_active, d_model)``.

        Attention here runs across the T time axis (per ticker); the
        cross-ticker attention is a SEPARATE stage downstream.
        """
        # x_window: (N_active, T, F)
        h = self.input_proj(x_window)                          # (N, T, d)
        h = h + self.pos_emb[:, : h.shape[1], :]               # (N, T, d)
        h = self.encoder(h)                                    # (N, T, d)
        h = self.norm(h)
        return h[:, -1, :]                                     # (N, d)


class MacroFiLM(nn.Module):
    """Macro-conditioned FiLM modulation of the per-ticker tokens.

    The genuinely new, NON-GRAPH piece. From the daily macro-rate state
    vector ``m_state`` (produced by the SAME ``MacroStateEncoder`` the
    macro head uses) it produces a per-channel scale ``gamma`` and shift
    ``beta`` applied to every per-ticker token BEFORE the cross-ticker
    Encoder:
        token_t = token + g * ((gamma - 1) * token + beta)
    ``gamma`` is initialised to 1 and ``beta`` to 0 (the gamma/beta
    Linear weights+biases are zeroed, then a +1 bias is set on the gamma
    head), and ``g`` is a learned conservative scalar gate
    (sigmoid, bias init -3) so at init the modulation is the identity
    (InVAR-STX == plain temporal-encoder + cross-ticker iTransformer).
    There is NO adjacency / NO A_corr / NO A_dur / NO
    attention-bias-from-graph: this is dense modulation, not a graph.
    """

    def __init__(self, macro_state_dim: int, d_model: int,
                 gate_init_bias: float) -> None:
        super().__init__()
        self.gamma = nn.Linear(macro_state_dim, d_model)
        self.beta = nn.Linear(macro_state_dim, d_model)
        # Conservative scalar gate over the macro state.
        self.scale_gate = nn.Linear(macro_state_dim, 1)
        with torch.no_grad():
            # gamma -> 1 (zero weight, bias +1); beta -> 0 (zero both).
            self.gamma.weight.zero_()
            self.gamma.bias.fill_(1.0)
            self.beta.weight.zero_()
            self.beta.bias.zero_()
            # scalar gate sigmoid starts ~0.05 (identity modulation).
            self.scale_gate.weight.zero_()
            self.scale_gate.bias.fill_(float(gate_init_bias))

    def forward(self, tokens: Tensor, m_state: Tensor) -> Tensor:
        """Modulate ``(N_active, d_model)`` tokens by ``(macro_dim,)``.

        Args:
            tokens: per-ticker tokens ``(N_active, d_model)``.
            m_state: daily macro state ``(macro_state_dim,)``.

        Returns:
            FiLM-modulated tokens ``(N_active, d_model)``.
        """
        gamma = self.gamma(m_state).unsqueeze(0)               # (1, d)
        beta = self.beta(m_state).unsqueeze(0)                 # (1, d)
        g = torch.sigmoid(self.scale_gate(m_state)).squeeze()  # scalar
        delta = (gamma - 1.0) * tokens + beta                  # (N, d)
        return tokens + g.to(tokens.dtype) * delta             # (N, d)


class InvarSTXModel(nn.Module):
    """Per-ticker temporal encoder + macro-FiLM + cross-ticker
    iTransformer Encoder + SWA-InVAR retrieval + RAG-STAR day-memory +
    RAG-STAR macro rate-sensitivity head.

    STRICTLY NON-GRAPH. The backbone differs from
    ``train_swa_invar_dgm_v2.SWAInvarDGMModel`` only in
    ``_backbone_hidden`` (temporal encoder + macro-FiLM + vendored
    cross-ticker Encoder, replacing the flat ``Linear(T*F, d)``
    embedding). Every other sub-module (GumbelTopKRetrievalBank +
    cross_attn fusion, EpisodeMemoryBank day-memory fusion,
    DurationExposureEncoder / MacroStateEncoder macro head) is
    constructed and wired VERBATIM from SWA-InVAR-DGM, so the only
    behavioural change is the backbone.
    """

    def __init__(
        self,
        cfg: InvarSTXV2Config,
        n_features: int,
        day_key_dim: int,
        duration_input_dim: int,
        macro_input_dim: int,
        macro_gate_in_dim: int,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model

        # --- NEW backbone: per-ticker temporal encoder (temporal) ---
        self.temporal_encoder = PerTickerTemporalEncoder(
            n_features=n_features,
            temporal_window=cfg.temporal_window,
            d_model=d,
            n_heads=cfg.n_heads,
            d_ff=cfg.d_ff,
            e_layers=cfg.temporal_e_layers,
            dropout=cfg.dropout,
            activation=cfg.activation,
        )
        # ABLATION (--abl_no_temporal_encoder): the OLD flat token used
        # by SWA-InVAR-DGM, (N, T, F) -> (N, T*F) -> Linear(T*F, d).
        # Built ONLY when the flag is set; when the flag is off this
        # module does not exist and the path is byte-identical to the
        # unmodified InVAR. Everything downstream is identical.
        if cfg.abl_no_temporal_encoder:
            self.flat_token = nn.Linear(
                n_features * int(cfg.temporal_window), d
            )
        # --- NEW: macro-conditioned FiLM (identity-initialised) ---
        # macro_out_dim is the MacroStateEncoder output width (== the
        # m_state used by the macro head; reused verbatim, no new
        # macro encoder / no new macro input).
        self.macro_film = MacroFiLM(
            macro_state_dim=cfg.macro_out_dim,
            d_model=d,
            gate_init_bias=cfg.film_gate_init_bias,
        )
        # --- NEW: vendored iTransformer Encoder for CROSS-TICKER
        # attention. Built from the SAME vendored EncoderLayer /
        # AttentionLayer / FullAttention as ITransformerModel.encoder,
        # with d_model / n_heads / d_ff / e_layers, fed temporal-
        # embedding tokens. Attention runs across the N (ticker) token
        # axis. The vendored files are NOT modified.
        self.cross_ticker_encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(attention_dropout=cfg.dropout),
                        d,
                        cfg.n_heads,
                    ),
                    d,
                    cfg.d_ff,
                    dropout=cfg.dropout,
                    activation=cfg.activation,
                )
                for _ in range(cfg.e_layers)
            ],
            norm_layer=nn.LayerNorm(d),
        )

        # --- SWA-InVAR retrieval bank + fusion (byte-identical) ---
        self.bank = GumbelTopKRetrievalBank(
            d_model=d,
            bank_size=cfg.bank_size,
            top_k=cfg.top_k_retrieve,
            gumbel_tau=cfg.gumbel_tau,
            random_retrieval=cfg.abl_random_retrieval,
        )
        self.q_proj = nn.Linear(d, d)
        self.regime_norm = nn.LayerNorm(d)
        self.cross_attn = nn.MultiheadAttention(
            d, cfg.n_heads, dropout=cfg.dropout, batch_first=False,
        )
        with torch.no_grad():
            self.cross_attn.out_proj.weight.zero_()
            self.cross_attn.out_proj.bias.zero_()
        self.head = nn.Linear(d, 1)

        # --- (A) RAG-STAR day-level regime memory (verbatim) ---
        self.day_memory = EpisodeMemoryBank(
            EpisodeMemoryConfig(),
            key_dim=day_key_dim,
            value_dim=cfg.day_value_dim,
        )
        self.day_value_proj = nn.Linear(cfg.day_value_dim, d)
        self.day_cross_attn = nn.MultiheadAttention(
            embed_dim=d, num_heads=cfg.cross_attn_heads,
            dropout=cfg.dropout, batch_first=True,
        )
        self.day_fusion_mlp = nn.Sequential(
            nn.Linear(2 * d, d), nn.GELU(),
            nn.Dropout(cfg.dropout), nn.Linear(d, d),
        )
        day_gate_in_dim = 2 + 2 + day_key_dim
        self.day_gate_mlp = nn.Sequential(
            nn.Linear(day_gate_in_dim, cfg.gate_hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.gate_hidden_dim, 1),
        )
        with torch.no_grad():
            self.day_fusion_mlp[-1].weight.zero_()
            self.day_fusion_mlp[-1].bias.zero_()

        # --- (B) RAG-STAR macro rate-sensitivity head (verbatim) ---
        self.duration_encoder = DurationExposureEncoder(
            DurationExposureConfig(
                input_dim=duration_input_dim,
                hidden_dim=cfg.duration_hidden_dim,
                out_dim=cfg.duration_out_dim,
                dropout=cfg.dropout,
            )
        )
        self.macro_encoder = MacroStateEncoder(
            MacroStateConfig(
                input_dim=macro_input_dim,
                hidden_dim=cfg.macro_hidden_dim,
                out_dim=cfg.macro_out_dim,
                gate_state_dim=cfg.macro_gate_state_dim,
                dropout=cfg.dropout,
            )
        )
        d_dur = cfg.duration_out_dim
        m_mac = cfg.macro_out_dim
        if d_dur != m_mac:
            self.duration_proj = nn.Linear(d_dur, m_mac)
            common = m_mac
        else:
            self.duration_proj = nn.Identity()
            common = d_dur
        self.duration_head = nn.Sequential(
            nn.LayerNorm(2 * common + common),
            nn.Linear(2 * common + common, cfg.head_hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.head_dropout),
            nn.Linear(cfg.head_hidden_dim, 1),
        )
        self.lambda_gate = nn.Sequential(
            nn.Linear(macro_gate_in_dim, cfg.head_hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.head_hidden_dim, 1),
        )
        _init_sigmoid_gate_low(self.lambda_gate, bias=cfg.gate_init_bias)

    def _backbone_hidden(self, x_window: Tensor,
                         m_state: Tensor) -> Tensor:
        """NEW backbone -> per-ticker hiddens (N, d).

        Replaces the SWA-InVAR-DGM flat ``Linear(T*F, d)`` embedding
        with: per-ticker temporal encoder -> macro-FiLM -> vendored
        cross-ticker iTransformer Encoder.

        Args:
            x_window: ``(N_active, T, F)`` lookback window.
            m_state: daily macro state ``(macro_out_dim,)`` from the
                SAME MacroStateEncoder the macro head uses.

        Returns:
            ``(N_active, d_model)`` per-ticker hiddens.
        """
        # (a) per-ticker temporal encoder: (N, T, F) -> (N, d).
        # ABLATION (--abl_no_temporal_encoder): replace with the OLD
        # flat token (N, T, F) -> (N, T*F) -> Linear(T*F, d). When the
        # flag is off this branch is never taken and the path is the
        # unmodified temporal encoder.
        if self.cfg.abl_no_temporal_encoder:
            tok = self.flat_token(x_window.reshape(x_window.shape[0], -1))
        else:
            tok = self.temporal_encoder(x_window)              # (N, d)
        # (c) macro-conditioned FiLM (identity-initialised) on tokens.
        # ABLATION (--abl_no_macro_film): skip the FiLM module entirely
        # (hard identity: no gamma/beta/gate effect). Off by default =
        # unmodified FiLM call.
        if not self.cfg.abl_no_macro_film:
            tok = self.macro_film(tok, m_state)                # (N, d)
        # (b) cross-ticker spatial attention: feed (1, N, d) so the
        # vendored Encoder attends ACROSS the N (ticker) token axis.
        # ABLATION (--abl_no_spatial): skip the cross-ticker Encoder and
        # use the per-ticker token directly (no cross-ticker mixing).
        # Off by default = unmodified Encoder call.
        if self.cfg.abl_no_spatial:
            return tok                                         # (N, d)
        enc_in = tok.unsqueeze(0)                              # (1, N, d)
        enc_out = self.cross_ticker_encoder(enc_in, attn_mask=None)
        return enc_out.squeeze(0)                              # (N, d)

    def forward(
        self,
        x_window: Tensor,
        day_query_key: Tensor,
        query_day_idx: int,
        allowed_day_indices: Tensor,
        regime_scalars: Tensor,
        duration_input: Tensor,
        macro_input: Tensor,
        macro_gate_input: Tensor,
    ) -> Tensor:
        """Score every active ticker for a single day.

        Args:
            x_window: ``(N_active, T, F)`` lookback window.
            day_query_key: ``(day_key_dim,)`` raw regime key for the day.
            query_day_idx: integer panel day index.
            allowed_day_indices: training-day allowlist for day memory.
            regime_scalars: ``(2,)`` standardised regime scalars
                (VIX z, avg pairwise corr) for the day gate.
            duration_input: ``(N_active, duration_input_dim)`` per-ticker
                rate-sensitivity feature vector.
            macro_input: ``(macro_input_dim,)`` daily macro features.
            macro_gate_input: ``(macro_gate_in_dim,)`` daily gate scalars.

        Returns:
            ``(N_active,)`` raw ranking scores.
        """
        # Macro state from the SAME MacroStateEncoder the macro head
        # uses (computed once; fed to the FiLM backbone AND reused
        # verbatim by the macro head below).
        m_state, _ = self.macro_encoder(macro_input)           # (common,)

        h = self._backbone_hidden(x_window, m_state)           # (N, d)

        # --- Gumbel-topk regime retrieval bank ---
        # CANONICAL (2026-05-16): the bank is OFF by default (not
        # load-bearing on the broad S&P panel; ablation A6). It runs ONLY
        # if explicitly re-enabled via --enable_retrieval_bank, and the
        # --abl_no_retrieval_bank ablation flag still force-disables it.
        if self.cfg.enable_retrieval_bank and not self.cfg.abl_no_retrieval_bank:
            q_t = self.q_proj(h.mean(dim=0))                   # (d,)
            regime_tokens = self.bank(q_t)                     # (K, d)
            rk = self.regime_norm(regime_tokens).unsqueeze(1)  # (K, 1, d)
            hq = h.unsqueeze(1)                                # (N, 1, d)
            ca_out, _ = self.cross_attn(hq, rk, rk, need_weights=False)
            h = h + ca_out.squeeze(1)                          # (N, d)

        # --- (A) RAG-STAR day-level regime memory fusion (verbatim) ---
        # ABLATION (--abl_no_day_memory): skip the entire day-memory
        # retrieval + cross-attn + gated fusion (h unchanged). Off by
        # default = unmodified day-memory fusion.
        if not self.cfg.abl_no_day_memory:
            day_ret = self.day_memory.retrieve(
                query_raw_key=day_query_key,
                query_day_idx=query_day_idx,
                allowed_day_indices=allowed_day_indices,
            )
            day_proj = self.day_value_proj(day_ret["values"]).unsqueeze(0)
            h_day, _ = self.day_cross_attn(
                query=h.unsqueeze(0), key=day_proj, value=day_proj,
            )
            h_day = h_day.squeeze(0)                           # (N, d)
            day_q_std = self.day_memory.standardize_query(day_query_key)
            day_gate_in = torch.cat([
                day_ret["top1_sim"].unsqueeze(0),
                day_ret["sim_entropy"].unsqueeze(0),
                regime_scalars,
                day_q_std,
            ])
            alpha_day = torch.sigmoid(
                self.day_gate_mlp(day_gate_in)
            ).squeeze()
            h = h + alpha_day * self.day_fusion_mlp(
                torch.cat([h, h_day], dim=-1)
            )
        idio_score = self.head(h).squeeze(-1)                  # (N,)

        # --- (B) RAG-STAR macro rate-sensitivity head (verbatim) ---
        # ABLATION (--abl_no_macro_head): final score = idio only; the
        # duration/macro head + lambda_gate path is skipped entirely.
        # Off by default = unmodified macro-head path.
        if self.cfg.abl_no_macro_head:
            return idio_score                                  # (N,)
        d_exp = self.duration_encoder(duration_input)          # (N, d_dur)
        d_exp_proj = self.duration_proj(d_exp)                 # (N, common)
        m_state_b = m_state.unsqueeze(0).expand(d_exp_proj.shape[0], -1)
        interaction = torch.cat(
            [d_exp_proj, m_state_b, d_exp_proj * m_state_b], dim=-1
        )
        s_dur = self.duration_head(interaction).squeeze(-1)    # (N,)
        lambda_logit = self.lambda_gate(macro_gate_input).squeeze()
        lambda_macro = torch.sigmoid(lambda_logit)

        # Final per-ticker score = idio + lambda_macro * s_dur.
        return (idio_score
                + lambda_macro.to(idio_score.dtype)
                * s_dur.to(idio_score.dtype))                  # (N,)


def main() -> None:
    """CLI entry point. Mirrors train_swa_invar_dgm_v2.main argparse."""
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, choices=[1, 2, 3, 4, 5], required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true",
                   help="Limit to 2 epochs and abbreviated output.")
    p.add_argument("--max_epochs", type=int, default=None,
                   help="Override config.epochs (e.g. 1 for a smoke check).")
    p.add_argument("--panel_kind", type=str, default="biotech",
                   choices=["biotech", "lattice_native"])
    p.add_argument("--two_regime_val", action="store_true")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--panel_end", type=str, default=None)
    # ---- ABLATION FLAGS (EXPERIMENT-ONLY; never in any paper) -------
    # All default off (store_true) so the full model path is
    # byte-identical to the unmodified InVAR. Each disables EXACTLY ONE
    # component and is independent.
    p.add_argument("--abl_no_temporal_encoder", action="store_true",
                   help="Replace temporal encoder with the old flat token.")
    p.add_argument("--abl_no_spatial", action="store_true",
                   help="Skip the cross-ticker iTransformer Encoder.")
    p.add_argument("--abl_no_day_memory", action="store_true",
                   help="Disable day-level regime memory fusion.")
    p.add_argument("--abl_no_macro_film", action="store_true",
                   help="Disable the macro-conditioned spatial FiLM.")
    p.add_argument("--abl_no_macro_head", action="store_true",
                   help="Disable the macro rate-sensitivity head.")
    p.add_argument("--abl_no_retrieval_bank", action="store_true",
                   help="Disable the GumbelTopK retrieval-bank fusion.")
    p.add_argument("--abl_no_swa", action="store_true",
                   help="Disable SWA (use the live best-val iterate).")
    p.add_argument("--abl_random_retrieval", action="store_true",
                   help="Retrieval bank returns a random subset (sanity).")
    p.add_argument("--abl_shuffle_macro", action="store_true",
                   help="Permute macro state across days (sanity).")
    p.add_argument("--enable_retrieval_bank", action="store_true",
                   help="Re-enable the (off-by-default) retrieval bank; "
                        "canonical InVAR is bankless.")
    p.add_argument("--ablation_tag", type=str, default="",
                   help="Output subdir tag (naming only; no effect).")
    args = p.parse_args()

    cfg = InvarSTXV2Config(fold=args.fold, seed=args.seed)
    if args.smoke:
        cfg.epochs = 2
    if args.max_epochs is not None:
        cfg.epochs = int(args.max_epochs)
    cfg.panel_kind = args.panel_kind
    cfg.two_regime_val = args.two_regime_val
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.panel_end:
        cfg.panel_end = args.panel_end
    elif args.panel_kind == "lattice_native":
        cfg.panel_end = "2025-12-31"
    if cfg.swa_warmup_epochs >= cfg.epochs:
        cfg.swa_warmup_epochs = max(0, cfg.epochs - 1)

    # ---- Wire ablation flags into cfg. All default off => the full
    # model path stays byte-identical to the unmodified InVAR. ----
    cfg.abl_no_temporal_encoder = bool(args.abl_no_temporal_encoder)
    cfg.abl_no_spatial = bool(args.abl_no_spatial)
    cfg.abl_no_day_memory = bool(args.abl_no_day_memory)
    cfg.abl_no_macro_film = bool(args.abl_no_macro_film)
    cfg.abl_no_macro_head = bool(args.abl_no_macro_head)
    cfg.abl_no_retrieval_bank = bool(args.abl_no_retrieval_bank)
    cfg.abl_no_swa = bool(args.abl_no_swa)
    cfg.abl_random_retrieval = bool(args.abl_random_retrieval)
    cfg.abl_shuffle_macro = bool(args.abl_shuffle_macro)
    cfg.enable_retrieval_bank = bool(args.enable_retrieval_bank)
    cfg.ablation_tag = str(args.ablation_tag)
    # --abl_no_swa reuses the existing cfg.use_swa=False path verbatim;
    # the SWA EMA loop / _eval_split / best-state selection are
    # unchanged and simply take their already-present use_swa==False
    # branch.
    if cfg.abl_no_swa:
        cfg.use_swa = False

    set_seeds(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[InVAR-STX-v2] fold={cfg.fold} seed={cfg.seed} "
          f"device={device}")

    # ---- v2_runner data / fold / eval calls: BYTE-IDENTICAL to
    # train_swa_invar_dgm_v2.py (same args, same order). ----
    x_raw, y, tickers, dates = build_panel(cfg)
    T, N, Fdim = x_raw.shape
    print(f"[InVAR-STX-v2] panel: T={T} N={N} F={Fdim}")
    if N < 50:
        raise RuntimeError("Panel too small")

    mm = build_masks(cfg, dates, tickers)
    tradable = mm["tradable_mask"]
    loss_mask = mm["loss_mask"]
    hist20 = mm["history_valid_20d"]
    hist60 = mm["history_valid_60d"]

    train_idx, val_idx, test_idx = fold_split(cfg, dates)
    print(f"[InVAR-STX-v2] fold {cfg.fold}: "
          f"train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    x = standardize_features(x_raw, tradable, train_idx)
    age_feat = build_age_features(tradable, hist20, hist60)
    age_days = age_feat[..., 0].astype(np.int64)

    x_t = torch.from_numpy(x).to(device)
    y_t = torch.from_numpy(y).to(device)

    W = cfg.temporal_window

    # ------------------------------------------------------------------
    # (A) Day-level regime memory: keys + values exactly as RAG-STAR.
    #     (train_dow_epistar.py lines ~638-652.) Mirrors
    #     train_swa_invar_dgm_v2.py byte-for-byte.
    # ------------------------------------------------------------------
    day_keys, _ = build_episode_keys(
        dates=dates, log_returns=x_raw[..., 0], mask=tradable,
        cfg=EpisodeKeyConfig(),
    )
    feature_idx = [0, 1, 5, 6]
    n_summary = 2 * len(feature_idx) + 1
    day_values = np.zeros(
        (len(dates), day_keys.shape[1] + n_summary), dtype=np.float32,
    )
    day_values[:, : day_keys.shape[1]] = day_keys
    for t in range(len(dates)):
        m = tradable[t]
        if m.sum() < 5:
            continue
        for j, fi in enumerate(feature_idx):
            v = x_raw[t, m, fi]
            day_values[t, day_keys.shape[1] + 2 * j] = float(np.mean(v))
            day_values[t, day_keys.shape[1] + 2 * j + 1] = float(np.std(v))
        day_values[t, -1] = float(m.sum()) / 250.0
    cfg.day_value_dim = day_values.shape[1]

    # ------------------------------------------------------------------
    # (B) Macro input + macro-gate input: SAME composition RAG-STAR uses
    #     (train_dow_epistar.py lines ~685-725). Mirrors
    #     train_swa_invar_dgm_v2.py byte-for-byte.
    # ------------------------------------------------------------------
    if cfg.panel_kind == "lattice_native":
        macro_path = Path(cfg.universal_macro_duration_parquet)
    else:
        macro_path = Path(cfg.biotech_macro_duration_parquet)
    if not macro_path.exists():
        print("[InVAR-STX-v2] macro parquet missing; building...")
        build_macro_duration_features()
    macro = pd.read_parquet(macro_path)
    macro_arr, macro_cols, _ = standardize_macro_duration(
        macro, dates, train_idx,
    )
    print(f"[InVAR-STX-v2] macro features: {len(macro_cols)} dims")

    # ABLATION (--abl_shuffle_macro): permute the macro-state feature
    # matrix across the day axis (sanity baseline that destroys the
    # day<->macro alignment feeding MacroStateEncoder / FiLM / macro
    # head). Deterministic per seed for reproducibility. Off by default
    # => macro_arr is the unmodified standardised matrix and every
    # downstream consumer is byte-identical to the unmodified InVAR.
    if cfg.abl_shuffle_macro:
        _macro_perm = np.random.RandomState(cfg.seed).permutation(
            macro_arr.shape[0]
        )
        macro_arr = macro_arr[_macro_perm]
        print("[InVAR-STX-v2] ABLATION: macro state permuted across days")

    gate_indices = [macro_cols.index(c) for c in MACRO_GATE_COLS
                    if c in macro_cols]
    if len(gate_indices) != len(MACRO_GATE_COLS):
        missing = [c for c in MACRO_GATE_COLS if c not in macro_cols]
        print(f"[InVAR-STX-v2] WARN missing gate cols: {missing}")
    macro_gate_macro = macro_arr[:, gate_indices].astype(np.float32)
    avg_corr_idx = EPISODE_KEY_COLS.index("cs_avg_pairwise_corr_60d")
    cs_disp_idx = EPISODE_KEY_COLS.index("cs_dispersion")
    avg_corr = day_keys[:, avg_corr_idx].astype(np.float32)
    cs_disp = day_keys[:, cs_disp_idx].astype(np.float32)
    avg_corr_tr = avg_corr[train_idx]
    cs_disp_tr = cs_disp[train_idx]
    avg_corr_z = ((avg_corr - avg_corr_tr.mean())
                  / max(avg_corr_tr.std(), 1e-6)).astype(np.float32)
    cs_disp_z = ((cs_disp - cs_disp_tr.mean())
                 / max(cs_disp_tr.std(), 1e-6)).astype(np.float32)
    macro_gate_arr = np.concatenate(
        [macro_gate_macro, avg_corr_z[:, None], cs_disp_z[:, None]], axis=1,
    ).astype(np.float32)
    print(f"[InVAR-STX-v2] macro_gate input: "
          f"{macro_gate_arr.shape[1]} dims")

    # ------------------------------------------------------------------
    # (B) Per-ticker duration input: PANEL-KIND-AWARE column resolution,
    #     EXACTLY as RAG-STAR (train_dow_epistar.py lines ~766-780).
    #     Mirrors train_swa_invar_dgm_v2.py byte-for-byte.
    # ------------------------------------------------------------------
    duration_indices = resolve_duration_indices(cfg.panel_kind)
    duration_panel_block = _gather_or_zero(x, duration_indices).astype(
        np.float32
    )
    if cfg.panel_kind == "lattice_native":
        betas_path = Path(cfg.universal_rolling_betas_parquet)
    else:
        betas_path = Path(cfg.biotech_rolling_betas_parquet)
    if not betas_path.exists():
        print("[InVAR-STX-v2] rolling betas parquet missing; "
              "building...")
        build_rolling_betas()
        betas_path = Path(cfg.biotech_rolling_betas_parquet)
    betas_long = pd.read_parquet(betas_path)
    betas_tensor = betas_to_tensor(betas_long, dates, tickers)
    bt_train = betas_tensor[train_idx]
    train_mask = tradable[train_idx]
    betas_std = np.zeros_like(betas_tensor)
    for fi in range(betas_tensor.shape[-1]):
        vals = bt_train[..., fi][train_mask]
        if vals.size < 2:
            mu, sd = 0.0, 1.0
        else:
            mu = float(np.mean(vals))
            sd = float(np.std(vals))
            if sd < 1e-6:
                sd = 1.0
        betas_std[..., fi] = (betas_tensor[..., fi] - mu) / sd
    betas_std = (betas_std * tradable[..., None]).astype(np.float32)
    duration_input_full = np.concatenate(
        [duration_panel_block, age_feat, betas_std], axis=-1,
    ).astype(np.float32)
    duration_input_dim = duration_input_full.shape[-1]
    print(f"[InVAR-STX-v2] duration input dim: "
          f"{duration_input_dim} (panel={duration_panel_block.shape[-1]} "
          f"+ age={age_feat.shape[-1]} + betas={betas_std.shape[-1]})")

    # ------------------------------------------------------------------
    # Model + optimiser (SWA / AdamW / scheduler / scaler all identical
    # to train_swa_invar_dgm_v2.py).
    # ------------------------------------------------------------------
    model = InvarSTXModel(
        cfg,
        n_features=Fdim,
        day_key_dim=day_keys.shape[1],
        duration_input_dim=duration_input_dim,
        macro_input_dim=macro_arr.shape[1],
        macro_gate_in_dim=macro_gate_arr.shape[1],
    ).to(device)
    model.day_memory.populate(
        keys=day_keys, values=day_values,
        day_indices=np.arange(len(dates)),
        train_day_indices=train_idx,
    )
    model.day_memory.to(device)

    allowed_train = torch.from_numpy(train_idx).long().to(device)

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    total_steps = cfg.epochs * max(1, len(train_idx))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optim,
        lr_lambda=lambda s: warmup_cosine_lr(s, cfg.warmup_steps, total_steps),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    macro_arr_t = macro_arr.astype(np.float32)
    macro_gate_t = macro_gate_arr.astype(np.float32)

    def run_split(idx: np.ndarray, train_: bool):
        """Run one pass over ``idx`` days. Mirrors the SWA-InVAR loop."""
        model.train(train_)
        losses = []
        y_hat_all = np.zeros((T, N), dtype=np.float32)
        emask = np.zeros((T, N), dtype=bool)
        for t in idx:
            t = int(t)
            if t < W - 1:
                continue
            m_np = tradable[t]
            if m_np.sum() < 3:
                continue
            active_idx = np.flatnonzero(m_np)
            active_t = torch.from_numpy(active_idx).to(device)
            x_win = x_t[t - W + 1: t + 1, active_t, :].transpose(0, 1)
            y_target_full = y_t[t]
            lmask_t = torch.from_numpy(loss_mask[t]).to(device)

            day_query_key = torch.from_numpy(
                day_keys[t]
            ).float().to(device)
            regime_scalars = model.day_memory.standardize_query(
                day_query_key
            )[[0, 9]].clone()
            if torch.isnan(regime_scalars).any():
                regime_scalars = torch.zeros(2, device=device)

            dur_in = torch.from_numpy(
                duration_input_full[t, active_idx]
            ).float().to(device)
            macro_in = torch.from_numpy(
                macro_arr_t[t]
            ).float().to(device)
            macro_gate_in = torch.from_numpy(
                macro_gate_t[t]
            ).float().to(device)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                y_hat_active = model(
                    x_win,
                    day_query_key=day_query_key,
                    query_day_idx=t,
                    allowed_day_indices=allowed_train,
                    regime_scalars=regime_scalars,
                    duration_input=dur_in,
                    macro_input=macro_in,
                    macro_gate_input=macro_gate_in,
                )
                y_full = torch.zeros(N, device=device,
                                     dtype=y_hat_active.dtype)
                y_full[active_t] = y_hat_active
                cs_loss = cs_mse_loss(y_full, y_target_full, lmask_t)

            if train_:
                optim.zero_grad()
                scaler.scale(cs_loss).backward()
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               cfg.grad_clip)
                scaler.step(optim)
                scaler.update()
                scheduler.step()
                _maybe_update_swa()
            losses.append(float(cs_loss.item()))
            y_hat_all[t] = y_full.detach().float().cpu().numpy()
            emask[t] = loss_mask[t]
        return (float(np.mean(losses)) if losses else float("nan"),
                y_hat_all, emask)

    # SWA EMA state (byte-identical to train_swa_invar_dgm_v2.py,
    # including the torch.is_floating_point guard).
    ema_state: dict[str, torch.Tensor] | None = None
    swa_epoch_ref = {"epoch": 0}

    def _maybe_update_swa() -> None:
        nonlocal ema_state
        if not cfg.use_swa or swa_epoch_ref["epoch"] < cfg.swa_warmup_epochs:
            return
        with torch.no_grad():
            sd = model.state_dict()
            if ema_state is None:
                ema_state = {k: v.detach().clone() for k, v in sd.items()}
            else:
                d = float(cfg.swa_decay)
                for k in ema_state:
                    cur = sd[k].detach()
                    # EMA only floating-point params/buffers. Integer or
                    # Long buffers (e.g. retrieval index buffers,
                    # num_batches_tracked) cannot be averaged in-place
                    # (Float -> Long cast error); keep the latest value.
                    if torch.is_floating_point(ema_state[k]):
                        ema_state[k].mul_(d).add_(cur, alpha=1.0 - d)
                    else:
                        ema_state[k].copy_(cur)

    def _eval_split(idx: np.ndarray):
        if cfg.use_swa and ema_state is not None:
            saved = {k: v.detach().clone()
                     for k, v in model.state_dict().items()}
            model.load_state_dict(ema_state)
            res = run_split(idx, train_=False)
            model.load_state_dict(saved)
            return res
        return run_split(idx, train_=False)

    history: list = []
    best_val_ic = -1e9
    best_state = None
    patience = 0
    for epoch in range(cfg.epochs):
        t0 = time.time()
        swa_epoch_ref["epoch"] = epoch
        np.random.seed(cfg.seed + epoch)
        perm = np.random.permutation(train_idx)
        train_loss, _, _ = run_split(perm, train_=True)
        val_loss, val_yhat, val_mask = _eval_split(val_idx)
        val_metrics = evaluate_predictions(val_yhat, y, val_mask, age_days)
        dt = time.time() - t0
        improved = val_metrics["ic"] > best_val_ic + 1e-5
        print(f"[InVAR-STX-v2] epoch {epoch}: "
              f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"val_ic={val_metrics['ic']:+.4f} ({dt:.1f}s)"
              + ("  *best*" if improved else ""))
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_ic": val_metrics["ic"],
            "val_rank_ic": val_metrics["rank_ic"],
            "time_sec": round(dt, 2),
        })
        if improved:
            best_val_ic = val_metrics["ic"]
            src_state = (
                ema_state if (cfg.use_swa and ema_state is not None)
                else model.state_dict()
            )
            best_state = {k: v.detach().cpu().clone()
                          for k, v in src_state.items()}
            patience = 0
        else:
            patience += 1
            if patience >= cfg.early_stop_patience:
                print(f"[InVAR-STX-v2] early stop epoch {epoch} "
                      f"best_val_ic={best_val_ic:+.4f}")
                break

    if cfg.use_swa and ema_state is not None:
        final_state = {k: v.detach().cpu().clone()
                       for k, v in ema_state.items()}
        print("[InVAR-STX-v2] SWA: using final EMA state for test")
    elif best_state is not None:
        final_state = best_state
    else:
        final_state = {k: v.detach().cpu().clone()
                       for k, v in model.state_dict().items()}
    model.load_state_dict(final_state)

    _, test_yhat, test_mask = run_split(test_idx, train_=False)
    test_metrics = evaluate_predictions(test_yhat, y, test_mask, age_days)
    val_metrics_final = evaluate_predictions(val_yhat, y, val_mask, age_days)

    print(f"[InVAR-STX-v2] TEST ic={test_metrics['ic']:+.4f} "
          f"rank_ic={test_metrics['rank_ic']:+.4f} "
          f"ndcg10={test_metrics['ndcg10']:.4f} "
          f"ndcg50={test_metrics['ndcg50']:.4f}")

    # ---- Disk-safe result write. NO predictions npz. Same JSON schema
    # keys as v2_runner.save_result, including a "history" list whose
    # entries contain "epoch" so the sbatch skip-if-done test passes. ----
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"fold{cfg.fold}_seed{cfg.seed}.json"
    payload = {
        "fold": cfg.fold,
        "seed": cfg.seed,
        "model": "InVAR-STX (v2 protocol)",
        "panel_T": int(T),
        "panel_N": int(N),
        "panel_F": int(Fdim),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "ic": test_metrics["ic"],
        "rank_ic": test_metrics["rank_ic"],
        "ndcg10": test_metrics["ndcg10"],
        "ndcg50": test_metrics["ndcg50"],
        "test_cohort_ic": test_metrics["cohort_ic"],
        "val_ic": val_metrics_final["ic"],
        "val_rank_ic": val_metrics_final["rank_ic"],
        "val_cohort_ic": val_metrics_final["cohort_ic"],
        "history": history,
        "config": asdict(cfg),
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"[InVAR-STX-v2] wrote {out_path} (no npz; disk-safe)")


if __name__ == "__main__":
    main()
