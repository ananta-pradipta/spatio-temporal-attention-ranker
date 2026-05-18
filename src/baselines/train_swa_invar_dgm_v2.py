"""SWA-InVAR-DGM-nonGraph: SWA-InVAR + RAG-STAR day-memory + macro head.

EXPERIMENT-ONLY variant (NOT a paper baseline). It takes the exact
``src.baselines.train_swa_invar_v2`` model + harness and bolts on two
RAG-STAR (DOW-epiSTAR) components, ported analogously. This model is
STRICTLY NON-GRAPH: the iTransformer's dense cross-variate (cross-ticker)
self-attention IS the spatio-temporal backbone. There is ZERO graph
machinery: no correlation graph, no duration-similarity graph, no
GraphSourceGate, no attention-bias injection, no neighbour list. The
vendored iTransformer encoder runs UNMODIFIED, exactly as
``src.baselines.train_swa_invar_v2`` runs it.

The two RAG-STAR additions are:

  (A) Day-level regime memory
      The leakage-safe day-memory bank from RAG-STAR
      (``src.v2.model.episode_memory.EpisodeMemoryBank`` /
      ``EpisodeMemoryConfig``, top_m=8) populated with the SAME per-day
      14-d regime-fingerprint keys RAG-STAR uses
      (``src.v2.data.episode_keys.build_episode_keys`` over the panel's
      log_return column + tradable mask) and the SAME per-day value
      construction RAG-STAR uses in ``train_dow_epistar.py`` (day_keys
      concatenated with per-day cross-sectional mean/std summaries of
      feature columns [0, 1, 5, 6] plus an active count regulariser).
      Standardisation stats are computed on the fold train days only,
      populated exactly as RAG-STAR does
      (``populate(..., train_day_indices=train_idx)``). Retrieval
      respects the SAME embargo as RAG-STAR (EpisodeMemory's internal
      rule ``s + horizon + embargo < t`` with horizon=5, embargo=5, i.e.
      s < t - 10) and the SAME training-day allowlist
      (``allowed_day_indices = train_idx``). The fusion mirrors
      ``ow_epistar.py`` lines ~147-176: day_proj = day_value_proj
      (retrieved values); h_day = day_cross_attn(query=z, key/value=
      day_proj); alpha_day = sigmoid(day_gate_mlp([top1_sim,
      sim_entropy, regime_scalars, standardised query key]));
      z <- z + alpha_day * day_fusion_mlp([z, h_day]).

  (B) Macro rate-sensitivity HEAD (additive MLP residual, NOT a graph)
      Ported faithfully from RAG-STAR's DOW-epiSTAR Section E
      "macro-duration head" (``src.v2.model.dow_epistar.py``
      forward_day lines ~224-247) and its building blocks
      (``src.v2.model.duration_exposure.DurationExposureEncoder``,
      ``src.v2.model.macro_state.MacroStateEncoder``,
      ``src.v2.data.macro_duration_features``). Per active day:
        d_exp        = DurationEncoder(duration_input)
        d_exp_proj   = duration_proj(d_exp)
        m_state, _   = MacroEncoder(macro_input)
        interaction  = [d_exp_proj ; m_state ; d_exp_proj * m_state]
        s_dur        = duration_head(interaction)
        lambda_macro = sigmoid(lambda_gate(macro_gate_input))
      with the lambda_gate's FINAL Linear bias initialised to -3.0
      (RAG-STAR's conservative init, so lambda ~= 0.05 at start). The
      final per-ticker score is
        score = idio_score + lambda_macro * s_dur
      where idio_score is the SWA-InVAR (iTransformer + retrieval +
      day-memory fusion) score. The duration_input columns are resolved
      PANEL-KIND-AWARE for lattice_native via RAG-STAR's
      ``resolve_duration_indices`` / ``_gather_or_zero`` over
      ``_DURATION_SLOT_SEMANTICS`` (NOT the biotech column ordering).
      The duration_input is the SAME [panel-block ; age_feat ; betas]
      concat RAG-STAR builds. Macro features are standardised with
      train-fold stats only (``standardize_macro_duration``); the
      macro_gate input is the SAME composition RAG-STAR uses
      (MACRO_GATE_COLS + train-z avg_pairwise_corr_60d + train-z
      cs_dispersion).

Everything else (panel, masks, fold split, embargo, seeds, two-regime
val protocol, cs_mse_loss, SWA EMA, AdamW, warmup-cosine, fp16,
grad-clip, epochs=10, patience-3 on two-regime val IC,
evaluate_predictions) is BYTE-IDENTICAL to
``src.baselines.train_swa_invar_v2``. The ``src.baselines.v2_runner``
data / fold / eval calls (build_panel, build_masks, fold_split,
standardize_features, build_age_features, cs_mse_loss,
evaluate_predictions, set_seeds, warmup_cosine_lr) are invoked with the
SAME arguments in the SAME order as train_swa_invar_v2.py.

Disk: experiment-only and home storage is nearly full. This trainer does
NOT write the predictions npz. It writes only the fold{F}_seed{S}.json
(same schema keys as save_result, including a "history" list whose
entries contain "epoch" so skip-if-done works).

Run:
    python -m src.baselines.train_swa_invar_dgm_v2 --fold 1 --seed 42 \
        --panel_kind lattice_native --two_regime_val \
        --output_dir results/swa_invar_dgm
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
    ITransformerAdapter,
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
class SWAInvarDGMV2Config(V2BaselineConfig):
    """v2 protocol + SWA-InVAR knobs + RAG-STAR day-memory / macro-head knobs.

    The iTransformer backbone, retrieval-bank and SWA knobs are copied
    verbatim from ``SWAInvarV2Config`` so the SWA-InVAR core is
    byte-identical to ``train_swa_invar_v2.py``. The day-memory knobs
    mirror RAG-STAR (DOW-epiSTAR) defaults (14-d day key, top_m=8). The
    macro-head knobs mirror RAG-STAR's DurationExposureConfig /
    MacroStateConfig defaults.
    """

    output_dir: str = "results/swa_invar_dgm"
    # iTransformer backbone (verbatim from ITransformerV2Config / SWA-InVAR).
    d_model: int = 128
    n_heads: int = 4
    d_ff: int = 256
    e_layers: int = 2
    dropout: float = 0.1
    activation: str = "gelu"
    use_norm: bool = False
    # InVAR differentiable regime-retrieval bank (verbatim from SWA-InVAR).
    bank_size: int = 64
    top_k_retrieve: int = 32
    retrieval_mode: str = "gumbel_topk"
    gumbel_tau: float = 1.0
    # Stochastic Weight Averaging (verbatim from SWA-InVAR).
    use_swa: bool = True
    swa_decay: float = 0.999
    swa_warmup_epochs: int = 5
    # RAG-STAR day-memory (EpisodeMemoryConfig defaults RAG-STAR uses).
    day_value_dim: int = 32
    gate_hidden_dim: int = 64
    cross_attn_heads: int = 4
    # RAG-STAR macro rate-sensitivity head (DurationExposure / MacroState
    # defaults from src/v2/model; conservative lambda-gate bias init).
    duration_hidden_dim: int = 64
    duration_out_dim: int = 32
    macro_hidden_dim: int = 64
    macro_out_dim: int = 32
    macro_gate_state_dim: int = 16
    head_hidden_dim: int = 64
    head_dropout: float = 0.1
    gate_init_bias: float = -3.0
    # Universal-panel macro / betas feeds (same parquet paths RAG-STAR
    # uses for panel_kind="lattice_native").
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

    Byte-identical to ``train_swa_invar_v2.GumbelTopKRetrievalBank``: a
    faithful re-implementation of the gumbel_topk path of InVAR's
    ``RegimeAxisRetrieval`` with learned ``keys`` / ``values``
    parameters. No data-leakage surface of its own.
    """

    def __init__(self, d_model: int, bank_size: int, top_k: int,
                 gumbel_tau: float) -> None:
        super().__init__()
        self.bank_size = int(bank_size)
        self.top_k = max(1, min(int(top_k), self.bank_size))
        self.gumbel_tau = float(gumbel_tau)
        self.keys = nn.Parameter(torch.randn(self.bank_size, d_model) * 0.02)
        self.values = nn.Parameter(torch.randn(self.bank_size, d_model) * 0.02)

    def forward(self, q_t: torch.Tensor) -> torch.Tensor:
        scores = self.keys @ q_t                                  # (bank_size,)
        k = min(self.top_k, self.bank_size)
        tau = max(self.gumbel_tau, 1.0e-3)
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
    ``src.v2.model.dow_epistar._init_sigmoid_gate_low``: starts the
    sigmoid output near 0.05 instead of 0.5 so the macro head begins as
    a small residual correction to the strong SWA-InVAR backbone.
    """
    for layer in reversed(list(module)):
        if isinstance(layer, nn.Linear):
            nn.init.constant_(layer.bias, bias)
            break


class SWAInvarDGMModel(nn.Module):
    """SWA-InVAR + day-memory fusion + macro rate-sensitivity head.

    STRICTLY NON-GRAPH. The SWA-InVAR core (iTransformer backbone via
    ITransformerAdapter + GumbelTopKRetrievalBank + cross-attention
    fusion + linear head) is byte-identical to
    ``train_swa_invar_v2.SWAInvarModel``; the iTransformer encoder runs
    UNMODIFIED (dense cross-variate attention, no bias). Two RAG-STAR
    components are added:

      (A) day-memory: an ``EpisodeMemoryBank`` queried by the pooled
          backbone representation; fused via day_cross_attn + alpha_day
          gate + day_fusion_mlp, mirroring ow_epistar.py lines ~147-176.

      (B) macro rate-sensitivity head: an additive MLP residual ported
          from RAG-STAR's DOW-epiSTAR Section E. Per active day,
          s_dur = duration_head([d_exp_proj ; m_state ; d_exp_proj *
          m_state]); lambda_macro = sigmoid(lambda_gate(macro_gate));
          final score = idio_score + lambda_macro * s_dur. The
          lambda_gate's final bias is initialised to -3.0 so the head
          starts as a ~0.05 residual (SWA-InVAR-DGM-nonGraph == plain
          SWA-InVAR at init).
    """

    def __init__(
        self,
        cfg: SWAInvarDGMV2Config,
        n_features: int,
        day_key_dim: int,
        duration_input_dim: int,
        macro_input_dim: int,
        macro_gate_in_dim: int,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        hp = ITransformerHyperparams(
            d_feat=n_features,
            context_window=cfg.temporal_window,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            d_ff=cfg.d_ff,
            e_layers=cfg.e_layers,
            dropout=cfg.dropout,
            activation=cfg.activation,
            use_norm=cfg.use_norm,
            pred_len=1,
        )
        # Backbone built EXACTLY as train_swa_invar_v2.py builds it.
        self.backbone = ITransformerAdapter(hp)
        d = cfg.d_model

        # --- SWA-InVAR retrieval bank + fusion (byte-identical) ---
        self.bank = GumbelTopKRetrievalBank(
            d_model=d,
            bank_size=cfg.bank_size,
            top_k=cfg.top_k_retrieve,
            gumbel_tau=cfg.gumbel_tau,
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

        # --- (A) RAG-STAR day-level regime memory ---
        # EpisodeMemoryConfig() defaults exactly as RAG-STAR uses
        # (top_m=8, horizon_days=5, embargo_days=5, raw_key mode).
        self.day_memory = EpisodeMemoryBank(
            EpisodeMemoryConfig(),
            key_dim=day_key_dim,
            value_dim=cfg.day_value_dim,
        )
        # Fusion path mirrors ow_epistar.py lines ~147-176.
        self.day_value_proj = nn.Linear(cfg.day_value_dim, d)
        self.day_cross_attn = nn.MultiheadAttention(
            embed_dim=d, num_heads=cfg.cross_attn_heads,
            dropout=cfg.dropout, batch_first=True,
        )
        self.day_fusion_mlp = nn.Sequential(
            nn.Linear(2 * d, d), nn.GELU(),
            nn.Dropout(cfg.dropout), nn.Linear(d, d),
        )
        # Day-gate input is [top1_sim, sim_entropy, 2 regime scalars,
        # standardised day key] (same composition as ow_epistar.py
        # day_gate_in: 2 + 2 + day_key_dim).
        day_gate_in_dim = 2 + 2 + day_key_dim
        self.day_gate_mlp = nn.Sequential(
            nn.Linear(day_gate_in_dim, cfg.gate_hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.gate_hidden_dim, 1),
        )
        # Zero-init the fusion MLP output so day-memory starts as a
        # no-op (SWA-InVAR-DGM-nonGraph == plain SWA-InVAR at init).
        with torch.no_grad():
            self.day_fusion_mlp[-1].weight.zero_()
            self.day_fusion_mlp[-1].bias.zero_()

        # --- (B) RAG-STAR macro rate-sensitivity head ---
        # Faithful port of dow_epistar.py Section E + its building
        # blocks (DurationExposureEncoder, MacroStateEncoder). Not a
        # graph: a per-ticker additive MLP residual gated by a daily
        # macro scalar.
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
        # Projection if duration_out_dim != macro_out_dim for the
        # elementwise interaction (mirrors dow_epistar.py lines ~96-104).
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
        # Lambda gate over macro scalars (dow_epistar.py lines ~112-119).
        self.lambda_gate = nn.Sequential(
            nn.Linear(macro_gate_in_dim, cfg.head_hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.head_hidden_dim, 1),
        )
        # RAG-STAR v2.3 patch A: lambda_gate final bias = -3.0 so the
        # sigmoid starts at ~0.05 (conservative residual init).
        _init_sigmoid_gate_low(self.lambda_gate, bias=cfg.gate_init_bias)

    def _backbone_hidden(self, x_window: Tensor) -> Tensor:
        """iTransformer backbone -> per-ticker hiddens (N, d).

        Byte-identical to
        ``train_swa_invar_v2.SWAInvarModel._backbone_hidden``: the
        vendored iTransformer encoder runs UNMODIFIED with dense
        cross-variate self-attention and no attention bias.
        """
        n_active, _, _ = x_window.shape
        x_flat = x_window.reshape(n_active, -1)                # (N, T*F)
        x_in = x_flat.transpose(0, 1).unsqueeze(0)             # (1, T*F, N)
        m = self.backbone.model
        if m.use_norm:
            means = x_in.mean(1, keepdim=True).detach()
            x_in = x_in - means
            stdev = torch.sqrt(
                torch.var(x_in, dim=1, keepdim=True, unbiased=False) + 1e-5
            )
            x_in = x_in / stdev
        enc_out = m.enc_embedding(x_in)                        # (1, N, d)
        enc_out = m.encoder(enc_out, attn_mask=None)           # (1, N, d)
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
        h = self._backbone_hidden(x_window)                    # (N, d)

        # --- SWA-InVAR Gumbel-topk regime retrieval (byte-identical) ---
        q_t = self.q_proj(h.mean(dim=0))                       # (d,)
        regime_tokens = self.bank(q_t)                         # (K, d)
        rk = self.regime_norm(regime_tokens).unsqueeze(1)      # (K, 1, d)
        hq = h.unsqueeze(1)                                    # (N, 1, d)
        ca_out, _ = self.cross_attn(hq, rk, rk, need_weights=False)
        h = h + ca_out.squeeze(1)                              # (N, d)

        # --- (A) RAG-STAR day-level regime memory fusion ---
        # Mirrors ow_epistar.py lines ~147-176 exactly.
        day_ret = self.day_memory.retrieve(
            query_raw_key=day_query_key,
            query_day_idx=query_day_idx,
            allowed_day_indices=allowed_day_indices,
        )
        day_proj = self.day_value_proj(day_ret["values"]).unsqueeze(0)
        h_day, _ = self.day_cross_attn(
            query=h.unsqueeze(0), key=day_proj, value=day_proj,
        )
        h_day = h_day.squeeze(0)                               # (N, d)
        day_q_std = self.day_memory.standardize_query(day_query_key)
        day_gate_in = torch.cat([
            day_ret["top1_sim"].unsqueeze(0),
            day_ret["sim_entropy"].unsqueeze(0),
            regime_scalars,
            day_q_std,
        ])
        alpha_day = torch.sigmoid(self.day_gate_mlp(day_gate_in)).squeeze()
        h = h + alpha_day * self.day_fusion_mlp(
            torch.cat([h, h_day], dim=-1)
        )
        idio_score = self.head(h).squeeze(-1)                  # (N,)

        # --- (B) RAG-STAR macro rate-sensitivity head ---
        # Faithful port of dow_epistar.py forward_day Section E
        # (lines ~224-247). Additive MLP residual, NOT a graph.
        d_exp = self.duration_encoder(duration_input)          # (N, d_dur)
        d_exp_proj = self.duration_proj(d_exp)                 # (N, common)
        m_state, _ = self.macro_encoder(macro_input)           # (common,)
        m_state_b = m_state.unsqueeze(0).expand(d_exp_proj.shape[0], -1)
        interaction = torch.cat(
            [d_exp_proj, m_state_b, d_exp_proj * m_state_b], dim=-1
        )
        s_dur = self.duration_head(interaction).squeeze(-1)    # (N,)
        lambda_logit = self.lambda_gate(macro_gate_input).squeeze()
        lambda_macro = torch.sigmoid(lambda_logit)

        # Final per-ticker score = idio + lambda_macro * s_dur
        # (dow_epistar.py lines ~291-296).
        return (idio_score
                + lambda_macro.to(idio_score.dtype)
                * s_dur.to(idio_score.dtype))                  # (N,)


def main() -> None:
    """CLI entry point. Mirrors train_swa_invar_v2.main argparse."""
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
    args = p.parse_args()

    cfg = SWAInvarDGMV2Config(fold=args.fold, seed=args.seed)
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

    set_seeds(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SWA-InVAR-DGM-nG-v2] fold={cfg.fold} seed={cfg.seed} "
          f"device={device}")

    # ---- v2_runner data / fold / eval calls: BYTE-IDENTICAL to
    # train_swa_invar_v2.py (same args, same order). ----
    x_raw, y, tickers, dates = build_panel(cfg)
    T, N, Fdim = x_raw.shape
    print(f"[SWA-InVAR-DGM-nG-v2] panel: T={T} N={N} F={Fdim}")
    if N < 50:
        raise RuntimeError("Panel too small")

    mm = build_masks(cfg, dates, tickers)
    tradable = mm["tradable_mask"]
    loss_mask = mm["loss_mask"]
    hist20 = mm["history_valid_20d"]
    hist60 = mm["history_valid_60d"]

    train_idx, val_idx, test_idx = fold_split(cfg, dates)
    print(f"[SWA-InVAR-DGM-nG-v2] fold {cfg.fold}: "
          f"train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    x = standardize_features(x_raw, tradable, train_idx)
    age_feat = build_age_features(tradable, hist20, hist60)
    age_days = age_feat[..., 0].astype(np.int64)

    x_t = torch.from_numpy(x).to(device)
    y_t = torch.from_numpy(y).to(device)

    W = cfg.temporal_window

    # ------------------------------------------------------------------
    # (A) Day-level regime memory: keys + values exactly as RAG-STAR.
    #     (train_dow_epistar.py lines ~638-652.)
    #
    #     RAG-STAR uses feature_idx = [0, 1, 5, 6] positionally and
    #     UNCONDITIONALLY (no panel-kind resolution). This is safe /
    #     panel-agnostic for these specific columns: in
    #     train_dow_epistar._PANEL_SEMANTIC_MAP, indices [0,1,5,6] map to
    #     the SAME semantics under BOTH "biotech" and "lattice_native"
    #     (log_return=0, log_return_5d=1, rv_20d=5, rv_60d=6). So
    #     mirroring RAG-STAR byte-for-byte here is already correct for
    #     lattice_native.
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
    #     (train_dow_epistar.py lines ~685-725): standardised
    #     MACRO_FEATURE_COLS_FULL for the macro encoder; MACRO_GATE_COLS
    #     + train-z avg_pairwise_corr_60d + train-z cs_dispersion for the
    #     lambda gate. Standardisation uses train-fold stats only.
    # ------------------------------------------------------------------
    if cfg.panel_kind == "lattice_native":
        macro_path = Path(cfg.universal_macro_duration_parquet)
    else:
        macro_path = Path(cfg.biotech_macro_duration_parquet)
    if not macro_path.exists():
        print("[SWA-InVAR-DGM-nG-v2] macro parquet missing; building...")
        build_macro_duration_features()
    macro = pd.read_parquet(macro_path)
    macro_arr, macro_cols, _ = standardize_macro_duration(
        macro, dates, train_idx,
    )
    print(f"[SWA-InVAR-DGM-nG-v2] macro features: {len(macro_cols)} dims")

    gate_indices = [macro_cols.index(c) for c in MACRO_GATE_COLS
                    if c in macro_cols]
    if len(gate_indices) != len(MACRO_GATE_COLS):
        missing = [c for c in MACRO_GATE_COLS if c not in macro_cols]
        print(f"[SWA-InVAR-DGM-nG-v2] WARN missing gate cols: {missing}")
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
    print(f"[SWA-InVAR-DGM-nG-v2] macro_gate input: "
          f"{macro_gate_arr.shape[1]} dims")

    # ------------------------------------------------------------------
    # (B) Per-ticker duration input: PANEL-KIND-AWARE column resolution,
    #     EXACTLY as RAG-STAR (train_dow_epistar.py lines ~766-780):
    #         duration_indices = resolve_duration_indices(panel_kind)
    #         duration_panel_block = _gather_or_zero(x, duration_indices)
    #         duration_input_full = concat([panel_block, age_feat, betas])
    #     resolve_duration_indices maps _DURATION_SLOT_SEMANTICS through
    #     _PANEL_SEMANTIC_MAP[panel_kind]; for lattice_native the
    #     unavailable slots resolve to None and _gather_or_zero
    #     zero-fills them (NOT the biotech column ordering).
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
        print("[SWA-InVAR-DGM-nG-v2] rolling betas parquet missing; "
              "building...")
        build_rolling_betas()
        betas_path = Path(cfg.biotech_rolling_betas_parquet)
    betas_long = pd.read_parquet(betas_path)
    betas_tensor = betas_to_tensor(betas_long, dates, tickers)
    # Standardise betas using train-fold cells only (per-feature),
    # exactly as RAG-STAR (train_dow_epistar.py lines ~751-763).
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
    print(f"[SWA-InVAR-DGM-nG-v2] duration input dim: "
          f"{duration_input_dim} (panel={duration_panel_block.shape[-1]} "
          f"+ age={age_feat.shape[-1]} + betas={betas_std.shape[-1]})")

    # ------------------------------------------------------------------
    # Model + optimiser (SWA / AdamW / scheduler / scaler all identical
    # to train_swa_invar_v2.py).
    # ------------------------------------------------------------------
    model = SWAInvarDGMModel(
        cfg,
        n_features=Fdim,
        day_key_dim=day_keys.shape[1],
        duration_input_dim=duration_input_dim,
        macro_input_dim=macro_arr.shape[1],
        macro_gate_in_dim=macro_gate_arr.shape[1],
    ).to(device)
    # Populate the day-memory exactly as RAG-STAR does (train fold stats
    # only; train_dow_epistar.py lines ~928-931).
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

    # SWA EMA state (byte-identical to train_swa_invar_v2.py).
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
        print(f"[SWA-InVAR-DGM-nG-v2] epoch {epoch}: "
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
                print(f"[SWA-InVAR-DGM-nG-v2] early stop epoch {epoch} "
                      f"best_val_ic={best_val_ic:+.4f}")
                break

    if cfg.use_swa and ema_state is not None:
        final_state = {k: v.detach().cpu().clone()
                       for k, v in ema_state.items()}
        print("[SWA-InVAR-DGM-nG-v2] SWA: using final EMA state for test")
    elif best_state is not None:
        final_state = best_state
    else:
        final_state = {k: v.detach().cpu().clone()
                       for k, v in model.state_dict().items()}
    model.load_state_dict(final_state)

    _, test_yhat, test_mask = run_split(test_idx, train_=False)
    test_metrics = evaluate_predictions(test_yhat, y, test_mask, age_days)
    val_metrics_final = evaluate_predictions(val_yhat, y, val_mask, age_days)

    print(f"[SWA-InVAR-DGM-nG-v2] TEST ic={test_metrics['ic']:+.4f} "
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
        "model": "SWA-InVAR-DGM-nonGraph (v2 protocol)",
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
    print(f"[SWA-InVAR-DGM-nG-v2] wrote {out_path} (no npz; disk-safe)")


if __name__ == "__main__":
    main()
