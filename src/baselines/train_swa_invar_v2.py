"""SWA-InVAR trainer using the v2 protocol (matches RAG-STAR exactly).

This script reports an author's-design-space comparison point. It runs
on the IDENTICAL panel, masks, fold definitions, embargo, seeds, loss,
and metrics as ``src.baselines.train_itransformer_v2`` and the headline
RAG-STAR sweep. The ONLY thing that differs is the model.

SWA-InVAR composes three pieces:

  1. Backbone: the vendored inverted-Transformer (iTransformer; Liu, Hu,
     Liu, Zhou, Li, Long, ICLR 2024) wrapped by
     ``ITransformerAdapter`` exactly as the iTransformer baseline builds
     it (d_model=128, n_heads=4, d_ff=256, e_layers=2). It produces a
     per-day, per-ticker latent representation and a base ranking score.

  2. Differentiable regime-retrieval bank: a port of InVAR's
     ``RegimeAxisRetrieval`` (src/invar/model/invar.py, lines ~199-294)
     run in ``retrieval_mode="gumbel_topk"``. A learned key-value memory
     bank is queried by a per-day regime query derived from the
     backbone's pooled representation; Gumbel-noise top-K selection with
     a straight-through estimator returns weighted regime values that are
     fused back into every active ticker's representation before the
     ranking head. Because the bank is parameterised (keys and values are
     nn.Parameters, not gathered training-day tensors), it carries no
     data-leakage surface of its own; the SAME leakage discipline as the
     v2 harness is enforced because the bank parameters only ever receive
     gradient on ``train_idx`` days, under the harness's tradable / loss
     masks, with the harness embargo baked into ``fold_split``.

  3. Stochastic Weight Averaging (Izmailov et al. 2018, "Averaging
     Weights Leads to Wider Optima and Better Generalization"). After a
     burn-in of ``swa_warmup_epochs`` epochs, an exponential moving
     average of the full model ``state_dict`` is maintained each training
     step (mirroring the SWA block in src/invar/training/train.py,
     lines ~442-456 and ~547-557). Validation early-stop selection and
     final test scoring are performed on the SWA-averaged weights, not
     the raw SGD iterate.

Loss: pure cross-sectional MSE on z-scored 5d forward log returns
(``cs_mse_loss``), identical to every other v2 baseline. No auxiliary
losses.

Run:
    python -m src.baselines.train_swa_invar_v2 --fold 1 --seed 42 \
        --panel_kind lattice_native --two_regime_val

Output: results/<output_dir>/fold{F}_seed{S}.json (+ predictions npz),
written via ``save_result`` with the same schema as the other v2
baselines.
"""
from __future__ import annotations

import argparse
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

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
    save_result,
    set_seeds,
    standardize_features,
    warmup_cosine_lr,
)


@dataclass
class SWAInvarV2Config(V2BaselineConfig):
    """Top-level config bundling the v2 protocol + SWA-InVAR knobs.

    Backbone knobs (d_model, n_heads, d_ff, e_layers, dropout,
    activation, use_norm) are copied verbatim from
    ``ITransformerV2Config`` so the backbone is byte-identical to the
    iTransformer baseline. The retrieval-bank and SWA knobs mirror the
    InVAR defaults at the gumbel_topk + SWA commit.

    Retrieval-bank rationale (ported from InVAR ``InvarConfig``):
      - bank_size=64 and top_k_retrieve=32 are InVAR's v3 defaults.
      - retrieval_mode="gumbel_topk": Gumbel-noise top-K with a
        straight-through estimator so gradients flow into every key via
        the softmax denominator (InVAR's strongest diff-retrieval mode).
      - gumbel_tau=1.0 matches InVAR's default temperature.

    SWA rationale (Izmailov et al. 2018):
      - swa_warmup_epochs=5: average only the back half of the 10-epoch
        budget, after the SGD iterate has reached a low-loss basin.
      - swa_decay=0.999: exponential moving average of the state_dict;
        identical to the InVAR SWA-InVAR active-head decay.
    """

    output_dir: str = "results/baselines_universal_two_regime_val/swa_invar"
    # iTransformer backbone (verbatim from ITransformerV2Config).
    d_model: int = 128
    n_heads: int = 4
    d_ff: int = 256
    e_layers: int = 2
    dropout: float = 0.1
    activation: str = "gelu"
    use_norm: bool = False
    # InVAR differentiable regime-retrieval bank.
    bank_size: int = 64
    top_k_retrieve: int = 32
    retrieval_mode: str = "gumbel_topk"
    gumbel_tau: float = 1.0
    # Stochastic Weight Averaging (Izmailov et al. 2018).
    use_swa: bool = True
    swa_decay: float = 0.999
    swa_warmup_epochs: int = 5


class GumbelTopKRetrievalBank(nn.Module):
    """Differentiable regime-retrieval bank ported from InVAR.

    This is a faithful re-implementation of
    ``src.invar.model.invar.RegimeAxisRetrieval`` restricted to the
    ``gumbel_topk`` path (src/invar/model/invar.py lines ~199-294). The
    bank holds learned ``keys`` and ``values`` (nn.Parameters). Given a
    per-day regime query, scores = keys @ q; Gumbel(0,1) noise is added
    to scores at train time; a softmax over the noisy scores yields
    weights; the top-K positions are selected and their values returned
    weighted by the soft weights (straight-through: forward uses the
    soft weights at selected positions, backward inherits the full
    softmax gradient so every key receives gradient).

    The bank is parameterised, not populated from training-day tensors,
    so it carries no data-leakage surface of its own.
    """

    def __init__(self, d_model: int, bank_size: int, top_k: int,
                 gumbel_tau: float) -> None:
        """Build the learned key-value bank.

        Args:
            d_model: regime-token width (matches backbone d_model).
            bank_size: number of slots in the memory bank.
            top_k: number of slots returned per query.
            gumbel_tau: Gumbel-softmax temperature.
        """
        super().__init__()
        self.bank_size = int(bank_size)
        self.top_k = max(1, min(int(top_k), self.bank_size))
        self.gumbel_tau = float(gumbel_tau)
        self.keys = nn.Parameter(torch.randn(self.bank_size, d_model) * 0.02)
        self.values = nn.Parameter(torch.randn(self.bank_size, d_model) * 0.02)

    def forward(self, q_t: torch.Tensor) -> torch.Tensor:
        """Retrieve regime tokens for a single day.

        Args:
            q_t: per-day regime query, shape ``(d_model,)``.

        Returns:
            Regime tokens, shape ``(top_k, d_model)``, weighted by the
            Gumbel-softmax weights at the selected positions.
        """
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


class SWAInvarModel(nn.Module):
    """iTransformer backbone + Gumbel-topk regime retrieval + ranking head.

    Per active day the model:
      1. Runs the whole active panel through ``ITransformerAdapter`` so
         cross-ticker attention can fire (identical to the iTransformer
         baseline's forward pass).
      2. Reads the backbone's pre-head hidden states (one vector per
         active ticker), pools them with a masked mean to form the
         per-day regime query ``q_t``.
      3. Retrieves ``top_k`` regime value tokens from the differentiable
         bank, attends each ticker's hidden state to the retrieved
         tokens via a single multi-head cross-attention, and adds the
         result back (residual) before a linear ranking head.

    SWA is handled by the trainer (it averages this module's
    state_dict); the module itself is SWA-agnostic.
    """

    def __init__(self, cfg: SWAInvarV2Config, n_features: int) -> None:
        """Construct the composed model.

        Args:
            cfg: SWA-InVAR config (backbone + bank knobs).
            n_features: panel feature dimension F.
        """
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
        # Backbone built EXACTLY as train_itransformer_v2.py builds it.
        self.backbone = ITransformerAdapter(hp)
        d = cfg.d_model
        self.bank = GumbelTopKRetrievalBank(
            d_model=d,
            bank_size=cfg.bank_size,
            top_k=cfg.top_k_retrieve,
            gumbel_tau=cfg.gumbel_tau,
        )
        # Query projection from the pooled backbone representation to the
        # bank-key space, and a single cross-attention to fuse retrieved
        # regime tokens back into each ticker's representation. This
        # mirrors InVAR's variate-to-regime cross-attention path
        # (InvarBlock.cross_attn, src/invar/model/invar.py lines ~324-382)
        # in a single block, zero-initialised so SWA-InVAR starts as the
        # plain iTransformer baseline and only deviates once the bank
        # accumulates gradient signal.
        self.q_proj = nn.Linear(d, d)
        self.regime_norm = nn.LayerNorm(d)
        self.cross_attn = nn.MultiheadAttention(
            d, cfg.n_heads, dropout=cfg.dropout, batch_first=False,
        )
        with torch.no_grad():
            self.cross_attn.out_proj.weight.zero_()
            self.cross_attn.out_proj.bias.zero_()
        self.head = nn.Linear(d, 1)

    def _backbone_hidden(self, x_window: torch.Tensor) -> torch.Tensor:
        """Run the iTransformer backbone and return per-ticker hiddens.

        Reproduces ``ITransformerAdapter.forward`` up to the final
        projection so we can intercept the per-variate (per-ticker)
        d_model hidden states for retrieval fusion.

        Args:
            x_window: ``(N_active, T, F)`` lookback window.

        Returns:
            Per-ticker hidden states, shape ``(N_active, d_model)``.
        """
        n_active, _, _ = x_window.shape
        x_flat = x_window.reshape(n_active, -1)                # (N, T*F)
        x_in = x_flat.transpose(0, 1).unsqueeze(0)             # (1, T*F, N)
        m = self.backbone.model
        # iTransformer forward up to (but not including) the per-variate
        # projector. Mirrors ITransformerModel.forward (vendored
        # module.py lines ~237-250). use_norm is False under the v2
        # protocol so no non-stationary rescale is applied; we keep the
        # branch for faithfulness with the vendored backbone.
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

    def forward(self, x_window: torch.Tensor) -> torch.Tensor:
        """Score every active ticker for a single day.

        Args:
            x_window: ``(N_active, T, F)`` lookback window.

        Returns:
            ``(N_active,)`` raw ranking scores.
        """
        h = self._backbone_hidden(x_window)                    # (N, d)
        # Per-day regime query: masked mean of ticker hiddens projected
        # into the bank-key space.
        q_t = self.q_proj(h.mean(dim=0))                       # (d,)
        regime_tokens = self.bank(q_t)                         # (K, d)
        rk = self.regime_norm(regime_tokens).unsqueeze(1)      # (K, 1, d)
        hq = h.unsqueeze(1)                                    # (N, 1, d)
        ca_out, _ = self.cross_attn(hq, rk, rk, need_weights=False)
        h = h + ca_out.squeeze(1)                              # (N, d)
        return self.head(h).squeeze(-1)                        # (N,)


def main() -> None:
    """CLI entry point. Mirrors train_itransformer_v2.main argparse."""
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

    cfg = SWAInvarV2Config(fold=args.fold, seed=args.seed)
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
    # If SWA burn-in would consume the whole (possibly shortened) budget,
    # fall back to averaging the final epoch only.
    if cfg.swa_warmup_epochs >= cfg.epochs:
        cfg.swa_warmup_epochs = max(0, cfg.epochs - 1)

    set_seeds(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SWA-InVAR-v2] fold={cfg.fold} seed={cfg.seed} device={device}")

    x_raw, y, tickers, dates = build_panel(cfg)
    T, N, Fdim = x_raw.shape
    print(f"[SWA-InVAR-v2] panel: T={T} N={N} F={Fdim}")
    if N < 50:
        raise RuntimeError("Panel too small")

    mm = build_masks(cfg, dates, tickers)
    tradable = mm["tradable_mask"]
    loss_mask = mm["loss_mask"]
    hist20 = mm["history_valid_20d"]
    hist60 = mm["history_valid_60d"]

    train_idx, val_idx, test_idx = fold_split(cfg, dates)
    print(f"[SWA-InVAR-v2] fold {cfg.fold}: "
          f"train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    x = standardize_features(x_raw, tradable, train_idx)
    age_feat = build_age_features(tradable, hist20, hist60)
    age_days = age_feat[..., 0].astype(np.int64)

    x_t = torch.from_numpy(x).to(device)
    y_t = torch.from_numpy(y).to(device)

    W = cfg.temporal_window

    model = SWAInvarModel(cfg, n_features=Fdim).to(device)
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    total_steps = cfg.epochs * max(1, len(train_idx))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optim, lr_lambda=lambda s: warmup_cosine_lr(s, cfg.warmup_steps, total_steps)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    def run_split(idx: np.ndarray, train_: bool) -> tuple[float, np.ndarray, np.ndarray]:
        """Run one pass over ``idx`` days. Mirrors the iTransformer loop."""
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
            x_win = x_t[t - W + 1: t + 1, active_t, :].transpose(0, 1)  # (A, W, F)
            y_target_full = y_t[t]                                       # (N,)
            lmask_t = torch.from_numpy(loss_mask[t]).to(device)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                y_hat_active = model(x_win)                              # (A,)
                y_full = torch.zeros(N, device=device, dtype=y_hat_active.dtype)
                y_full[active_t] = y_hat_active
                cs_loss = cs_mse_loss(y_full, y_target_full, lmask_t)

            if train_:
                optim.zero_grad()
                scaler.scale(cs_loss).backward()
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optim)
                scaler.update()
                scheduler.step()
                _maybe_update_swa()
            losses.append(float(cs_loss.item()))
            y_hat_all[t] = y_full.detach().float().cpu().numpy()
            emask[t] = loss_mask[t]
        return (float(np.mean(losses)) if losses else float("nan"),
                y_hat_all, emask)

    # SWA-InVAR weight-averaging state. Faithfully mirrors the EMA block
    # in src/invar/training/train.py (lines ~442-456): after the burn-in
    # epoch, accumulate an exponential moving average of the full
    # state_dict; evaluate / test on the averaged copy.
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
                    ema_state[k].mul_(d).add_(sd[k].detach(), alpha=1.0 - d)

    def _eval_split(idx: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        """Evaluate ``idx`` on the SWA-averaged weights when available.

        Restores the live SGD iterate afterwards so training continues
        on its own trajectory (mirrors train.py lines ~464-470).
        """
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
        print(f"[SWA-InVAR-v2] epoch {epoch}: train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} val_ic={val_metrics['ic']:+.4f} "
              f"({dt:.1f}s)" + ("  *best*" if improved else ""))
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
            # Track the SWA-averaged weights when SWA is active, else the
            # live iterate (mirrors train.py lines ~534-540).
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
                print(f"[SWA-InVAR-v2] early stop epoch {epoch} "
                      f"best_val_ic={best_val_ic:+.4f}")
                break

    # SWA standard protocol (Izmailov et al. 2018; train.py lines
    # ~547-557): final test on the final averaged weights. If SWA never
    # populated (e.g. very short smoke run), fall back to the best-val
    # state, then to the live iterate.
    if cfg.use_swa and ema_state is not None:
        final_state = {k: v.detach().cpu().clone()
                       for k, v in ema_state.items()}
        print("[SWA-InVAR-v2] SWA: using final EMA state for test eval")
    elif best_state is not None:
        final_state = best_state
    else:
        final_state = {k: v.detach().cpu().clone()
                       for k, v in model.state_dict().items()}
    model.load_state_dict(final_state)

    _, test_yhat, test_mask = run_split(test_idx, train_=False)
    test_metrics = evaluate_predictions(test_yhat, y, test_mask, age_days)
    val_metrics_final = evaluate_predictions(val_yhat, y, val_mask, age_days)

    print(f"[SWA-InVAR-v2] TEST ic={test_metrics['ic']:+.4f} "
          f"rank_ic={test_metrics['rank_ic']:+.4f} "
          f"ndcg10={test_metrics['ndcg10']:.4f} "
          f"ndcg50={test_metrics['ndcg50']:.4f}")

    out_path = save_result(
        out_dir=Path(cfg.output_dir),
        fold=cfg.fold, seed=cfg.seed,
        model_name="SWA-InVAR (v2 protocol)",
        test_metrics=test_metrics,
        val_metrics=val_metrics_final,
        test_y_hat=test_yhat,
        test_eval_mask=test_mask,
        history=history,
        config=asdict(cfg),
        n_panel=(T, N, Fdim),
        n_train=len(train_idx), n_val=len(val_idx), n_test=len(test_idx),
        y_true=y, tickers=tickers, dates=dates,
        age_days=age_days, tradable_mask=tradable,
    )
    print(f"[SWA-InVAR-v2] wrote {out_path}")


if __name__ == "__main__":
    main()
