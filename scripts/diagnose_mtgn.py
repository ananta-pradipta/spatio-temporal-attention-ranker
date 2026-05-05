"""Deep diagnostic: why cross-entity doesn't lift aggregate IC.

Runs a short instrumented training (fixed seed, small universe, few epochs)
and logs per-epoch diagnostics about the MTGN attention pathway:

  1. |h_temporal| vs |h_spatial| magnitude (is retrieval contributing?)
  2. Attention entropy per query (is attention distinguishing, or uniform?)
  3. Retrieval diversity (are top-K genuinely similar, or random?)
  4. Store health: per-entry memory variance, dead dimensions.
  5. Gradient norms on W_q, W_k, W_v (is retrieval getting gradients?)

Output: docs/mtgn_diagnostic.md with findings and numeric tables.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from src.mtgn.attention.temporal import EpisodicTemporalAttention, TemporalAttentionConfig
from src.mtgn.store.episodic_store import EpisodicStore, StoreConfig, StoredEntry
from src.mtgn.store.salience_gate import GatingConfig, SalienceGate
from src.mtgn.training.graph_builder import GraphConfig, build_correlation_edges
from src.mtgn.training.panel import FEATURE_COLS, PanelConfig, build_panel, panel_to_tensors
from src.mtgn.training.train import information_coefficient, listnet_loss, pinball_loss, rank_ic, temporal_split
from src.mtgn.training.train_mtgn import MTGNLite, MTGNTrainConfig, _retrieve_for_batch


def run_diagnostic(cfg: MTGNTrainConfig, mode: str = "cross_entity") -> dict:
    cfg.retrieval_mode = mode
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device("cpu")

    panel_cfg = PanelConfig(
        start_date=cfg.start_date,
        end_date=cfg.end_date,
        horizon_days=cfg.horizon_days,
        max_tickers=cfg.max_tickers,
    )
    panel, tickers, dates = build_panel(panel_cfg)
    tensors = panel_to_tensors(panel, tickers, dates)
    x = torch.from_numpy(tensors["x"]).to(device)
    y = torch.from_numpy(tensors["y"]).to(device)
    mask_all = torch.from_numpy(tensors["mask"]).to(device)
    T, N, F = x.shape

    train_slice, val_slice, test_slice = temporal_split(T, cfg.val_fraction, cfg.test_fraction)
    mu = x[train_slice].reshape(-1, F).mean(dim=0)
    sd = x[train_slice].reshape(-1, F).std(dim=0).clamp(min=1e-6)
    x = (x - mu) / sd

    head_arr = tensors["x"][train_slice.start : min(train_slice.start + 60, train_slice.stop)]
    edge_index_np, _ = build_correlation_edges(head_arr, GraphConfig())
    edge_index = torch.from_numpy(edge_index_np).to(device)

    model = MTGNLite(F, cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    gate = SalienceGate(GatingConfig())

    x_np_raw = tensors["x"]
    y_np = tensors["y"]
    mask_np_full = tensors["mask"]
    st_vol_col = FEATURE_COLS.index("st_volume_24h")
    log_ret_col = FEATURE_COLS.index("log_return")

    diag_rows = []
    store = EpisodicStore(StoreConfig(dim=cfg.hidden_dim))

    for epoch in range(cfg.epochs):
        model.train(True)
        store = EpisodicStore(StoreConfig(dim=cfg.hidden_dim))
        stats = {
            "epoch": epoch,
            "n_retrieval_days": 0,
            "h_spatial_norm_mean": [],
            "h_temporal_norm_mean": [],
            "temporal_over_spatial_ratio": [],
            "attention_entropy_mean": [],
            "attention_entropy_max": [],
            "valid_candidates_per_query": [],
            "retrieved_same_ticker_frac": [],
            "grad_norm_Wq": 0.0,
            "grad_norm_Wk": 0.0,
            "grad_norm_Wv": 0.0,
            "train_loss": 0.0,
        }
        losses = []

        for t in range(train_slice.start, train_slice.stop):
            m = mask_all[t]
            if m.sum() < 3:
                continue

            h_spatial = model.forward_spatial(x[t], edge_index)
            h_spatial_np = h_spatial.detach().cpu().numpy().astype(np.float32)
            h_spatial_norm = h_spatial.norm(dim=-1)

            if store.size > 0 and cfg.retrieval_mode != "none":
                mem_np, dt_np, mk_np = _retrieve_for_batch(
                    store, h_spatial_np,
                    ticker_ids=np.arange(N),
                    t_now=float(t),
                    k=cfg.retrieval_k,
                    mode=cfg.retrieval_mode,
                    hidden_dim=cfg.hidden_dim,
                )
                entries_memory = torch.from_numpy(mem_np).to(device)
                entries_dt = torch.from_numpy(dt_np).to(device)
                mk = torch.from_numpy(mk_np).to(device)

                # === Instrumentation ===
                # compute h_temporal only (bypass residual fusion for measurement)
                with torch.no_grad():
                    qp = model.temporal.W_q(h_spatial)
                    kp_inp = torch.cat(
                        [entries_memory, model.temporal.time_encoder(entries_dt.float())], dim=-1
                    )
                    kp = model.temporal.W_k(kp_inp)
                    vp = model.temporal.W_v(entries_memory)

                    Nq = h_spatial.shape[0]
                    H = cfg.attention_heads
                    d = cfg.hidden_dim // H
                    qh = qp.view(Nq, 1, H, d)
                    kh = kp.view(Nq, cfg.retrieval_k, H, d)
                    sc = (qh * kh).sum(-1) / math.sqrt(d)
                    mk_h = mk.unsqueeze(-1).expand(-1, -1, H)
                    sc = sc.masked_fill(~mk_h, float("-inf"))
                    no_valid = (~mk).all(dim=-1)
                    attn = torch.softmax(sc, dim=1)
                    attn = torch.where(
                        no_valid.unsqueeze(-1).unsqueeze(-1), torch.zeros_like(attn), attn
                    )

                    # entropy per query (averaged over heads and nodes with valid candidates)
                    eps = 1e-12
                    ent_per_q = -(attn * (attn.clamp(min=eps).log())).sum(dim=1).mean(dim=-1)  # [N]
                    valid_nodes = (~no_valid)
                    if valid_nodes.sum() > 0:
                        ent_mean = float(ent_per_q[valid_nodes].mean())
                        ent_max = float(ent_per_q[valid_nodes].max())
                    else:
                        ent_mean = ent_max = float("nan")

                z = model.fuse(h_spatial, entries_memory, entries_dt, mk)
                h_temporal_approx = z - torch.nn.functional.layer_norm(
                    h_spatial, h_spatial.shape[-1:]
                )
                h_temporal_norm = h_temporal_approx.norm(dim=-1)

                valid_nodes_cpu = valid_nodes.cpu().numpy()
                if valid_nodes_cpu.any():
                    ratio = (h_temporal_norm / (h_spatial_norm + 1e-8))[valid_nodes].detach().cpu().numpy()
                    stats["h_spatial_norm_mean"].append(float(h_spatial_norm[valid_nodes].mean()))
                    stats["h_temporal_norm_mean"].append(float(h_temporal_norm[valid_nodes].mean()))
                    stats["temporal_over_spatial_ratio"].append(float(ratio.mean()))
                    stats["attention_entropy_mean"].append(ent_mean)
                    stats["attention_entropy_max"].append(ent_max)
                    stats["valid_candidates_per_query"].append(float(mk.sum(dim=-1).float().mean()))
                    # Retrieval diversity: fraction of top-1 from same ticker
                    # Approximate via store lookup on a random sample of 5 query nodes
                    idxs = np.random.choice(
                        np.where(valid_nodes_cpu)[0], size=min(5, valid_nodes_cpu.sum()), replace=False
                    )
                    same = 0; tot = 0
                    for qi in idxs:
                        ents, _ = store.retrieve(
                            h_spatial_np[qi], k=1, t_max=float(t),
                            self_ticker_id=int(qi), mode=cfg.retrieval_mode,
                        )
                        for e in ents:
                            tot += 1
                            if e.ticker_id == int(qi):
                                same += 1
                    if tot > 0:
                        stats["retrieved_same_ticker_frac"].append(same / tot)
                    stats["n_retrieval_days"] += 1
            else:
                z = model.fuse(h_spatial, None, None, None)

            out = model.predict(z)
            y_hat = out["y_hat"]
            q_hat = out["q_hat"]

            l_rank = listnet_loss(y_hat.unsqueeze(0), y[t].unsqueeze(0), m.unsqueeze(0))
            l_risk = pinball_loss(y[t], q_hat, model.taus, m)
            loss = l_rank + cfg.quantile_weight * l_risk
            opt.zero_grad()
            loss.backward()

            if cfg.retrieval_mode != "none" and store.size > 0:
                if model.temporal.W_q.weight.grad is not None:
                    stats["grad_norm_Wq"] += float(model.temporal.W_q.weight.grad.norm())
                    stats["grad_norm_Wk"] += float(model.temporal.W_k.weight.grad.norm())
                    stats["grad_norm_Wv"] += float(model.temporal.W_v.weight.grad.norm())

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())

            # gate + store write
            mask_today = mask_np_full[t]
            for i in range(N):
                if not mask_today[i]:
                    continue
                ret_prior = x_np_raw[max(0, t - 30): t, i, log_ret_col]
                vol_prior = x_np_raw[max(0, t - 30): t, i, st_vol_col]
                res = gate.evaluate(
                    ticker_id=i, return_prior=ret_prior, st_volume_prior=vol_prior,
                    return_today=float(x_np_raw[t, i, log_ret_col]),
                    st_volume_today=float(x_np_raw[t, i, st_vol_col]),
                    catalyst_event_type=None, memory_delta=None, epoch=epoch,
                )
                if res.any:
                    store.write(StoredEntry(
                        ticker_id=i, time=float(t),
                        memory=h_spatial_np[i].copy(),
                        meta={"forward_return_h": float(y_np[t, i])},
                    ))

        # Aggregate epoch stats
        stats["train_loss"] = float(np.mean(losses)) if losses else float("nan")
        for key in [
            "h_spatial_norm_mean", "h_temporal_norm_mean", "temporal_over_spatial_ratio",
            "attention_entropy_mean", "attention_entropy_max", "valid_candidates_per_query",
            "retrieved_same_ticker_frac",
        ]:
            vals = stats[key]
            stats[key] = float(np.mean(vals)) if vals else float("nan")
        stats["store_final_size"] = store.size
        diag_rows.append(stats)

        print(
            f"[{epoch:02d}]  loss={stats['train_loss']:.4f}  "
            f"|h_s|={stats['h_spatial_norm_mean']:.3f}  "
            f"|h_t|={stats['h_temporal_norm_mean']:.3f}  "
            f"ratio={stats['temporal_over_spatial_ratio']:.3f}  "
            f"attn_entr={stats['attention_entropy_mean']:.3f}/{math.log(cfg.retrieval_k):.3f}(max)  "
            f"valid_K={stats['valid_candidates_per_query']:.1f}/{cfg.retrieval_k}  "
            f"same_tkr={stats['retrieved_same_ticker_frac']:.2f}  "
            f"|gradW|=q{stats['grad_norm_Wq']:.2f} k{stats['grad_norm_Wk']:.2f} v{stats['grad_norm_Wv']:.2f}  "
            f"store={stats['store_final_size']}"
        )

    return {"config": asdict(cfg), "mode": mode, "epochs": diag_rows}


def main() -> None:
    cfg = MTGNTrainConfig(
        start_date="2020-01-01", end_date="2022-12-31",
        max_tickers=100, epochs=6, retrieval_mode="cross_entity",
        retrieval_k=32, hidden_dim=128, attention_heads=4,
        quantile_weight=0.5, seed=42, early_stopping=False,
    )
    print(f"Running diagnostic: mode=cross_entity  max_tickers=100  epochs=6  seed=42")
    result = run_diagnostic(cfg, mode="cross_entity")
    out = Path("results/mtgn_diagnostic.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=str))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
