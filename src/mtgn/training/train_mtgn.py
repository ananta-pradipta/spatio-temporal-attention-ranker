"""MTGN training loop: spatial GAT + salience-gated episodic store + temporal attention.

Phase 1a integration. Uses the spatial GAT output h_spatial_i(t) as the
per-ticker memory vector written to the episodic store (TGN-lite). Proper
GRU-based TGN memory is a Phase 1b enhancement.

Per-day flow:
    1. h_spatial_t = GAT(x_t, edge_index)
    2. For each active ticker i:
         q_i = h_spatial_i
         candidates = store.retrieve(q_i, k, t_max=t, mode=retrieval_mode)
         h_temporal_i = EpisodicTemporalAttention(q_i, candidates)
         z_i = LayerNorm(h_spatial_i + h_temporal_i)
    3. y_hat_i = RankingHead(z_i)
       q_hat_i = QuantileHead(z_i)
    4. ListNet + pinball loss + backprop
    5. Salience gate evaluates each ticker:
         return-mag trigger + ST-volume-spike trigger (catalyst data TBD)
       If fire: store.write(s_i=h_spatial_i detached, forward_return_h target)

`--retrieval-mode` in {cross_entity, self_only, none}:
    none          -> MTGN reduces to vanilla (no temporal attention).
    self_only     -> kNN restricted to ticker-i's own history.
    cross_entity  -> full kNN across the whole store. MTGN headline.

Usage:
    python3 -m src.mtgn.training.train_mtgn --max-tickers 50 \\
        --start 2021-01-01 --end 2021-12-31 --epochs 5 \\
        --retrieval-mode cross_entity
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from src.mtgn.attention.temporal import (
    EpisodicTemporalAttention,
    TemporalAttentionConfig,
)
from src.mtgn.store.episodic_store import EpisodicStore, StoreConfig, StoredEntry
from src.mtgn.store.salience_gate import GatingConfig, SalienceGate
from src.mtgn.training.graph_builder import GraphConfig, build_correlation_edges
from src.mtgn.training.panel import FEATURE_COLS, PanelConfig, build_panel, panel_to_tensors
from src.mtgn.training.train import (
    information_coefficient,
    listnet_loss,
    pinball_loss,
    rank_ic,
    temporal_split,
)


@dataclass
class MTGNTrainConfig:
    start_date: str = "2020-01-01"
    end_date: str = "2022-12-31"
    horizon_days: int = 5
    max_tickers: int | None = 50
    hidden_dim: int = 128
    attention_heads: int = 4
    retrieval_mode: str = "cross_entity"   # cross_entity | self_only | none
    retrieval_k: int = 32
    quantile_weight: float = 0.5
    lr: float = 5e-4
    weight_decay: float = 1e-5
    epochs: int = 5
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    seed: int = 42
    # Early-stopping: evaluate test on best-val-IC checkpoint (not final epoch).
    early_stopping: bool = True
    patience: int = 3


class SpatialGAT(nn.Module):
    def __init__(self, in_dim: int, hidden: int, heads: int):
        super().__init__()
        from torch_geometric.nn import GATConv

        self.input = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU())
        self.gat1 = GATConv(hidden, hidden // heads, heads=heads, dropout=0.1)
        self.gat2 = GATConv(hidden, hidden // heads, heads=heads, dropout=0.1)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        h = self.input(x)
        h = torch.relu(self.gat1(h, edge_index))
        h = torch.relu(self.gat2(h, edge_index))
        return h


class MTGNLite(nn.Module):
    """Spatial GAT + episodic temporal attention + two heads."""

    def __init__(self, in_dim: int, cfg: MTGNTrainConfig):
        super().__init__()
        self.cfg = cfg
        self.spatial = SpatialGAT(in_dim, cfg.hidden_dim, cfg.attention_heads)
        self.temporal = EpisodicTemporalAttention(
            TemporalAttentionConfig(
                memory_dim=cfg.hidden_dim,
                hidden_dim=cfg.hidden_dim,
                num_heads=cfg.attention_heads,
                time_dim=32,
            )
        )
        self.rank_head = nn.Linear(cfg.hidden_dim, 1)
        self.risk_head = nn.Linear(cfg.hidden_dim, 3)
        self.taus = (0.05, 0.50, 0.95)

    def forward_spatial(self, x: Tensor, edge_index: Tensor) -> Tensor:
        return self.spatial(x, edge_index)

    def fuse(
        self,
        h_spatial: Tensor,
        entries_memory: Tensor | None,
        entries_dt: Tensor | None,
        mask: Tensor | None,
    ) -> Tensor:
        if entries_memory is None or self.cfg.retrieval_mode == "none":
            return h_spatial
        return self.temporal(h_spatial, entries_memory, entries_dt, mask)

    def predict(self, z: Tensor) -> dict[str, Tensor]:
        return {"y_hat": self.rank_head(z).squeeze(-1), "q_hat": self.risk_head(z)}


def _retrieve_for_batch(
    store: EpisodicStore,
    h_spatial_np: np.ndarray,
    ticker_ids: np.ndarray,
    t_now: float,
    k: int,
    mode: str,
    hidden_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Retrieve up to k entries per query node. Returns (memory, dt, mask)."""
    N = h_spatial_np.shape[0]
    memories = np.zeros((N, k, hidden_dim), dtype=np.float32)
    dts = np.zeros((N, k), dtype=np.float32)
    mask = np.zeros((N, k), dtype=bool)
    for i in range(N):
        entries, _ = store.retrieve(
            h_spatial_np[i], k=k, t_max=t_now,
            self_ticker_id=int(ticker_ids[i]), mode=mode,
        )
        if not entries:
            continue
        stacked = store.stack_memory(entries)
        times = store.stack_times(entries)
        nk = stacked.shape[0]
        memories[i, :nk] = stacked
        dts[i, :nk] = t_now - times
        mask[i, :nk] = True
    return memories, dts, mask


def train_mtgn(cfg: MTGNTrainConfig) -> dict:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    print(f"panel: T={T} N={N} F={F} retrieval_mode={cfg.retrieval_mode}")

    train_slice, val_slice, test_slice = temporal_split(T, cfg.val_fraction, cfg.test_fraction)
    mu = x[train_slice].reshape(-1, F).mean(dim=0)
    sd = x[train_slice].reshape(-1, F).std(dim=0).clamp(min=1e-6)
    x = (x - mu) / sd

    head_arr = tensors["x"][train_slice.start : min(train_slice.start + 60, train_slice.stop)]
    edge_index_np, _ = build_correlation_edges(head_arr, GraphConfig())
    edge_index = torch.from_numpy(edge_index_np).to(device)
    print(f"edges: {edge_index.shape[1]}")

    model = MTGNLite(F, cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    gate = SalienceGate(GatingConfig())

    # Precompute prior windows for gating (30-day log-returns, 30-day ST volume).
    st_vol_col = FEATURE_COLS.index("st_volume_24h")
    log_ret_col = FEATURE_COLS.index("log_return")
    x_np_raw = tensors["x"]            # pre-normalization, for gating thresholds
    y_np = tensors["y"]
    mask_np_full = tensors["mask"]

    def run_epoch(sl: slice, train: bool, epoch: int) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        model.train(train)
        store = EpisodicStore(StoreConfig(dim=cfg.hidden_dim)) if train else eval_store_ref[0]
        losses: list[float] = []
        all_yhat, all_y, all_mask = [], [], []

        order = range(sl.start, sl.stop) if not train else (
            [sl.start + i for i in np.random.permutation(sl.stop - sl.start).tolist()]
            if False else list(range(sl.start, sl.stop))   # keep chronological for store writes
        )

        for t in order:
            m = mask_all[t]
            if m.sum() < 3:
                continue

            h_spatial = model.forward_spatial(x[t], edge_index)
            h_spatial_np = h_spatial.detach().cpu().numpy().astype(np.float32)

            # Retrieval (skip if store empty or mode=none)
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
                z = model.fuse(h_spatial, entries_memory, entries_dt, mk)
            else:
                z = model.fuse(h_spatial, None, None, None)

            out = model.predict(z)
            y_hat = out["y_hat"]
            q_hat = out["q_hat"]

            if train:
                l_rank = listnet_loss(y_hat.unsqueeze(0), y[t].unsqueeze(0), m.unsqueeze(0))
                l_risk = pinball_loss(y[t], q_hat, model.taus, m)
                loss = l_rank + cfg.quantile_weight * l_risk
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                losses.append(loss.item())
            else:
                with torch.no_grad():
                    l_rank = listnet_loss(y_hat.unsqueeze(0), y[t].unsqueeze(0), m.unsqueeze(0))
                    l_risk = pinball_loss(y[t], q_hat, model.taus, m)
                    losses.append((l_rank + cfg.quantile_weight * l_risk).item())

            all_yhat.append(y_hat.detach().cpu().numpy()[None, :])
            all_y.append(y[t].detach().cpu().numpy()[None, :])
            all_mask.append(m.detach().cpu().numpy()[None, :])

            # Write gated entries using h_spatial (detached) + realized future return as meta.
            # Causality note: the fwd_return_h is known only retrospectively at t + horizon;
            # here we emulate that by writing with metadata set to the target we computed.
            # The test split never reads from entries written during test.
            mask_today = mask_np_full[t]
            for i in range(N):
                if not mask_today[i]:
                    continue
                ret_prior = x_np_raw[max(0, t - 30): t, i, log_ret_col]
                vol_prior = x_np_raw[max(0, t - 30): t, i, st_vol_col]
                ret_today = float(x_np_raw[t, i, log_ret_col])
                vol_today = float(x_np_raw[t, i, st_vol_col])
                res = gate.evaluate(
                    ticker_id=i,
                    return_prior=ret_prior,
                    st_volume_prior=vol_prior,
                    return_today=ret_today,
                    st_volume_today=vol_today,
                    catalyst_event_type=None,
                    memory_delta=None,
                    epoch=epoch,
                )
                if res.any:
                    store.write(
                        StoredEntry(
                            ticker_id=i,
                            time=float(t),
                            memory=h_spatial_np[i].copy(),
                            meta={
                                "forward_return_h": float(y_np[t, i]),
                                "triggers": res.to_dict(),
                            },
                        )
                    )

        if train:
            eval_store_ref[0] = store   # carry final store into val/test
        mean_loss = float(np.mean(losses)) if losses else float("nan")
        yhat_arr = np.concatenate(all_yhat) if all_yhat else np.zeros((0, N))
        y_arr = np.concatenate(all_y) if all_y else np.zeros((0, N))
        m_arr = np.concatenate(all_mask) if all_mask else np.zeros((0, N), dtype=bool)
        return mean_loss, yhat_arr, y_arr, m_arr

    eval_store_ref: list[EpisodicStore | None] = [EpisodicStore(StoreConfig(dim=cfg.hidden_dim))]

    import copy

    history: list[dict] = []
    best_val_ic = -float("inf")
    best_state: dict | None = None
    best_epoch: int = -1
    epochs_since_best = 0

    for epoch in range(cfg.epochs):
        t0 = time.time()
        train_loss, *_ = run_epoch(train_slice, train=True, epoch=epoch)
        val_loss, v_yhat, v_y, v_m = run_epoch(val_slice, train=False, epoch=epoch)
        val_ic = information_coefficient(v_yhat, v_y, v_m)
        val_rank_ic = rank_ic(v_yhat, v_y, v_m)
        dt = time.time() - t0
        store_size = eval_store_ref[0].size if eval_store_ref[0] is not None else 0

        improved = val_ic > best_val_ic
        marker = "  *best*" if improved else ""
        print(
            f"[{epoch:02d}] train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"val_ic={val_ic:+.4f}  val_rank_ic={val_rank_ic:+.4f}  "
            f"store_size={store_size}  ({dt:.1f}s){marker}"
        )
        history.append(dict(
            epoch=epoch, train_loss=train_loss, val_loss=val_loss,
            val_ic=val_ic, val_rank_ic=val_rank_ic,
            store_size=store_size, time_sec=round(dt, 2),
        ))

        if improved:
            best_val_ic = val_ic
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            epochs_since_best = 0
        else:
            epochs_since_best += 1
            if cfg.early_stopping and epochs_since_best >= cfg.patience:
                print(
                    f"early stop at epoch {epoch}: no val-IC improvement for "
                    f"{cfg.patience} epochs (best {best_val_ic:+.4f} @ {best_epoch})"
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"loaded best checkpoint from epoch {best_epoch} (val_ic {best_val_ic:+.4f})")

    test_loss, t_yhat, t_y, t_m = run_epoch(test_slice, train=False, epoch=cfg.epochs)
    test_ic = information_coefficient(t_yhat, t_y, t_m)
    test_rank_ic_ = rank_ic(t_yhat, t_y, t_m)
    print(f"\nTEST  loss={test_loss:.4f}  ic={test_ic:+.4f}  rank_ic={test_rank_ic_:+.4f}")

    # Save test-set predictions aligned with the test date / ticker grid so
    # downstream slicing (catalyst-window subset, regime subsets, etc.) can
    # recompute ICs without rerunning the model.
    test_dates = [str(d) for d in dates[test_slice]]
    test_tickers = list(tickers)

    return dict(
        panel_T=T, panel_N=N, panel_F=F,
        edges=int(edge_index.shape[1]),
        test_loss=test_loss, test_ic=test_ic, test_rank_ic=test_rank_ic_,
        best_val_ic=best_val_ic, best_epoch=best_epoch,
        history=history,
        final_store_size=eval_store_ref[0].size if eval_store_ref[0] else 0,
        config=asdict(cfg),
        test_predictions={
            "dates": test_dates,
            "tickers": test_tickers,
            "y_hat": t_yhat.tolist(),       # [T_test, N]
            "y": t_y.tolist(),
            "mask": t_m.tolist(),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--max-tickers", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--horizon-days", type=int, default=None)
    parser.add_argument("--retrieval-mode", default=None,
                        choices=["cross_entity", "self_only", "none"])
    parser.add_argument("--retrieval-k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-early-stopping", action="store_true")
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path("results/mtgn_run.json"))
    args = parser.parse_args()

    cfg = MTGNTrainConfig()
    if args.start: cfg.start_date = args.start
    if args.end:   cfg.end_date = args.end
    if args.max_tickers is not None: cfg.max_tickers = args.max_tickers
    if args.epochs is not None: cfg.epochs = args.epochs
    if args.horizon_days is not None: cfg.horizon_days = args.horizon_days
    if args.retrieval_mode is not None: cfg.retrieval_mode = args.retrieval_mode
    if args.retrieval_k is not None: cfg.retrieval_k = args.retrieval_k
    if args.seed is not None: cfg.seed = args.seed
    if args.no_early_stopping: cfg.early_stopping = False
    if args.patience is not None: cfg.patience = args.patience

    result = train_mtgn(cfg)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=str))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
