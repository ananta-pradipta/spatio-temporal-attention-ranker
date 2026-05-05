"""Ridge and LSTM baselines on the enriched 5-year panel.

Both run under the same window / feature set / train-val-test split /
evaluation metrics so their results are directly comparable. Ridge
establishes the linear floor; LSTM is the temporal-only-no-graph
deep baseline. Neither has graph aggregation, memory, or retrieval.

Usage:
    python3 -m src.mtgn.training.train_baselines --model ridge --seed 11
    python3 -m src.mtgn.training.train_baselines --model lstm  --seed 11
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

from src.mtgn.training.panel_enriched import (
    FEATURE_COLS, EnrichedPanelConfig, build_enriched_panel, panel_to_tensors,
)
from src.mtgn.training.train import (
    information_coefficient, pinball_loss, rank_ic, ranknet_loss, temporal_split,
)


@dataclass
class BaselineConfig:
    start_date: str = "2018-01-01"
    end_date: str = "2022-12-31"
    horizon_days: int = 5
    max_tickers: int | None = None
    model: str = "ridge"        # ridge | lstm | gcn | gat | rgcn | tgcn
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    window: int = 20
    quantile_weight: float = 0.5
    lr: float = 5e-4
    weight_decay: float = 1e-5
    epochs: int = 30
    val_fraction: float = 0.125
    test_fraction: float = 0.20
    seed: int = 42
    patience: int = 5


class TickerLSTM(nn.Module):
    def __init__(self, feature_dim: int, hidden: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(feature_dim, hidden, num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.rank_head = nn.Linear(hidden, 1)
        self.risk_head = nn.Linear(hidden, 3)
        self.taus = (0.05, 0.50, 0.95)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        # x: [batch, window, F]
        _, (h_n, _) = self.lstm(x)
        h = h_n[-1]
        return {"y_hat": self.rank_head(h).squeeze(-1), "q_hat": self.risk_head(h)}


def _eval_daily_ic(yhat: np.ndarray, y: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    return information_coefficient(yhat, y, mask), rank_ic(yhat, y, mask)


def run_ridge(cfg: BaselineConfig, x: np.ndarray, y: np.ndarray, mask: np.ndarray, slices) -> dict:
    from sklearn.linear_model import Ridge
    T, N, F = x.shape
    train_sl, val_sl, test_sl = slices
    Xtr = x[train_sl].reshape(-1, F); ytr = y[train_sl].reshape(-1); mtr = mask[train_sl].reshape(-1)
    mu = Xtr[mtr].mean(axis=0); sd = Xtr[mtr].std(axis=0).clip(min=1e-6)
    Xtr_n = (Xtr - mu) / sd

    # Simple alpha grid, select by val IC
    best_alpha, best_val_ic = None, -float("inf")
    best_model = None
    for alpha in [0.1, 1.0, 10.0, 100.0]:
        m = Ridge(alpha=alpha).fit(Xtr_n[mtr], ytr[mtr])
        yhat_all = m.predict(((x.reshape(-1, F) - mu) / sd)).reshape(T, N)
        v_ic, _ = _eval_daily_ic(yhat_all[val_sl], y[val_sl], mask[val_sl])
        if v_ic > best_val_ic:
            best_val_ic = v_ic; best_alpha = alpha; best_model = m

    yhat_all = best_model.predict(((x.reshape(-1, F) - mu) / sd)).reshape(T, N)
    test_ic, test_rank = _eval_daily_ic(yhat_all[test_sl], y[test_sl], mask[test_sl])
    val_ic, val_rank = _eval_daily_ic(yhat_all[val_sl], y[val_sl], mask[val_sl])
    return {
        "model": "ridge",
        "best_alpha": best_alpha,
        "val_ic": val_ic, "val_rank_ic": val_rank,
        "test_ic": test_ic, "test_rank_ic": test_rank,
        "_test_preds": yhat_all[test_sl].astype(np.float32),
        "_test_y": y[test_sl].astype(np.float32),
        "_test_mask": mask[test_sl].astype(bool),
    }


def run_lstm(cfg: BaselineConfig, x: np.ndarray, y: np.ndarray, mask: np.ndarray, slices) -> dict:
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    device = torch.device("cpu")
    T, N, F = x.shape
    train_sl, val_sl, test_sl = slices
    mu = x[train_sl].reshape(-1, F).mean(axis=0)
    sd = x[train_sl].reshape(-1, F).std(axis=0).clip(min=1e-6)
    xn = (x - mu) / sd

    model = TickerLSTM(F, cfg.hidden_dim, cfg.num_layers, cfg.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    def build_day_windows(t: int):
        # Produce [N_active, window, F] for day t. Masks out tickers where the window has NaN or insufficient history.
        W = cfg.window
        if t < W:
            return None, None, None
        xw = xn[t - W: t]       # [W, N, F]
        xw = np.transpose(xw, (1, 0, 2))  # [N, W, F]
        m = mask[t] & mask[t - W]
        if m.sum() < 3:
            return None, None, None
        return torch.from_numpy(xw[m]).to(device), torch.from_numpy(y[t, m]).to(device), np.where(m)[0]

    def run_epoch(sl: slice, train: bool) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        model.train(train)
        preds_full = np.zeros((sl.stop - sl.start, N), dtype=np.float32)
        y_full = np.zeros((sl.stop - sl.start, N), dtype=np.float32)
        m_full = np.zeros((sl.stop - sl.start, N), dtype=bool)
        losses = []
        for i, t in enumerate(range(sl.start, sl.stop)):
            xw, yt, idx = build_day_windows(t)
            if xw is None:
                continue
            out = model(xw)
            yhat = out["y_hat"]
            qhat = out["q_hat"]
            if train:
                lr_t = ranknet_loss(yhat.unsqueeze(0), yt.unsqueeze(0),
                                    torch.ones_like(yt, dtype=torch.bool).unsqueeze(0))
                lq_t = pinball_loss(yt, qhat, model.taus, torch.ones_like(yt, dtype=torch.bool))
                l = lr_t + cfg.quantile_weight * lq_t
                opt.zero_grad(); l.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
                losses.append(l.item())
            preds_full[i, idx] = yhat.detach().cpu().numpy()
            y_full[i, idx] = yt.cpu().numpy()
            m_full[i, idx] = True
        return preds_full, y_full, m_full, (float(np.mean(losses)) if losses else float("nan"))

    best_val = -float("inf"); best_val_rank = float("nan"); best_state = None; un = 0
    for epoch in range(cfg.epochs):
        _p, _y, _m, tl = run_epoch(train_sl, train=True)
        vp, vy, vm, _ = run_epoch(val_sl, train=False)
        v_ic = information_coefficient(vp, vy, vm)
        v_rank = rank_ic(vp, vy, vm)
        if v_ic > best_val:
            best_val = v_ic; best_val_rank = v_rank; un = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            un += 1
            if un >= cfg.patience:
                break
        print(f"[{epoch:02d}] train_loss={tl:.4f}  val_ic={v_ic:+.4f}" + ("  *best*" if un == 0 else ""))

    if best_state is not None:
        model.load_state_dict(best_state)
    tp, ty, tm, _ = run_epoch(test_sl, train=False)
    test_ic = information_coefficient(tp, ty, tm)
    test_rank = rank_ic(tp, ty, tm)
    return {
        "model": "lstm",
        "best_val_ic": best_val,
        "best_val_rank_ic": best_val_rank,
        "test_ic": test_ic, "test_rank_ic": test_rank,
        "_test_preds": tp, "_test_y": ty, "_test_mask": tm,
    }


class DailyGNN(nn.Module):
    """Vanilla GNN over a static correlation graph: graph-only, no temporal memory.

    Takes daily cross-sectional features [N, F] plus a fixed edge_index
    and predicts fwd_return for every ticker on that day. This is the
    "graph, no time" rung below LSTM ("time, no graph") and below the
    full MTGN which combines both.
    """

    def __init__(self, feature_dim: int, hidden: int, layer_type: str = "gcn",
                 heads: int = 4, dropout: float = 0.2):
        super().__init__()
        from torch_geometric.nn import GCNConv, GATConv
        self.input = nn.Sequential(nn.Linear(feature_dim, hidden), nn.ReLU())
        if layer_type == "gat":
            self.g1 = GATConv(hidden, hidden // heads, heads=heads, dropout=dropout)
            self.g2 = GATConv(hidden, hidden // heads, heads=heads, dropout=dropout)
        else:
            self.g1 = GCNConv(hidden, hidden)
            self.g2 = GCNConv(hidden, hidden)
        self.drop = nn.Dropout(dropout)
        self.rank_head = nn.Linear(hidden, 1)
        self.risk_head = nn.Linear(hidden, 3)
        self.taus = (0.05, 0.50, 0.95)

    def forward(self, x: Tensor, edge_index: Tensor) -> dict[str, Tensor]:
        h = self.input(x)
        h = torch.relu(self.g1(h, edge_index))
        h = self.drop(h)
        h = torch.relu(self.g2(h, edge_index))
        return {"y_hat": self.rank_head(h).squeeze(-1), "q_hat": self.risk_head(h)}


def run_gnn(cfg: BaselineConfig, x: np.ndarray, y: np.ndarray, mask: np.ndarray, slices) -> dict:
    from src.mtgn.training.graph_builder import GraphConfig, build_correlation_edges

    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T, N, F = x.shape
    train_sl, val_sl, test_sl = slices
    mu = x[train_sl].reshape(-1, F).mean(axis=0)
    sd = x[train_sl].reshape(-1, F).std(axis=0).clip(min=1e-6)
    xn = ((x - mu) / sd).astype(np.float32)

    # Static correlation graph from first 60 train days (matches SimpleMTGN convention)
    head_arr = x[train_sl.start : min(train_sl.start + 60, train_sl.stop)]
    edge_index_np, _ = build_correlation_edges(head_arr, GraphConfig())
    edge_index = torch.from_numpy(edge_index_np).long().to(device)
    print(f"graph: {edge_index.shape[1]} edges over {N} nodes "
          f"(mean deg {edge_index.shape[1]/max(N,1):.1f})")

    model = DailyGNN(F, cfg.hidden_dim, layer_type=cfg.model,
                     dropout=cfg.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    def run_epoch(sl: slice, train: bool):
        model.train(train)
        preds_full = np.zeros((sl.stop - sl.start, N), dtype=np.float32)
        y_full = np.zeros((sl.stop - sl.start, N), dtype=np.float32)
        m_full = np.zeros((sl.stop - sl.start, N), dtype=bool)
        losses = []
        for i, t in enumerate(range(sl.start, sl.stop)):
            m_t = mask[t]
            if m_t.sum() < 3:
                continue
            # Zero inactive nodes' features BEFORE the GNN, otherwise their
            # normalized-zero feature vectors propagate as messages to active
            # nodes through the static correlation edges and pollute the active
            # nodes' representations. Loss-side masking alone is insufficient.
            xt_np = xn[t] * m_t[:, None].astype(np.float32)
            xt = torch.from_numpy(xt_np).to(device)         # [N, F]
            yt = torch.from_numpy(y[t]).to(device)          # [N]
            m_ten = torch.from_numpy(m_t).to(device)
            out = model(xt, edge_index)
            yhat = out["y_hat"]; qhat = out["q_hat"]
            if train:
                lr_t = ranknet_loss(yhat[m_ten].unsqueeze(0),
                                    yt[m_ten].unsqueeze(0),
                                    torch.ones_like(yt[m_ten], dtype=torch.bool).unsqueeze(0))
                lq_t = pinball_loss(yt[m_ten], qhat[m_ten], model.taus,
                                    torch.ones_like(yt[m_ten], dtype=torch.bool))
                l = lr_t + cfg.quantile_weight * lq_t
                opt.zero_grad(); l.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
                losses.append(l.item())
            preds_full[i] = yhat.detach().cpu().numpy()
            y_full[i] = yt.cpu().numpy()
            m_full[i] = m_t
        return preds_full, y_full, m_full, (float(np.mean(losses)) if losses else float("nan"))

    best_val = -float("inf"); best_val_rank = float("nan"); best_state = None; un = 0
    for epoch in range(cfg.epochs):
        _p, _y, _m, tl = run_epoch(train_sl, train=True)
        vp, vy, vm, _ = run_epoch(val_sl, train=False)
        v_ic = information_coefficient(vp, vy, vm)
        v_rank = rank_ic(vp, vy, vm)
        if v_ic > best_val:
            best_val = v_ic; best_val_rank = v_rank; un = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            un += 1
            if un >= cfg.patience:
                break
        print(f"[{epoch:02d}] train_loss={tl:.4f}  val_ic={v_ic:+.4f}" + ("  *best*" if un == 0 else ""))

    if best_state is not None:
        model.load_state_dict(best_state)
    tp, ty, tm, _ = run_epoch(test_sl, train=False)
    test_ic = information_coefficient(tp, ty, tm)
    test_rank = rank_ic(tp, ty, tm)
    return {
        "model": cfg.model,
        "best_val_ic": best_val,
        "best_val_rank_ic": best_val_rank,
        "test_ic": test_ic, "test_rank_ic": test_rank,
        "graph_edges": int(edge_index.shape[1]),
        "_test_preds": tp, "_test_y": ty, "_test_mask": tm,
    }


class DailyRGNN(nn.Module):
    """Heterogeneous GNN over 3 edge types (trial, comention, sector).

    Uses RGCNConv: a separate learnable weight matrix per relation, then
    sums the per-relation messages. This is the minimal extension of
    vanilla GCN to multi-relation graphs. No attention, no temporal.
    """

    def __init__(self, feature_dim: int, hidden: int, num_relations: int,
                 dropout: float = 0.2, num_bases: int | None = None):
        super().__init__()
        from torch_geometric.nn import RGCNConv
        self.input = nn.Sequential(nn.Linear(feature_dim, hidden), nn.ReLU())
        self.g1 = RGCNConv(hidden, hidden, num_relations=num_relations,
                           num_bases=num_bases)
        self.g2 = RGCNConv(hidden, hidden, num_relations=num_relations,
                           num_bases=num_bases)
        self.drop = nn.Dropout(dropout)
        self.rank_head = nn.Linear(hidden, 1)
        self.risk_head = nn.Linear(hidden, 3)
        self.taus = (0.05, 0.50, 0.95)

    def forward(self, x: Tensor, edge_index: Tensor, edge_type: Tensor) -> dict[str, Tensor]:
        h = self.input(x)
        h = torch.relu(self.g1(h, edge_index, edge_type))
        h = self.drop(h)
        h = torch.relu(self.g2(h, edge_index, edge_type))
        return {"y_hat": self.rank_head(h).squeeze(-1), "q_hat": self.risk_head(h)}


def run_rgcn(cfg: BaselineConfig, x: np.ndarray, y: np.ndarray, mask: np.ndarray,
             slices, tickers: list[str], dates=None) -> dict:
    from src.mtgn.graph.edges import EdgeBuildConfig, build_mechanistic_edges_per_relation
    from src.mtgn.training.graph_builder import GraphConfig, build_correlation_edges

    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T, N, F = x.shape
    train_sl, val_sl, test_sl = slices
    mu = x[train_sl].reshape(-1, F).mean(axis=0)
    sd = x[train_sl].reshape(-1, F).std(axis=0).clip(min=1e-6)
    xn = ((x - mu) / sd).astype(np.float32)

    # Build per-relation edges on the current fold's TRAIN slice only (strict, no
    # val/test peek). If dates are available, use the train slice's last date;
    # otherwise fall back to the conservative 2021-03-31 cut.
    edge_train_end = (
        str(pd.Timestamp(dates[train_sl.stop - 1]).date())
        if dates is not None and train_sl.stop > train_sl.start else "2021-03-31"
    )
    rel_edges = build_mechanistic_edges_per_relation(
        tickers,
        EdgeBuildConfig(train_start=cfg.start_date, train_end=edge_train_end),
        require_nonempty=False,
    )
    # 4th relation: price correlation, tightened vs the vanilla-GCN baseline.
    # Vanilla used 60d / |corr|>=0.3 / top-30, which at N=60 only barely clears
    # the Pearson null band (+/-0.25 at 95%) and produced ~4358 edges that
    # heavily overlapped with sector. Here we use 250 trading days (~1y),
    # |corr|>=0.5, top-15 neighbors so the correlation relation is a clean,
    # sparser complement to sector.
    corr_win = min(250, train_sl.stop - train_sl.start)
    corr_head = x[train_sl.start : train_sl.start + corr_win]
    corr_ei, _ = build_correlation_edges(
        corr_head,
        GraphConfig(correlation_window_days=corr_win,
                    correlation_threshold=0.5,
                    max_degree=15),
    )
    rel_edges["correlation"] = (corr_ei, np.ones(corr_ei.shape[1], dtype=np.float32))

    rel_names = ["trial", "comention", "sector", "correlation"]
    ei_list, et_list = [], []
    for rel_idx, name in enumerate(rel_names):
        ei, _ = rel_edges[name]
        if ei.shape[1] == 0:
            continue
        ei_list.append(ei)
        et_list.append(np.full(ei.shape[1], rel_idx, dtype=np.int64))
    edge_index = torch.from_numpy(np.concatenate(ei_list, axis=1)).long().to(device)
    edge_type  = torch.from_numpy(np.concatenate(et_list)).long().to(device)
    print(f"graph: {edge_index.shape[1]} edges over {N} nodes across "
          f"{len(rel_names)} relations")
    for rel_idx, name in enumerate(rel_names):
        cnt = int((edge_type == rel_idx).sum().item())
        print(f"  {name:10s}: {cnt} edges")

    model = DailyRGNN(F, cfg.hidden_dim, num_relations=len(rel_names),
                      dropout=cfg.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    def run_epoch(sl: slice, train: bool):
        model.train(train)
        preds_full = np.zeros((sl.stop - sl.start, N), dtype=np.float32)
        y_full = np.zeros((sl.stop - sl.start, N), dtype=np.float32)
        m_full = np.zeros((sl.stop - sl.start, N), dtype=bool)
        losses = []
        for i, t in enumerate(range(sl.start, sl.stop)):
            m_t = mask[t]
            if m_t.sum() < 3:
                continue
            xt_np = xn[t] * m_t[:, None].astype(np.float32)
            xt = torch.from_numpy(xt_np).to(device)
            yt = torch.from_numpy(y[t]).to(device)
            m_ten = torch.from_numpy(m_t).to(device)
            out = model(xt, edge_index, edge_type)
            yhat = out["y_hat"]; qhat = out["q_hat"]
            if train:
                lr_t = ranknet_loss(yhat[m_ten].unsqueeze(0),
                                    yt[m_ten].unsqueeze(0),
                                    torch.ones_like(yt[m_ten], dtype=torch.bool).unsqueeze(0))
                lq_t = pinball_loss(yt[m_ten], qhat[m_ten], model.taus,
                                    torch.ones_like(yt[m_ten], dtype=torch.bool))
                l = lr_t + cfg.quantile_weight * lq_t
                opt.zero_grad(); l.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
                losses.append(l.item())
            preds_full[i] = yhat.detach().cpu().numpy()
            y_full[i] = yt.cpu().numpy()
            m_full[i] = m_t
        return preds_full, y_full, m_full, (float(np.mean(losses)) if losses else float("nan"))

    best_val = -float("inf"); best_val_rank = float("nan"); best_state = None; un = 0
    for epoch in range(cfg.epochs):
        _p, _y, _m, tl = run_epoch(train_sl, train=True)
        vp, vy, vm, _ = run_epoch(val_sl, train=False)
        v_ic = information_coefficient(vp, vy, vm)
        v_rank = rank_ic(vp, vy, vm)
        if v_ic > best_val:
            best_val = v_ic; best_val_rank = v_rank; un = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            un += 1
            if un >= cfg.patience:
                break
        print(f"[{epoch:02d}] train_loss={tl:.4f}  val_ic={v_ic:+.4f}" + ("  *best*" if un == 0 else ""))

    if best_state is not None:
        model.load_state_dict(best_state)
    tp, ty, tm, _ = run_epoch(test_sl, train=False)
    return {
        "model": "rgcn",
        "best_val_ic": best_val,
        "best_val_rank_ic": best_val_rank,
        "test_ic": information_coefficient(tp, ty, tm),
        "test_rank_ic": rank_ic(tp, ty, tm),
        "graph_edges": int(edge_index.shape[1]),
        "graph_relations": len(rel_names),
        "edges_per_relation": {
            name: int((edge_type == i).sum().item()) for i, name in enumerate(rel_names)
        },
        "_test_preds": tp, "_test_y": ty, "_test_mask": tm,
    }


class TemporalGCN(nn.Module):
    """GCN then LSTM: per-day graph convolution produces graph-conditioned node
    embeddings; a per-ticker LSTM aggregates those embeddings over a W-day
    window to predict the forward return. This is the "graph + time" rung
    between vanilla GCN (graph only) and LSTM (time only).
    """

    def __init__(self, feature_dim: int, hidden: int, num_layers: int, dropout: float):
        super().__init__()
        from torch_geometric.nn import GCNConv
        self.input = nn.Sequential(nn.Linear(feature_dim, hidden), nn.ReLU())
        self.g1 = GCNConv(hidden, hidden)
        self.g2 = GCNConv(hidden, hidden)
        self.gdrop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(hidden, hidden, num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.rank_head = nn.Linear(hidden, 1)
        self.risk_head = nn.Linear(hidden, 3)
        self.taus = (0.05, 0.50, 0.95)

    def graph_embed(self, x: Tensor, edge_index: Tensor) -> Tensor:
        h = self.input(x)
        h = torch.relu(self.g1(h, edge_index))
        h = self.gdrop(h)
        h = torch.relu(self.g2(h, edge_index))
        return h  # [N, hidden]

    def temporal_head(self, seq: Tensor) -> dict[str, Tensor]:
        # seq: [batch, W, hidden]
        _, (h_n, _) = self.lstm(seq)
        h = h_n[-1]
        return {"y_hat": self.rank_head(h).squeeze(-1), "q_hat": self.risk_head(h)}


def run_tgcn(cfg: BaselineConfig, x: np.ndarray, y: np.ndarray, mask: np.ndarray, slices) -> dict:
    from src.mtgn.training.graph_builder import GraphConfig, build_correlation_edges

    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T, N, F = x.shape
    train_sl, val_sl, test_sl = slices
    mu = x[train_sl].reshape(-1, F).mean(axis=0)
    sd = x[train_sl].reshape(-1, F).std(axis=0).clip(min=1e-6)
    xn = ((x - mu) / sd).astype(np.float32)

    # Static correlation graph — same as vanilla GCN baseline (60d, |corr|>=0.3, top-30)
    # so any gain from the temporal head is isolated from graph-construction changes.
    head = x[train_sl.start : min(train_sl.start + 60, train_sl.stop)]
    edge_index_np, _ = build_correlation_edges(head, GraphConfig())
    edge_index = torch.from_numpy(edge_index_np).long().to(device)
    print(f"graph: {edge_index.shape[1]} edges over {N} nodes")

    model = TemporalGCN(F, cfg.hidden_dim, cfg.num_layers, cfg.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    def run_epoch(sl: slice, train: bool):
        model.train(train)
        W = cfg.window
        preds_full = np.zeros((sl.stop - sl.start, N), dtype=np.float32)
        y_full = np.zeros((sl.stop - sl.start, N), dtype=np.float32)
        m_full = np.zeros((sl.stop - sl.start, N), dtype=bool)
        losses = []
        for i_rel, t in enumerate(range(sl.start, sl.stop)):
            if t < W:
                continue
            # Build graph-conditioned embeddings for the W-day window.
            # This is a sequence of per-day GCN passes.
            embeds = []
            for s in range(t - W, t):
                m_s = mask[s]
                xt_np = xn[s] * m_s[:, None].astype(np.float32)
                xt = torch.from_numpy(xt_np).to(device)
                embeds.append(model.graph_embed(xt, edge_index))
            seq_full = torch.stack(embeds, dim=0)      # [W, N, hidden]
            seq_full = seq_full.permute(1, 0, 2)       # [N, W, hidden]
            m_t = mask[t]
            if m_t.sum() < 3:
                continue
            m_ten = torch.from_numpy(m_t).to(device)
            active = m_ten.nonzero(as_tuple=True)[0]
            seq_act = seq_full[active]                 # [N_active, W, hidden]
            out = model.temporal_head(seq_act)
            yhat = out["y_hat"]; qhat = out["q_hat"]
            yt = torch.from_numpy(y[t, m_t]).to(device)
            if train:
                lr_t = ranknet_loss(yhat.unsqueeze(0), yt.unsqueeze(0),
                                    torch.ones_like(yt, dtype=torch.bool).unsqueeze(0))
                lq_t = pinball_loss(yt, qhat, model.taus,
                                    torch.ones_like(yt, dtype=torch.bool))
                l = lr_t + cfg.quantile_weight * lq_t
                opt.zero_grad(); l.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
                losses.append(l.item())
            idx_np = active.cpu().numpy()
            preds_full[i_rel, idx_np] = yhat.detach().cpu().numpy()
            y_full[i_rel, idx_np] = yt.cpu().numpy()
            m_full[i_rel, idx_np] = True
        return preds_full, y_full, m_full, (float(np.mean(losses)) if losses else float("nan"))

    best_val = -float("inf"); best_val_rank = float("nan"); best_state = None; un = 0
    for epoch in range(cfg.epochs):
        _p, _y, _m, tl = run_epoch(train_sl, train=True)
        vp, vy, vm, _ = run_epoch(val_sl, train=False)
        v_ic = information_coefficient(vp, vy, vm)
        v_rank = rank_ic(vp, vy, vm)
        if v_ic > best_val:
            best_val = v_ic; best_val_rank = v_rank; un = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            un += 1
            if un >= cfg.patience:
                break
        print(f"[{epoch:02d}] train_loss={tl:.4f}  val_ic={v_ic:+.4f}" + ("  *best*" if un == 0 else ""))

    if best_state is not None:
        model.load_state_dict(best_state)
    tp, ty, tm, _ = run_epoch(test_sl, train=False)
    return {
        "model": "tgcn",
        "best_val_ic": best_val,
        "best_val_rank_ic": best_val_rank,
        "test_ic": information_coefficient(tp, ty, tm),
        "test_rank_ic": rank_ic(tp, ty, tm),
        "graph_edges": int(edge_index.shape[1]),
        "_test_preds": tp, "_test_y": ty, "_test_mask": tm,
    }


FOLD_BOUNDS = {
    # (train_end, val_end, test_end) — test_start = val_end + 1 trading day
    # Extended 2015-2022 panel: each test window spans a distinct market regime.
    1: ("2018-12-31", "2019-12-31", "2020-12-31"),  # Train 2015-2018 | Val 2019 | Test 2020 (COVID crash + vaccine recovery)
    2: ("2020-12-31", "2021-06-30", "2022-06-30"),  # Train 2015-2020 | Val 2021H1 | Test 2021H2-2022H1 (SPAC peak + bull-to-bear)
    3: ("2021-12-31", "2022-06-30", "2022-12-31"),  # Train 2015-2021 | Val 2022H1 | Test 2022H2 (deep bear + recovery)
}


def walk_forward_fold(dates, fold: int, horizon_days: int = 5) -> tuple[slice, slice, slice]:
    """Walk-forward CV with a `horizon_days` embargo at every boundary.

    The embargo drops the last `horizon_days` rows of train before val and of
    val before test. That prevents label-leakage: a sample at day t has label
    y[t] = log(close[t+H]/close[t]), so without the embargo the last H train
    labels overlap into val, and the last H val labels overlap into test.
    """
    if fold not in FOLD_BOUNDS:
        raise ValueError(f"fold must be in {list(FOLD_BOUNDS)}")
    train_end, val_end, test_end = FOLD_BOUNDS[fold]
    d_arr = pd.DatetimeIndex([pd.Timestamp(d) for d in dates])
    train_stop_raw = int(d_arr.searchsorted(pd.Timestamp(train_end) + pd.Timedelta(days=1)))
    val_stop_raw   = int(d_arr.searchsorted(pd.Timestamp(val_end)   + pd.Timedelta(days=1)))
    test_stop      = int(d_arr.searchsorted(pd.Timestamp(test_end)  + pd.Timedelta(days=1)))
    train_stop = max(0, train_stop_raw - horizon_days)
    val_start  = train_stop_raw
    val_stop   = max(val_start, val_stop_raw - horizon_days)
    test_start = val_stop_raw
    return slice(0, train_stop), slice(val_start, val_stop), slice(test_start, test_stop)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["ridge", "lstm", "gcn", "gat", "rgcn", "tgcn"], required=True)
    parser.add_argument("--max-tickers", type=int, default=300)
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2022-12-31")
    parser.add_argument("--horizon-days", type=int, default=5)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--fold", type=int, default=0,
                        help="Walk-forward CV fold index (1, 2, or 3). "
                             "0 = legacy single-split (67/12.5/20).")
    parser.add_argument("--output", type=Path, default=Path("results/baseline.json"))
    args = parser.parse_args()

    cfg = BaselineConfig(
        model=args.model, start_date=args.start, end_date=args.end,
        horizon_days=args.horizon_days, max_tickers=args.max_tickers,
        window=args.window, epochs=args.epochs, seed=args.seed,
    )
    panel, tickers, dates = build_enriched_panel(EnrichedPanelConfig(
        start_date=cfg.start_date, end_date=cfg.end_date,
        horizon_days=cfg.horizon_days, max_tickers=cfg.max_tickers,
    ))
    tensors = panel_to_tensors(panel, tickers, dates)
    x, y, mask = tensors["x"], tensors["y"], tensors["mask"]
    if args.fold == 0:
        slices = temporal_split(x.shape[0], cfg.val_fraction, cfg.test_fraction)
        fold_label = "single"
    else:
        slices = walk_forward_fold(dates, args.fold, cfg.horizon_days)
        fold_label = f"fold{args.fold}"
    tr, va, te = slices
    print(f"split [{fold_label}]: "
          f"train {tr.stop-tr.start}d ({dates[tr.start].date() if tr.stop>tr.start else '-'}..{dates[tr.stop-1].date() if tr.stop>tr.start else '-'})  "
          f"val {va.stop-va.start}d ({dates[va.start].date() if va.stop>va.start else '-'}..{dates[va.stop-1].date() if va.stop>va.start else '-'})  "
          f"test {te.stop-te.start}d ({dates[te.start].date() if te.stop>te.start else '-'}..{dates[te.stop-1].date() if te.stop>te.start else '-'})")

    if cfg.model == "ridge":
        result = run_ridge(cfg, x, y, mask, slices)
    elif cfg.model == "lstm":
        result = run_lstm(cfg, x, y, mask, slices)
    elif cfg.model == "rgcn":
        result = run_rgcn(cfg, x, y, mask, slices, tickers, dates=dates)
    elif cfg.model == "tgcn":
        result = run_tgcn(cfg, x, y, mask, slices)
    else:  # gcn | gat
        result = run_gnn(cfg, x, y, mask, slices)

    result["config"] = asdict(cfg)
    result["panel_T"] = x.shape[0]; result["panel_N"] = x.shape[1]; result["panel_F"] = x.shape[2]
    result["fold"] = args.fold
    result["fold_label"] = fold_label
    tr, va, te = slices
    if tr.stop > tr.start:
        result["train_range"] = [str(dates[tr.start].date()), str(dates[tr.stop-1].date())]
    if va.stop > va.start:
        result["val_range"]   = [str(dates[va.start].date()), str(dates[va.stop-1].date())]
    if te.stop > te.start:
        result["test_range"]  = [str(dates[te.start].date()), str(dates[te.stop-1].date())]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Strip per-day test arrays before JSON dump; save them as a companion .npz
    # so downstream regime-stratified and catalyst-window analysis can slice by day.
    _tp = result.pop("_test_preds", None)
    _ty = result.pop("_test_y", None)
    _tm = result.pop("_test_mask", None)
    args.output.write_text(json.dumps(result, indent=2, default=str))
    if _tp is not None and te.stop > te.start:
        npz_path = args.output.with_suffix(".npz")
        test_dates_iso = np.array([str(pd.Timestamp(dates[i]).date())
                                   for i in range(te.start, te.stop)], dtype="U10")
        np.savez_compressed(
            npz_path,
            preds=_tp, y=_ty, mask=_tm,
            test_dates=test_dates_iso,
            tickers=np.array(tickers, dtype=object),
        )
    print(f"\n{cfg.model.upper()}  Test IC {result['test_ic']:+.4f}  RankIC {result['test_rank_ic']:+.4f}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
