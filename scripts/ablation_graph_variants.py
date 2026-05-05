"""Test three GAT variants to isolate why adding the graph hurts IC.

  A. GAT with EMPTY edges (no aggregation happens)
  B. GAT WITH residual skip (preserves direct-feature signal)
  C. GAT with SUM aggregation (pyg's aggr='add' instead of default attention)

All at 244-ticker full universe, seed 11, 10 epochs, ranknet loss.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from src.mtgn.graph.edges import EdgeBuildConfig, build_mechanistic_edges
from src.mtgn.training.panel import FEATURE_COLS, PanelConfig, build_panel, panel_to_tensors
from src.mtgn.training.train import information_coefficient, pinball_loss, rank_ic, ranknet_loss, temporal_split


SEED = 11
EPOCHS = 10
PATIENCE = 4
HIDDEN = 128
HEADS = 4
QW = 0.5


class GATVariant(nn.Module):
    def __init__(self, in_dim: int, hidden: int, heads: int, residual: bool = False, aggregator: str = "mean"):
        super().__init__()
        from torch_geometric.nn import GATConv
        self.input = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU())
        self.gat1 = GATConv(hidden, hidden // heads, heads=heads, dropout=0.1, aggr=aggregator)
        self.gat2 = GATConv(hidden, hidden // heads, heads=heads, dropout=0.1, aggr=aggregator)
        self.residual = residual
        self.rank_head = nn.Linear(hidden, 1)
        self.risk_head = nn.Linear(hidden, 3)
        self.taus = (0.05, 0.50, 0.95)

    def forward(self, x, edge_index):
        h0 = self.input(x)
        h1 = torch.relu(self.gat1(h0, edge_index))
        h2 = torch.relu(self.gat2(h1, edge_index))
        if self.residual:
            h2 = h2 + h0
        return {"y_hat": self.rank_head(h2).squeeze(-1), "q_hat": self.risk_head(h2)}


def run(name: str, model_ctor, x, y, mask, edge_index, slices):
    torch.manual_seed(SEED); np.random.seed(SEED)
    T, N, F = x.shape
    train_sl, val_sl, test_sl = slices
    mu = x[train_sl].reshape(-1, F).mean(dim=0)
    sd = x[train_sl].reshape(-1, F).std(dim=0).clamp(min=1e-6)
    xn = (x - mu) / sd

    model = model_ctor(F)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
    best_val = -float("inf"); best_state = None; un = 0

    for epoch in range(EPOCHS):
        model.train()
        for t in range(train_sl.start, train_sl.stop):
            m = mask[t]
            if m.sum() < 3: continue
            out = model(xn[t], edge_index)
            l = ranknet_loss(out["y_hat"].unsqueeze(0), y[t].unsqueeze(0), m.unsqueeze(0)) \
                + QW * pinball_loss(y[t], out["q_hat"], model.taus, m)
            opt.zero_grad(); l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        # val
        model.eval()
        yh, ya, mm = [], [], []
        with torch.no_grad():
            for t in range(val_sl.start, val_sl.stop):
                m = mask[t]
                if m.sum() < 3: continue
                out = model(xn[t], edge_index)
                yh.append(out["y_hat"].cpu().numpy()[None, :])
                ya.append(y[t].cpu().numpy()[None, :])
                mm.append(m.cpu().numpy()[None, :])
        yha = np.concatenate(yh); yaa = np.concatenate(ya); mma = np.concatenate(mm)
        v_ic = information_coefficient(yha, yaa, mma)
        if v_ic > best_val:
            best_val = v_ic; un = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            un += 1
            if un >= PATIENCE: break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    yh, ya, mm = [], [], []
    with torch.no_grad():
        for t in range(test_sl.start, test_sl.stop):
            m = mask[t]
            if m.sum() < 3: continue
            out = model(xn[t], edge_index)
            yh.append(out["y_hat"].cpu().numpy()[None, :])
            ya.append(y[t].cpu().numpy()[None, :])
            mm.append(m.cpu().numpy()[None, :])
    yha = np.concatenate(yh); yaa = np.concatenate(ya); mma = np.concatenate(mm)
    return {
        "variant": name,
        "test_ic": information_coefficient(yha, yaa, mma),
        "test_rank_ic": rank_ic(yha, yaa, mma),
        "best_val_ic": best_val,
    }


def main():
    cfg = PanelConfig(start_date="2020-01-01", end_date="2022-12-31", horizon_days=5, max_tickers=300)
    panel, tickers, dates = build_panel(cfg)
    tensors = panel_to_tensors(panel, tickers, dates)
    x = torch.from_numpy(tensors["x"])
    y = torch.from_numpy(tensors["y"])
    mask = torch.from_numpy(tensors["mask"])
    T, N, F = x.shape
    slices = temporal_split(T, 0.15, 0.15)

    ei_np, _ = build_mechanistic_edges(tickers, EdgeBuildConfig(
        train_start="2020-01-01", train_end=str(dates[slices[0].stop - 1].date()),
    ))
    edge_index = torch.from_numpy(ei_np)
    print(f"panel: T={T} N={N} F={F}  edges: {edge_index.shape[1]}")

    # A: empty edges
    empty = torch.zeros((2, 0), dtype=torch.long)
    results = []
    variants = [
        ("A_empty_edges",     lambda F: GATVariant(F, HIDDEN, HEADS, residual=False, aggregator="mean"), empty),
        ("B_residual",        lambda F: GATVariant(F, HIDDEN, HEADS, residual=True,  aggregator="mean"), edge_index),
        ("C_sum_aggregation", lambda F: GATVariant(F, HIDDEN, HEADS, residual=False, aggregator="add"),  edge_index),
        ("D_residual+sum",    lambda F: GATVariant(F, HIDDEN, HEADS, residual=True,  aggregator="add"),  edge_index),
    ]
    for name, ctor, ei in variants:
        t0 = time.time()
        r = run(name, ctor, x, y, mask, ei, slices)
        r["time_sec"] = round(time.time() - t0, 2)
        results.append(r)
        print(f"  {name:20s}  Test IC {r['test_ic']:+.4f}  RankIC {r['test_rank_ic']:+.4f}  (val {r['best_val_ic']:+.4f}, {r['time_sec']:.1f}s)")

    Path("results/ablation_graph_variants.json").write_text(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
