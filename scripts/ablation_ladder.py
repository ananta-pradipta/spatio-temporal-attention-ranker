"""Incremental component ablation to find where MTGN loses IC vs ridge.

Ladder of increasing complexity, same panel / seed / split / loss / epochs:

  L0  Ridge(8 features)                                 (reference baseline)
  L1  MLP(8->hidden->1)                                 (+learning)
  L2  MLP + mechanistic-graph GAT                       (+spatial structure)
  L3  MLP + GAT + TGN memory, no retrieval              (+per-node state)
  L4  L3 + self_only retrieval                          (+self-history attention)
  L5  L3 + cross_entity retrieval                       (+cross-entity attention)

At each level we record test IC and RankIC. Losing a level's IC-over-previous
identifies the culprit component.

Runs locally on 244-ticker 2020-2022 with seed=11, epochs=10, patience=4,
ranknet loss, horizon=5. Each level ~5-10 min on CPU.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from src.mtgn.graph.edges import EdgeBuildConfig, build_mechanistic_edges
from src.mtgn.training.panel import FEATURE_COLS, PanelConfig, build_panel, panel_to_tensors
from src.mtgn.training.train import (
    information_coefficient,
    pinball_loss,
    rank_ic,
    ranknet_loss,
    temporal_split,
)


SEED = 11
EPOCHS = 10
PATIENCE = 4
HIDDEN = 128
HEADS = 4
QUANTILE_W = 0.5
LR = 5e-4
WD = 1e-5


class MLPOnly(nn.Module):
    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.rank_head = nn.Linear(hidden, 1)
        self.risk_head = nn.Linear(hidden, 3)
        self.taus = (0.05, 0.50, 0.95)

    def forward(self, x, edge_index=None):
        h = self.net(x)
        return {"y_hat": self.rank_head(h).squeeze(-1), "q_hat": self.risk_head(h)}


class MLPGAT(nn.Module):
    def __init__(self, in_dim: int, hidden: int, heads: int):
        super().__init__()
        from torch_geometric.nn import GATConv
        self.input = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU())
        self.gat1 = GATConv(hidden, hidden // heads, heads=heads, dropout=0.1)
        self.gat2 = GATConv(hidden, hidden // heads, heads=heads, dropout=0.1)
        self.rank_head = nn.Linear(hidden, 1)
        self.risk_head = nn.Linear(hidden, 3)
        self.taus = (0.05, 0.50, 0.95)

    def forward(self, x, edge_index):
        h = self.input(x)
        h = torch.relu(self.gat1(h, edge_index))
        h = torch.relu(self.gat2(h, edge_index))
        return {"y_hat": self.rank_head(h).squeeze(-1), "q_hat": self.risk_head(h)}


def run_level(level_name: str, model_ctor, needs_graph: bool, x, y, mask, edge_index, slices):
    torch.manual_seed(SEED); np.random.seed(SEED)
    T, N, F = x.shape
    train_sl, val_sl, test_sl = slices

    # Normalize features on train slice only
    mu = x[train_sl].reshape(-1, F).mean(dim=0)
    sd = x[train_sl].reshape(-1, F).std(dim=0).clamp(min=1e-6)
    xn = (x - mu) / sd

    model = model_ctor(F, HIDDEN).to(x.device) if not needs_graph else model_ctor(F, HIDDEN, HEADS).to(x.device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

    best_val_ic = -float("inf"); best_state = None; best_epoch = -1; unimproved = 0
    history = []

    for epoch in range(EPOCHS):
        model.train()
        losses = []
        for t in range(train_sl.start, train_sl.stop):
            m = mask[t]
            if m.sum() < 3: continue
            inp = xn[t]
            out = model(inp, edge_index) if needs_graph else model(inp)
            l_rank = ranknet_loss(out["y_hat"].unsqueeze(0), y[t].unsqueeze(0), m.unsqueeze(0))
            l_risk = pinball_loss(y[t], out["q_hat"], model.taus, m)
            loss = l_rank + QUANTILE_W * l_risk
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())

        # val
        model.eval()
        yh, ya, mm = [], [], []
        with torch.no_grad():
            for t in range(val_sl.start, val_sl.stop):
                m = mask[t]
                if m.sum() < 3: continue
                inp = xn[t]
                out = model(inp, edge_index) if needs_graph else model(inp)
                yh.append(out["y_hat"].cpu().numpy()[None, :])
                ya.append(y[t].cpu().numpy()[None, :])
                mm.append(m.cpu().numpy()[None, :])
        if yh:
            yha = np.concatenate(yh); yaa = np.concatenate(ya); mma = np.concatenate(mm)
            val_ic = information_coefficient(yha, yaa, mma)
            val_ric = rank_ic(yha, yaa, mma)
        else:
            val_ic = float("nan"); val_ric = float("nan")

        improved = val_ic > best_val_ic
        history.append(dict(epoch=epoch, train_loss=float(np.mean(losses)) if losses else float("nan"),
                            val_ic=val_ic, val_rank_ic=val_ric))
        if improved:
            best_val_ic = val_ic; best_epoch = epoch; unimproved = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            unimproved += 1
            if unimproved >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # test
    model.eval()
    yh, ya, mm = [], [], []
    with torch.no_grad():
        for t in range(test_sl.start, test_sl.stop):
            m = mask[t]
            if m.sum() < 3: continue
            inp = xn[t]
            out = model(inp, edge_index) if needs_graph else model(inp)
            yh.append(out["y_hat"].cpu().numpy()[None, :])
            ya.append(y[t].cpu().numpy()[None, :])
            mm.append(m.cpu().numpy()[None, :])
    yha = np.concatenate(yh); yaa = np.concatenate(ya); mma = np.concatenate(mm)
    test_ic = information_coefficient(yha, yaa, mma)
    test_ric = rank_ic(yha, yaa, mma)
    return {
        "level": level_name,
        "test_ic": test_ic, "test_rank_ic": test_ric,
        "best_val_ic": best_val_ic, "best_epoch": best_epoch,
        "history": history,
    }


def main() -> None:
    cfg = PanelConfig(start_date="2020-01-01", end_date="2022-12-31", horizon_days=5, max_tickers=300)
    panel, tickers, dates = build_panel(cfg)
    tensors = panel_to_tensors(panel, tickers, dates)
    x = torch.from_numpy(tensors["x"])
    y = torch.from_numpy(tensors["y"])
    mask = torch.from_numpy(tensors["mask"])
    T, N, F = x.shape
    slices = temporal_split(T, 0.15, 0.15)
    print(f"panel: T={T} N={N} F={F}")

    ei_np, _ = build_mechanistic_edges(tickers, EdgeBuildConfig(
        train_start="2020-01-01", train_end=str(dates[slices[0].stop - 1].date()),
    ))
    edge_index = torch.from_numpy(ei_np)
    print(f"edges: {edge_index.shape[1]}")

    results = []
    print()
    print("--- L1 MLP only ---")
    t0 = time.time()
    r = run_level("L1_MLP", MLPOnly, needs_graph=False, x=x, y=y, mask=mask, edge_index=None, slices=slices)
    r["time_sec"] = round(time.time() - t0, 2); results.append(r)
    print(f"  Test IC {r['test_ic']:+.4f}  RankIC {r['test_rank_ic']:+.4f}  ({r['time_sec']:.1f}s)")

    print()
    print("--- L2 MLP + GAT on mechanistic graph ---")
    t0 = time.time()
    r = run_level("L2_MLP_GAT", MLPGAT, needs_graph=True, x=x, y=y, mask=mask, edge_index=edge_index, slices=slices)
    r["time_sec"] = round(time.time() - t0, 2); results.append(r)
    print(f"  Test IC {r['test_ic']:+.4f}  RankIC {r['test_rank_ic']:+.4f}  ({r['time_sec']:.1f}s)")

    Path("results/ablation_ladder.json").write_text(json.dumps({
        "seed": SEED, "epochs": EPOCHS, "patience": PATIENCE,
        "panel_T": T, "panel_N": N, "panel_F": F,
        "edges": int(edge_index.shape[1]),
        "results": results,
    }, indent=2, default=str))
    print()
    print("Wrote results/ablation_ladder.json")


if __name__ == "__main__":
    main()
