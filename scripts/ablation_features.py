"""Feature-group ablation on 244-ticker panel.

Three feature groups:
  PRICE  = log_return, log_volume, realized_vol                  (3 features)
  ST     = 5 StockTwits features                                 (5 features)
  ALL    = PRICE + ST                                            (8 features)

Three model classes:
  Ridge  (linear, alpha=10)
  MLP    (2-layer 128 hidden)
  GAT    (MLP projection + 2 GATConv layers on mechanistic graph)

Outputs 3 x 3 = 9 cells of (Test IC, Test RankIC) on the same panel,
same seed (11), same RankNet+pinball training, same split.
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


class MLPOnly(nn.Module):
    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
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


def run_ridge(x_sub, y, mask, slices):
    from sklearn.linear_model import Ridge
    x_np = x_sub.numpy() if hasattr(x_sub, "numpy") else x_sub
    y_np = y.numpy() if hasattr(y, "numpy") else y
    m_np = mask.numpy() if hasattr(mask, "numpy") else mask
    T, N, F = x_np.shape
    train_sl, val_sl, test_sl = slices
    Xtr = x_np[train_sl].reshape(-1, F); ytr = y_np[train_sl].reshape(-1); mtr = m_np[train_sl].reshape(-1)
    mu = Xtr[mtr].mean(axis=0); sd = Xtr[mtr].std(axis=0).clip(min=1e-6)
    Xtr_n = (Xtr - mu) / sd
    model = Ridge(alpha=10.0).fit(Xtr_n[mtr], ytr[mtr])
    yhat_all = model.predict(((x_np.reshape(-1, F) - mu) / sd)).reshape(T, N)
    ics, ricks = [], []
    for t in range(test_sl.start, test_sl.stop):
        m = m_np[t]
        if m.sum() < 3: continue
        a = yhat_all[t, m]; b = y_np[t, m]
        if a.std() < 1e-8 or b.std() < 1e-8: continue
        ics.append(float(np.corrcoef(a, b)[0, 1]))
        from scipy.stats import spearmanr
        rho, _ = spearmanr(a, b)
        if np.isfinite(rho): ricks.append(float(rho))
    return {"test_ic": float(np.mean(ics)), "test_rank_ic": float(np.mean(ricks))}


def run_torch(ctor_name, x_sub, y, mask, edge_index, slices):
    torch.manual_seed(SEED); np.random.seed(SEED)
    T, N, F = x_sub.shape
    train_sl, val_sl, test_sl = slices
    mu = x_sub[train_sl].reshape(-1, F).mean(dim=0)
    sd = x_sub[train_sl].reshape(-1, F).std(dim=0).clamp(min=1e-6)
    xn = (x_sub - mu) / sd

    if ctor_name == "MLP":
        model = MLPOnly(F, HIDDEN)
        needs_graph = False
    elif ctor_name == "GAT":
        model = MLPGAT(F, HIDDEN, HEADS)
        needs_graph = True
    else:
        raise ValueError(ctor_name)

    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
    best_val = -float("inf"); best_state = None; un = 0
    for epoch in range(EPOCHS):
        model.train()
        for t in range(train_sl.start, train_sl.stop):
            m = mask[t]
            if m.sum() < 3: continue
            out = model(xn[t], edge_index) if needs_graph else model(xn[t])
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
                out = model(xn[t], edge_index) if needs_graph else model(xn[t])
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
            out = model(xn[t], edge_index) if needs_graph else model(xn[t])
            yh.append(out["y_hat"].cpu().numpy()[None, :])
            ya.append(y[t].cpu().numpy()[None, :])
            mm.append(m.cpu().numpy()[None, :])
    yha = np.concatenate(yh); yaa = np.concatenate(ya); mma = np.concatenate(mm)
    return {"test_ic": information_coefficient(yha, yaa, mma),
            "test_rank_ic": rank_ic(yha, yaa, mma),
            "best_val_ic": best_val}


def main():
    cfg = PanelConfig(start_date="2020-01-01", end_date="2022-12-31", horizon_days=5, max_tickers=300)
    panel, tickers, dates = build_panel(cfg)
    tensors = panel_to_tensors(panel, tickers, dates)
    x_full = torch.from_numpy(tensors["x"])
    y = torch.from_numpy(tensors["y"])
    mask = torch.from_numpy(tensors["mask"])
    T, N, F = x_full.shape
    slices = temporal_split(T, 0.15, 0.15)

    ei_np, _ = build_mechanistic_edges(tickers, EdgeBuildConfig(
        train_start="2020-01-01", train_end=str(dates[slices[0].stop - 1].date()),
    ))
    edge_index = torch.from_numpy(ei_np)
    print(f"panel: T={T} N={N} F={F} edges={edge_index.shape[1]}")

    PRICE_IDX = [FEATURE_COLS.index(c) for c in ("log_return", "log_volume", "realized_vol")]
    ST_IDX = [FEATURE_COLS.index(c) for c in ("st_volume_24h", "st_volume_change_30d",
                                              "st_bullish_ratio", "st_sentiment_dispersion", "st_labeled_ratio")]
    groups = {
        "PRICE":  PRICE_IDX,
        "ST":     ST_IDX,
        "ALL":    PRICE_IDX + ST_IDX,
    }

    results = {}
    for group, idx in groups.items():
        x_sub = x_full[:, :, idx]
        print(f"\n=== group={group}  features={idx} ===")
        t0 = time.time()
        r = run_ridge(x_sub, y, mask, slices)
        print(f"  Ridge            Test IC {r['test_ic']:+.4f}  RankIC {r['test_rank_ic']:+.4f}  ({time.time()-t0:.1f}s)")
        results[(group, "Ridge")] = r

        t0 = time.time()
        r = run_torch("MLP", x_sub, y, mask, None, slices)
        print(f"  MLP              Test IC {r['test_ic']:+.4f}  RankIC {r['test_rank_ic']:+.4f}  ({time.time()-t0:.1f}s)")
        results[(group, "MLP")] = r

        t0 = time.time()
        r = run_torch("GAT", x_sub, y, mask, edge_index, slices)
        print(f"  GAT              Test IC {r['test_ic']:+.4f}  RankIC {r['test_rank_ic']:+.4f}  ({time.time()-t0:.1f}s)")
        results[(group, "GAT")] = r

    print("\n=== SUMMARY ===")
    print(f"{'Model':<10}  {'PRICE IC':>8}  {'ST IC':>8}  {'ALL IC':>8}   {'PRICE RankIC':>12}  {'ST RankIC':>10}  {'ALL RankIC':>10}")
    for model in ("Ridge", "MLP", "GAT"):
        row = []
        for g in ("PRICE", "ST", "ALL"):
            row.append(results[(g, model)]["test_ic"])
        row2 = []
        for g in ("PRICE", "ST", "ALL"):
            row2.append(results[(g, model)]["test_rank_ic"])
        print(f"{model:<10}  {row[0]:+.4f}  {row[1]:+.4f}  {row[2]:+.4f}   {row2[0]:+.4f}       {row2[1]:+.4f}     {row2[2]:+.4f}")

    Path("results/ablation_features.json").write_text(json.dumps(
        {f"{g}_{m}": v for (g, m), v in results.items()}, indent=2, default=str
    ))


if __name__ == "__main__":
    main()
