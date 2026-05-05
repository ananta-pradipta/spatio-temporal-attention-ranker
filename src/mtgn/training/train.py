"""First end-to-end MTGN training loop (Phase 1 vanilla TGN).

Minimal training script that validates the pipeline. For the first run
we use a static correlation graph + daily cross-sectional snapshots,
which mirrors DySTAGE's setting. The continuous-time event stream and
salience-gated episodic store layer on top in later iterations.

Usage:
    python3 -m src.mtgn.training.train --config configs/mtgn/phase1.yaml
    # or with a quick-run override:
    python3 -m src.mtgn.training.train --max-tickers 50 --start 2021-01-01 \\
        --end 2021-12-31 --epochs 5
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import Tensor, nn

from src.mtgn.training.graph_builder import GraphConfig, build_correlation_edges
from src.mtgn.training.panel import FEATURE_COLS, PanelConfig, build_panel, panel_to_tensors


@dataclass
class TrainConfig:
    start_date: str = "2020-01-01"
    end_date: str = "2022-12-31"
    horizon_days: int = 5
    max_tickers: int | None = 50
    hidden_dim: int = 128
    attention_heads: int = 4
    quantile_weight: float = 0.5
    lr: float = 5e-4
    weight_decay: float = 1e-5
    epochs: int = 5
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    seed: int = 42


def listnet_loss(y_hat: Tensor, y: Tensor, mask: Tensor) -> Tensor:
    """ListNet top-one probability cross-entropy over an active cross-section."""
    neg_inf = torch.finfo(y_hat.dtype).min
    y_hat = torch.where(mask, y_hat, torch.full_like(y_hat, neg_inf))
    y = torch.where(mask, y, torch.full_like(y, neg_inf))
    log_p_hat = torch.log_softmax(y_hat, dim=-1)
    p = torch.softmax(y, dim=-1)
    return -(p * log_p_hat).sum(dim=-1).mean()


def ranknet_loss(
    y_hat: Tensor, y: Tensor, mask: Tensor, max_pairs_per_day: int = 4096
) -> Tensor:
    """Pairwise RankNet with |Δy|-weighted margins.

    For each (t), sample ordered pairs (i, j) where y_t[i] > y_t[j] and
    both are active. Loss per pair: -(y_i - y_j) * log σ(s_i - s_j).
    The Δy weighting up-weights high-stakes pairs (analogous to LambdaRank).

    Avoids the ListNet saturation problem on large cross-sections: the
    gradient stays O(1) per pair regardless of N.

    Shapes:
      y_hat, y, mask: [B, N]   (B is the batch axis; typically 1 per day)
    """
    B, N = y_hat.shape
    loss_terms: list[Tensor] = []
    for b in range(B):
        m = mask[b]
        active = torch.where(m)[0]
        if active.numel() < 2:
            continue
        yb = y[b, active]
        sb = y_hat[b, active]
        # Pair matrix: y_diff[i, j] = y_i - y_j
        y_diff = yb.unsqueeze(1) - yb.unsqueeze(0)
        s_diff = sb.unsqueeze(1) - sb.unsqueeze(0)
        pos = y_diff > 0                                    # keep i where y_i > y_j
        if pos.sum() == 0:
            continue
        # Subsample if there are too many pairs
        idx = pos.nonzero(as_tuple=False)
        if idx.shape[0] > max_pairs_per_day:
            sel = torch.randperm(idx.shape[0], device=idx.device)[:max_pairs_per_day]
            idx = idx[sel]
        ii, jj = idx[:, 0], idx[:, 1]
        weight = y_diff[ii, jj]
        s_ij = s_diff[ii, jj]
        # -log σ(s_i - s_j) weighted by y_diff
        loss_terms.append((weight * torch.nn.functional.softplus(-s_ij)).mean())
    if not loss_terms:
        return torch.zeros((), dtype=y_hat.dtype, device=y_hat.device, requires_grad=True)
    return torch.stack(loss_terms).mean()


def pinball_loss(y: Tensor, q_hat: Tensor, taus: tuple[float, ...], mask: Tensor) -> Tensor:
    diff = y.unsqueeze(-1) - q_hat
    taus_t = torch.tensor(taus, dtype=diff.dtype, device=diff.device)
    loss = torch.maximum(taus_t * diff, (taus_t - 1.0) * diff)
    return loss[mask].mean()


def information_coefficient(y_hat: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    """Cross-sectional Pearson IC averaged over days. NaN-safe."""
    ics: list[float] = []
    for t in range(y_hat.shape[0]):
        m = mask[t]
        if m.sum() < 3:
            continue
        a = y_hat[t, m]
        b = y[t, m]
        if a.std() < 1e-8 or b.std() < 1e-8:
            continue
        ics.append(float(np.corrcoef(a, b)[0, 1]))
    return float(np.mean(ics)) if ics else float("nan")


def rank_ic(y_hat: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    from scipy.stats import spearmanr

    ics: list[float] = []
    for t in range(y_hat.shape[0]):
        m = mask[t]
        if m.sum() < 3:
            continue
        rho, _ = spearmanr(y_hat[t, m], y[t, m])
        if np.isnan(rho):
            continue
        ics.append(float(rho))
    return float(np.mean(ics)) if ics else float("nan")


class SimpleMTGN(nn.Module):
    """Phase-1 minimal MTGN proxy: spatial GAT over a static correlation graph.

    This is intentionally NOT the full PyG TGN pipeline, which requires
    event-level neighbor sampling and tends to obscure pipeline bugs on
    first end-to-end runs. It is the simplest architecture that still
    exercises the spatial-attention pathway and both heads (ranking +
    quantile), so we can verify the training loop, losses, metrics,
    and data loaders before layering the TGN memory and the episodic
    store on top.
    """

    def __init__(self, in_dim: int, cfg: TrainConfig):
        super().__init__()
        from torch_geometric.nn import GATConv

        self.input = nn.Sequential(nn.Linear(in_dim, cfg.hidden_dim), nn.ReLU())
        self.gat1 = GATConv(
            cfg.hidden_dim, cfg.hidden_dim // cfg.attention_heads,
            heads=cfg.attention_heads, dropout=0.1,
        )
        self.gat2 = GATConv(
            cfg.hidden_dim, cfg.hidden_dim // cfg.attention_heads,
            heads=cfg.attention_heads, dropout=0.1,
        )
        self.rank_head = nn.Linear(cfg.hidden_dim, 1)
        self.risk_head = nn.Linear(cfg.hidden_dim, 3)
        self.taus = (0.05, 0.50, 0.95)

    def forward(self, x: Tensor, edge_index: Tensor) -> dict[str, Tensor]:
        h = self.input(x)
        h = torch.relu(self.gat1(h, edge_index))
        h = torch.relu(self.gat2(h, edge_index))
        return {"y_hat": self.rank_head(h).squeeze(-1), "q_hat": self.risk_head(h)}


def temporal_split(T: int, val_frac: float, test_frac: float) -> tuple[slice, slice, slice]:
    test_start = int(T * (1 - test_frac))
    val_start = int(T * (1 - test_frac - val_frac))
    return slice(0, val_start), slice(val_start, test_start), slice(test_start, T)


def train_one_run(cfg: TrainConfig) -> dict[str, float]:
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
    x = torch.from_numpy(tensors["x"]).to(device)           # [T, N, F]
    y = torch.from_numpy(tensors["y"]).to(device)           # [T, N]
    mask = torch.from_numpy(tensors["mask"]).to(device)     # [T, N]
    T, N, F = x.shape
    print(f"panel shape: T={T} days, N={N} tickers, F={F} features")
    print(f"density: {mask.float().mean().item():.3f}")

    # Normalize features per-feature across (T, N) on train window only (approx).
    train_slice, val_slice, test_slice = temporal_split(T, cfg.val_fraction, cfg.test_fraction)
    mu = x[train_slice].reshape(-1, F).mean(dim=0)
    sd = x[train_slice].reshape(-1, F).std(dim=0).clamp(min=1e-6)
    x = (x - mu) / sd

    # Static correlation graph over the first 60 days of train data.
    head_arr = tensors["x"][train_slice.start : min(train_slice.start + 60, train_slice.stop)]
    edge_index_np, _ = build_correlation_edges(head_arr, GraphConfig())
    edge_index = torch.from_numpy(edge_index_np).to(device)
    print(f"graph edges: {edge_index.shape[1]}")

    model = SimpleMTGN(F, cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    def eval_slice(sl: slice) -> tuple[float, float, float]:
        model.eval()
        ys: list[np.ndarray] = []
        yhats: list[np.ndarray] = []
        ms: list[np.ndarray] = []
        losses: list[float] = []
        with torch.no_grad():
            for t in range(sl.start, sl.stop):
                out = model(x[t], edge_index)
                y_hat = out["y_hat"]
                q_hat = out["q_hat"]
                m = mask[t]
                if m.sum() < 3:
                    continue
                l_rank = listnet_loss(y_hat.unsqueeze(0), y[t].unsqueeze(0), m.unsqueeze(0))
                l_risk = pinball_loss(y[t], q_hat, model.taus, m)
                losses.append((l_rank + cfg.quantile_weight * l_risk).item())
                yhats.append(y_hat.cpu().numpy()[None, :])
                ys.append(y[t].cpu().numpy()[None, :])
                ms.append(m.cpu().numpy()[None, :])
        if not yhats:
            return float("nan"), float("nan"), float("nan")
        yhat_arr = np.concatenate(yhats)
        y_arr = np.concatenate(ys)
        m_arr = np.concatenate(ms)
        return (
            float(np.mean(losses)),
            information_coefficient(yhat_arr, y_arr, m_arr),
            rank_ic(yhat_arr, y_arr, m_arr),
        )

    history: list[dict] = []
    for epoch in range(cfg.epochs):
        model.train()
        losses: list[float] = []
        t0 = time.time()
        order = torch.randperm(train_slice.stop - train_slice.start)
        for ti in order.tolist():
            t = train_slice.start + ti
            m = mask[t]
            if m.sum() < 3:
                continue
            out = model(x[t], edge_index)
            y_hat = out["y_hat"]
            q_hat = out["q_hat"]
            l_rank = listnet_loss(y_hat.unsqueeze(0), y[t].unsqueeze(0), m.unsqueeze(0))
            l_risk = pinball_loss(y[t], q_hat, model.taus, m)
            loss = l_rank + cfg.quantile_weight * l_risk
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        train_loss = float(np.mean(losses)) if losses else float("nan")
        val_loss, val_ic, val_rank_ic = eval_slice(val_slice)
        dt = time.time() - t0
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_ic": val_ic,
            "val_rank_ic": val_rank_ic,
            "time_sec": round(dt, 2),
        }
        history.append(row)
        print(
            f"[{epoch:02d}] train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  val_ic={val_ic:+.4f}  "
            f"val_rank_ic={val_rank_ic:+.4f}  ({dt:.1f}s)"
        )

    test_loss, test_ic, test_rank_ic = eval_slice(test_slice)
    print(f"\nTEST  loss={test_loss:.4f}  ic={test_ic:+.4f}  rank_ic={test_rank_ic:+.4f}")

    out = {
        "panel_T": T, "panel_N": N, "panel_F": F,
        "edges": int(edge_index.shape[1]),
        "test_loss": test_loss, "test_ic": test_ic, "test_rank_ic": test_rank_ic,
        "history": history,
        "config": asdict(cfg),
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--max-tickers", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--horizon-days", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path("results/mtgn_phase1_run.json"))
    args = parser.parse_args()

    cfg = TrainConfig()
    if args.start: cfg.start_date = args.start
    if args.end:   cfg.end_date = args.end
    if args.max_tickers is not None: cfg.max_tickers = args.max_tickers
    if args.epochs is not None: cfg.epochs = args.epochs
    if args.horizon_days is not None: cfg.horizon_days = args.horizon_days

    result = train_one_run(cfg)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=str))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
