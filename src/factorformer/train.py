"""FactorFormer training loop.

Reuses ``src.invar.data.dataset.InvarDataset`` so the data interface is
byte-identical to the SWA-InVAR baseline. Predictions are saved per fold
and seed under ``experiments/factorformer/fold{F}/seed{S}_designFF/``.

Loss = MSE(y_hat, y_cs) + kl_weight * KL(q || p), where p is the prior
network conditioned on cross-sectional context only and q is the
posterior conditioned on context + future-return summary. At test time
only the prior is used (z = mu_p), so there is no future-info leakage.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from src.invar.data.dataset import InvarDataset, PANEL_FEATURE_DIM
from src.invar.evaluation.metrics import (
    daily_ic, daily_rank_ic, ndcg_at_k,
)
from src.factorformer.model import (
    FactorFormer, FactorFormerConfig, factor_kl, count_parameters,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collect_predictions(model: FactorFormer, ds: InvarDataset,
                        device: str) -> pd.DataFrame:
    model.eval()
    rows: list[dict] = []
    with torch.no_grad():
        for batch in ds:
            x = batch.features.to(device)
            out = model(x, y_future=None)
            y_hat = out["y_hat"].cpu().numpy()
            y_true = batch.y_cs.numpy()
            for ticker, p, t in zip(batch.tickers, y_hat, y_true):
                rows.append(dict(
                    date=batch.date, ticker=ticker,
                    y_hat=float(p), y_true=float(t),
                ))
    return pd.DataFrame(rows)


def evaluate(model: FactorFormer, ds: InvarDataset, device: str) -> dict:
    df = collect_predictions(model, ds, device)
    if df.empty:
        return dict(ic_mean=float("nan"), rank_ic_mean=float("nan"),
                    ndcg10_mean=float("nan"), n_days=0, df=df)
    ic = daily_ic(df)
    rk = daily_rank_ic(df)
    nd = ndcg_at_k(df, k=10)
    return dict(
        ic_mean=ic["mean"], rank_ic_mean=rk["mean"],
        ndcg10_mean=nd["mean"], n_days=ic["n_days"], df=df,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--lr", type=float, default=1.0e-4)
    p.add_argument("--weight-decay", type=float, default=1.0e-5)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--kl-weight", type=float, default=1.0e-3)
    p.add_argument("--n-factors", type=int, default=8)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--early-stop-patience", type=int, default=3)
    p.add_argument("--save-predictions", action="store_true")
    p.add_argument("--allow-fold-3", action="store_true")
    args = p.parse_args()

    if args.fold == 3 and not args.allow_fold_3:
        raise RuntimeError(
            "Fold 3 reserved; pass --allow-fold-3 explicitly.",
        )

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.output_dir) / f"seed{args.seed}_designFF"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = InvarDataset(fold=args.fold, split="train")
    val_ds = InvarDataset(fold=args.fold, split="val")
    test_ds = InvarDataset(fold=args.fold, split="test")
    print(f"[ff train] fold={args.fold} seed={args.seed} device={device}")
    print(f"[ff train] train days {len(train_ds)}  "
          f"val days {len(val_ds)}  test days {len(test_ds)}")

    cfg = FactorFormerConfig(
        n_features=PANEL_FEATURE_DIM,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_factors=args.n_factors,
        kl_weight=args.kl_weight,
    )
    model = FactorFormer(cfg).to(device)
    print(f"[ff train] params={count_parameters(model):,}  "
          f"k={cfg.n_factors}  kl_weight={cfg.kl_weight}")

    opt = AdamW(model.parameters(), lr=args.lr,
                weight_decay=args.weight_decay)

    n_train_steps = max(1, args.epochs * len(train_ds))

    def lr_lambda(step: int) -> float:
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, n_train_steps - args.warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return 0.1 + 0.9 * cosine

    scheduler = LambdaLR(opt, lr_lambda)

    best_val_ic = -float("inf")
    best_epoch = -1
    patience_left = args.early_stop_patience
    best_state: dict | None = None
    step = 0

    for epoch in range(args.epochs):
        model.train()
        loss_sum, mse_sum, kl_sum, nbatch = 0.0, 0.0, 0.0, 0
        for batch in train_ds:
            x = batch.features.to(device)
            y = batch.y_cs.to(device)
            opt.zero_grad()
            out = model(x, y_future=y)
            mse = ((out["y_hat"] - y) ** 2).mean()
            kl = factor_kl(out["mu_q"], out["log_sigma_q"],
                           out["mu_p"], out["log_sigma_p"])
            loss = mse + cfg.kl_weight * kl
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            scheduler.step()
            step += 1
            loss_sum += loss.item()
            mse_sum += mse.item()
            kl_sum += kl.item()
            nbatch += 1

        val = evaluate(model, val_ds, device)
        print(f"[ff train] epoch {epoch}: "
              f"loss={loss_sum/nbatch:.4f}  mse={mse_sum/nbatch:.4f}  "
              f"kl={kl_sum/nbatch:.4f}  "
              f"val_ic={val['ic_mean']:+.4f}  "
              f"val_rank_ic={val['rank_ic_mean']:+.4f}  "
              f"val_ndcg10={val['ndcg10_mean']:.4f}")

        if val["ic_mean"] > best_val_ic:
            best_val_ic = val["ic_mean"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            patience_left = args.early_stop_patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"[ff train] early stop at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"[ff train] reload best ckpt from epoch {best_epoch}")

    test = evaluate(model, test_ds, device)
    print(f"[ff train] test_ic={test['ic_mean']:+.4f}  "
          f"rank_ic={test['rank_ic_mean']:+.4f}  "
          f"ndcg10={test['ndcg10_mean']:.4f}")

    result = dict(
        config=dict(
            fold=args.fold, seed=args.seed,
            lr=args.lr, weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps, grad_clip=args.grad_clip,
            epochs=args.epochs,
            kl_weight=cfg.kl_weight, n_factors=cfg.n_factors,
            output_dir=str(out_dir),
        ),
        model_config=asdict(cfg),
        n_params=count_parameters(model),
        best_val_ic=best_val_ic,
        best_epoch=best_epoch,
        test_ic=test["ic_mean"],
        test_rank_ic=test["rank_ic_mean"],
        test_ndcg10=test["ndcg10_mean"],
        n_test_days=test["n_days"],
    )
    with open(out_dir / "results.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    if args.save_predictions and not test["df"].empty:
        test["df"].to_parquet(out_dir / "predictions.parquet", index=False)

    print("DONE")


if __name__ == "__main__":
    main()
