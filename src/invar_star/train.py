"""InVAR-STAR training loop.

Reuses src.invar.data.dataset.InvarDataset for per-day cross-sectional batches.
Each gradient step is one trading day; the batch dimension equals the day's
active universe size N_t (variable across days, no padding). The macro
lookback (L, F_macro) is broadcast across the N_t stocks on the day so the
model sees x_stock of shape (N_t, 26, L) and x_macro of shape (N_t, 24, L).

Composite loss (design doc Section 4.7):
    L_total = lambda_mse * L_mse
            + lambda_ic * L_neg_wic
            + lambda_throttle * L_throttle_kl
            + lambda_balance * L_load_balance

Schedule (design doc Section 4.8):
    - AdamW, lr 3e-4, weight_decay 1e-4 for epochs 0 to swa_start-1.
    - Cosine warmup over `warmup_epochs`, cosine anneal thereafter.
    - From swa_start onward, constant swa_lr (1e-4) and SWA `update_parameters`
      every epoch.
    - Gate temperature tau annealed linearly from tau_start (1.0) at epoch 0
      to tau_end (0.1) at the final epoch.

The smoke mode (--smoke N) overrides n_epochs to N and pulls swa_start to
N-1 so the final epoch updates the SWA wrapper at least once.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.optim import AdamW
from torch.optim.swa_utils import AveragedModel

from src.invar.data.dataset import InvarDataset
from src.invar.evaluation.metrics import (
    daily_ic, daily_rank_ic, ndcg_at_k,
)
from src.invar_star.losses import (
    load_balance_loss,
    throttle_kl_prior,
    weighted_pearson_ic_loss,
)
from src.invar_star.model import (
    InVARSTAR,
    count_parameters,
    set_global_seed,
)


def day_batch_to_tensors(batch, device: str) -> tuple:
    """Convert one InvarDayBatch to (x_stock, x_macro, y) tensors.

    Args:
        batch: one InvarDayBatch from InvarDataset.
        device: 'cpu' or 'cuda'.

    Returns:
        x_stock: (N_t, n_stock=26, L).
        x_macro: (N_t, n_macro=24, L), broadcast from the per-day macro vector.
        y:       (N_t,) cross-sectional z-scored fwd_return.
    """
    x_stock = batch.features.permute(0, 2, 1).to(device)
    n_t = x_stock.shape[0]
    x_macro = (
        batch.macro.transpose(0, 1).unsqueeze(0).expand(n_t, -1, -1).to(device)
    )
    y = batch.y_cs.to(device)
    return x_stock, x_macro, y


def evaluate(model, ds: InvarDataset, device: str, tau_eval: float = 0.1,
             fixed_beta: float | None = None) -> dict:
    """Run inference on a dataset split. Returns aggregated metrics and a DataFrame.

    Uses deterministic gate (model.eval(), so no logistic noise) with tau_eval.
    Records per-day beta_t and per-day expert routing entropy for the
    interpretability deliverables in design doc Section 7.4. When fixed_beta
    is set (A1 ablation), the gate is bypassed at eval time too.
    """
    model.eval()
    rows: list[dict] = []
    betas: list[float] = []
    routes: list[np.ndarray] = []
    with torch.no_grad():
        for batch in ds:
            x_stock, x_macro, y = day_batch_to_tensors(batch, device)
            out = model(x_stock, x_macro, tau=tau_eval, fixed_beta=fixed_beta)
            y_hat = out["y_hat"].squeeze(-1).cpu().numpy()
            for ticker, p, t in zip(batch.tickers, y_hat, y.cpu().numpy()):
                rows.append(dict(
                    date=batch.date, ticker=ticker,
                    y_hat=float(p), y_true=float(t),
                ))
            betas.append(float(out["beta"][0].item()))
            routes.append(out["route_probs"].mean(dim=0).cpu().numpy())
    df = pd.DataFrame(rows)
    if df.empty:
        return dict(
            ic_mean=float("nan"), rank_ic_mean=float("nan"),
            ndcg10_mean=float("nan"), n_days=0,
            beta_mean=float("nan"), beta_std=float("nan"),
            expert_entropy=float("nan"), df=df, betas=[], routes=[],
        )
    ic = daily_ic(df)
    rk = daily_rank_ic(df)
    nd = ndcg_at_k(df, k=10)
    betas_arr = np.array(betas)
    routes_arr = np.stack(routes)
    eps = 1.0e-9
    entropies = -np.sum(routes_arr * np.log(routes_arr + eps), axis=1)
    return dict(
        ic_mean=ic["mean"], rank_ic_mean=rk["mean"],
        ndcg10_mean=nd["mean"], n_days=ic["n_days"],
        beta_mean=float(betas_arr.mean()), beta_std=float(betas_arr.std()),
        expert_entropy=float(entropies.mean()),
        df=df, betas=betas_arr.tolist(), routes=routes_arr.tolist(),
    )


def lr_for_epoch(epoch: int, n_epochs: int, swa_start: int,
                 warmup_epochs: int, base_lr: float, swa_lr: float) -> float:
    if epoch >= swa_start:
        return swa_lr
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / max(1, warmup_epochs)
    progress = (epoch - warmup_epochs) / max(1, swa_start - warmup_epochs)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return base_lr * (0.1 + 0.9 * cosine)


def tau_for_epoch(epoch: int, n_epochs: int,
                  tau_start: float, tau_end: float) -> float:
    if n_epochs <= 1:
        return tau_start
    frac = epoch / (n_epochs - 1)
    return tau_start + (tau_end - tau_start) * frac


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--config", type=str, default="configs/invar_star.yaml")
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--smoke", type=int, default=0,
                   help="if > 0, override n_epochs with this many epochs")
    p.add_argument("--allow-fold-3", action="store_true")
    p.add_argument(
        "--fixed-beta", type=float, default=None,
        help="if set, bypass the self-throttling gate and use this value "
             "for beta_t throughout training and eval (A1 ablation).",
    )
    args = p.parse_args()

    if args.fold == 3 and not args.allow_fold_3:
        raise RuntimeError("Fold 3 reserved; pass --allow-fold-3 explicitly.")

    set_global_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"config missing: {cfg_path}")
    cfg = yaml.safe_load(cfg_path.read_text())

    mcfg = cfg["model"]
    tcfg = cfg["train"]
    lcfg = cfg["loss"]

    if args.smoke > 0:
        n_epochs = args.smoke
        swa_start = max(1, n_epochs - 1)
        warmup_epochs = min(tcfg["warmup_epochs"], max(1, n_epochs // 3))
    else:
        n_epochs = tcfg["n_epochs"]
        swa_start = tcfg["swa_start"]
        warmup_epochs = tcfg["warmup_epochs"]

    run_id = f"invar_star_fold{args.fold}_seed{args.seed}"
    out_dir = Path(args.out_dir) / f"seed{args.seed}_designSTAR"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = InvarDataset(fold=args.fold, split="train")
    val_ds = InvarDataset(fold=args.fold, split="val")
    test_ds = InvarDataset(fold=args.fold, split="test")
    print(f"[is train] {run_id} device={device}")
    print(f"[is train] train days {len(train_ds)}  "
          f"val days {len(val_ds)}  test days {len(test_ds)}")

    model = InVARSTAR(
        lookback=mcfg["lookback"],
        d_model=mcfg["d_model"],
        n_heads=mcfg["n_heads"],
        n_layers=mcfg["n_layers"],
        n_experts=mcfg["n_experts"],
        top_k=mcfg["top_k"],
        phi_dim=mcfg["phi_dim"],
        n_stock=mcfg["n_stock"],
        n_macro=mcfg["n_macro"],
        noise_std=mcfg["noise_std"],
        expert_dropout=mcfg["expert_dropout"],
    ).to(device)
    print(f"[is train] params={count_parameters(model):,}  "
          f"n_epochs={n_epochs}  swa_start={swa_start}  warmup={warmup_epochs}")

    opt = AdamW(model.parameters(), lr=tcfg["lr"],
                weight_decay=tcfg["weight_decay"])
    swa_model = AveragedModel(model)
    swa_updated = False

    csv_path = out_dir / "training_log.csv"
    csv_fields = [
        "run_id", "epoch", "tau", "lr",
        "train_loss", "train_mse", "train_ic", "train_throttle", "train_balance",
        "val_ic", "val_rank_ic", "val_ndcg10",
        "beta_mean", "beta_std", "expert_entropy",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()

    final_epoch = -1
    for epoch in range(n_epochs):
        tau = tau_for_epoch(epoch, n_epochs, tcfg["tau_start"], tcfg["tau_end"])
        cur_lr = lr_for_epoch(epoch, n_epochs, swa_start, warmup_epochs,
                              tcfg["lr"], tcfg["swa_lr"])
        for g in opt.param_groups:
            g["lr"] = cur_lr

        model.train()
        loss_sum, mse_sum, ic_sum, thr_sum, bal_sum, nbatch = 0.0, 0.0, 0.0, 0.0, 0.0, 0
        for batch in train_ds:
            x_stock, x_macro, y = day_batch_to_tensors(batch, device)
            opt.zero_grad()
            out = model(x_stock, x_macro, tau=tau, fixed_beta=args.fixed_beta)
            l_mse = F.mse_loss(out["y_hat"].squeeze(-1), y)
            l_ic = weighted_pearson_ic_loss(out["y_hat"], y)
            l_thr = throttle_kl_prior(out["beta"])
            l_bal = load_balance_loss(out["route_probs"])
            loss = (
                lcfg["lambda_mse"] * l_mse
                + lcfg["lambda_ic"] * l_ic
                + lcfg["lambda_throttle"] * l_thr
                + lcfg["lambda_balance"] * l_bal
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
            opt.step()
            loss_sum += loss.item()
            mse_sum += l_mse.item()
            ic_sum += l_ic.item()
            thr_sum += l_thr.item()
            bal_sum += l_bal.item()
            nbatch += 1

        train_loss = loss_sum / max(1, nbatch)

        if epoch >= swa_start:
            swa_model.update_parameters(model)
            swa_updated = True

        val = evaluate(model, val_ds, device, tau_eval=tcfg["tau_end"],
                       fixed_beta=args.fixed_beta)
        print(
            f"[is train] epoch {epoch}: "
            f"loss={train_loss:.4f}  "
            f"val_ic={val['ic_mean']:+.4f}  "
            f"val_rank_ic={val['rank_ic_mean']:+.4f}  "
            f"beta_mean={val['beta_mean']:.3f}  "
            f"beta_std={val['beta_std']:.3f}  "
            f"H_expert={val['expert_entropy']:.3f}  "
            f"tau={tau:.3f}  lr={cur_lr:.2e}"
        )

        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writerow({
                "run_id": run_id, "epoch": epoch, "tau": tau, "lr": cur_lr,
                "train_loss": train_loss,
                "train_mse": mse_sum / max(1, nbatch),
                "train_ic": ic_sum / max(1, nbatch),
                "train_throttle": thr_sum / max(1, nbatch),
                "train_balance": bal_sum / max(1, nbatch),
                "val_ic": val["ic_mean"], "val_rank_ic": val["rank_ic_mean"],
                "val_ndcg10": val["ndcg10_mean"],
                "beta_mean": val["beta_mean"], "beta_std": val["beta_std"],
                "expert_entropy": val["expert_entropy"],
            })
        final_epoch = epoch

    final_model = swa_model if swa_updated else model
    ckpt_path = out_dir / f"swa_invar_star_fold{args.fold}_seed{args.seed}.pt"
    torch.save(final_model.state_dict(), ckpt_path)
    print(f"[is train] saved checkpoint to {ckpt_path}")

    test = evaluate(final_model, test_ds, device, tau_eval=tcfg["tau_end"],
                    fixed_beta=args.fixed_beta)
    print(
        f"[is train] FINAL test_ic={test['ic_mean']:+.4f}  "
        f"test_rank_ic={test['rank_ic_mean']:+.4f}  "
        f"test_ndcg10={test['ndcg10_mean']:.4f}  "
        f"beta_mean={test['beta_mean']:.3f}  "
        f"beta_std={test['beta_std']:.3f}"
    )

    result = dict(
        config=dict(
            run_id=run_id, fold=args.fold, seed=args.seed,
            n_epochs=n_epochs, swa_start=swa_start,
            warmup_epochs=warmup_epochs, smoke=args.smoke,
            lr=tcfg["lr"], swa_lr=tcfg["swa_lr"],
            weight_decay=tcfg["weight_decay"],
            grad_clip=tcfg["grad_clip"],
            lambda_mse=lcfg["lambda_mse"], lambda_ic=lcfg["lambda_ic"],
            lambda_throttle=lcfg["lambda_throttle"],
            lambda_balance=lcfg["lambda_balance"],
            tau_start=tcfg["tau_start"], tau_end=tcfg["tau_end"],
            output_dir=str(out_dir),
        ),
        model_config=mcfg,
        n_params=count_parameters(model),
        test_ic=test["ic_mean"],
        test_rank_ic=test["rank_ic_mean"],
        test_ndcg10=test["ndcg10_mean"],
        n_test_days=test["n_days"],
        beta_mean_test=test["beta_mean"],
        beta_std_test=test["beta_std"],
        expert_entropy_test=test["expert_entropy"],
        final_epoch=final_epoch,
    )
    with open(out_dir / "results.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    if not test["df"].empty:
        test["df"].to_parquet(out_dir / "predictions.parquet", index=False)

    if test["betas"]:
        np.save(out_dir / "test_betas.npy", np.array(test["betas"]))
        np.save(out_dir / "test_routes.npy", np.array(test["routes"]))

    print("DONE")


if __name__ == "__main__":
    main()
