"""Training script for the pilot Temporal GNN stock ranking experiment.

Usage:
    python scripts/train/train.py --config configs/pilot_biotech.yaml

Loads processed snapshots, splits by time, trains a TemporalGAT with
ranking loss, evaluates on val/test, and logs metrics.
"""

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml

from model import ListNetLoss, PairwiseRankingLoss, build_model
from metrics import compute_all_metrics


def load_config(config_path: str) -> dict:
    """Load YAML config."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def split_by_time(
    dates: List[str],
    train_end: str,
    val_end: str,
) -> Tuple[List[int], List[int], List[int]]:
    """Split snapshot indices by date.

    Args:
        dates: List of date strings (YYYY-MM-DD).
        train_end: Last date (inclusive) for training.
        val_end: Last date (inclusive) for validation.

    Returns:
        Tuple of (train_indices, val_indices, test_indices).
    """
    train_idx, val_idx, test_idx = [], [], []
    for i, d in enumerate(dates):
        if d <= train_end:
            train_idx.append(i)
        elif d <= val_end:
            val_idx.append(i)
        else:
            test_idx.append(i)
    return train_idx, val_idx, test_idx


def evaluate_split(
    model: nn.Module,
    snapshots: list,
    indices: List[int],
    device: torch.device,
    top_k: int,
    bottom_k: int,
) -> Dict[str, float]:
    """Evaluate model on a split (val or test).

    Args:
        model: The trained model.
        snapshots: List of snapshot dicts.
        indices: Indices into snapshots for this split.
        device: Torch device.
        top_k: Long portfolio size.
        bottom_k: Short portfolio size.

    Returns:
        Dictionary of averaged metrics.
    """
    model.eval()
    all_metrics: Dict[str, List[float]] = {
        "ic": [], "rank_ic": [], "long_return": [],
        "short_return": [], "long_short_return": [],
    }

    with torch.no_grad():
        for idx in indices:
            snap = snapshots[idx]
            x = snap["x"].to(device)
            edge_index = snap["edge_index"].to(device)
            y = snap["y"]
            fwd_ret = snap["fwd_return"]

            scores = model(x, edge_index).cpu().numpy()
            y_np = y.numpy()
            fwd_np = fwd_ret.numpy()

            m = compute_all_metrics(scores, y_np, fwd_np, top_k, bottom_k)
            for k, v in m.items():
                all_metrics[k].append(v)

    # Average
    result = {}
    for k, v_list in all_metrics.items():
        result[k] = float(np.mean(v_list)) if v_list else 0.0
    return result


def train(config_path: str) -> None:
    """Main training loop.

    Args:
        config_path: Path to YAML config.
    """
    cfg = load_config(config_path)
    project_root = Path(config_path).resolve().parent.parent
    set_seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load data ---
    data_path = project_root / cfg["data"]["processed_dir"] / "biotech_snapshots.pt"
    if not data_path.exists():
        print(f"FAIL: {data_path} not found. Run build_features.py first.")
        sys.exit(1)

    data = torch.load(data_path, weights_only=False)
    snapshots = data["snapshots"]
    dates = data["dates"]
    tickers = data["tickers"]
    feature_names = data["feature_names"]
    print(f"Loaded {len(snapshots)} snapshots, {len(tickers)} tickers.")
    print(f"Features: {feature_names}")

    # --- Split ---
    train_cfg = cfg["training"]
    train_idx, val_idx, test_idx = split_by_time(
        dates, train_cfg["train_end"], train_cfg["val_end"]
    )
    print(f"Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    if len(train_idx) == 0 or len(val_idx) == 0:
        print("FAIL: Empty train or val split. Check date ranges.")
        sys.exit(1)

    # --- Model ---
    model_cfg = cfg["model"]
    in_channels = snapshots[0]["x"].shape[1]
    model = build_model(
        model_name=model_cfg["name"],
        in_channels=in_channels,
        cfg=model_cfg,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {param_count:,}")

    # --- Loss ---
    if train_cfg["loss"] == "listnet":
        criterion = ListNetLoss(temperature=train_cfg["listnet_temperature"])
    elif train_cfg["loss"] == "pairwise":
        criterion = PairwiseRankingLoss()
    else:
        print(f"FAIL: Unknown loss '{train_cfg['loss']}'")
        sys.exit(1)

    # --- Optimizer ---
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )

    # --- Logging setup ---
    eval_cfg = cfg["evaluation"]
    log_dir = project_root / cfg["logging"]["log_dir"]
    os.makedirs(log_dir, exist_ok=True)
    print_every = cfg["logging"]["print_every"]

    # --- Training loop ---
    best_val_ic = -float("inf")
    patience_counter = 0
    history: List[Dict] = []

    print(f"\nStarting training for up to {train_cfg['epochs']} epochs...")
    print("-" * 70)

    for epoch in range(1, train_cfg["epochs"] + 1):
        model.train()
        epoch_losses = []

        # Shuffle training indices each epoch
        shuffled = train_idx.copy()
        random.shuffle(shuffled)

        for idx in shuffled:
            snap = snapshots[idx]
            x = snap["x"].to(device)
            edge_index = snap["edge_index"].to(device)
            y = snap["y"].to(device)

            # Mask for valid targets (non-NaN)
            mask = ~torch.isnan(y)
            if mask.sum() < 2:
                continue

            optimizer.zero_grad()
            scores = model(x, edge_index)
            loss = criterion(scores, y, mask)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            epoch_losses.append(loss.item())

        avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0

        # --- Validation ---
        val_metrics = evaluate_split(
            model, snapshots, val_idx, device,
            eval_cfg["top_k"], eval_cfg["bottom_k"],
        )

        record = {
            "epoch": epoch,
            "train_loss": float(avg_loss),
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(record)

        # Early stopping on Rank IC
        current_val_ic = val_metrics["rank_ic"]
        if current_val_ic > best_val_ic:
            best_val_ic = current_val_ic
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), log_dir / "best_model.pt")
        else:
            patience_counter += 1

        if epoch % print_every == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d} | "
                f"Loss: {avg_loss:.4f} | "
                f"Val IC: {val_metrics['ic']:.4f} | "
                f"Val RankIC: {val_metrics['rank_ic']:.4f} | "
                f"Val L/S: {val_metrics['long_short_return']:.4f} | "
                f"Patience: {patience_counter}/{train_cfg['patience']}"
            )

        if patience_counter >= train_cfg["patience"]:
            print(f"\nEarly stopping at epoch {epoch}.")
            break

    print("-" * 70)

    # --- Test evaluation ---
    if test_idx:
        # Load best model
        model.load_state_dict(torch.load(log_dir / "best_model.pt", weights_only=True))
        test_metrics = evaluate_split(
            model, snapshots, test_idx, device,
            eval_cfg["top_k"], eval_cfg["bottom_k"],
        )
        print("\nTest Results:")
        for k, v in test_metrics.items():
            print(f"  {k}: {v:.4f}")
    else:
        test_metrics = {}
        print("\nWARN: No test data available.")

    # --- Save results ---
    results = {
        "config": cfg,
        "history": history,
        "best_val_rank_ic": float(best_val_ic),
        "test_metrics": test_metrics,
        "tickers": tickers,
        "split_sizes": {
            "train": len(train_idx),
            "val": len(val_idx),
            "test": len(test_idx),
        },
        "timestamp": datetime.now().isoformat(),
    }
    results_path = log_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    # Also save training history as CSV for easy plotting
    import pandas as pd
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(log_dir / "training_history.csv", index=False)
    print(f"Training history saved to {log_dir / 'training_history.csv'}")


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Train temporal GNN for stock ranking.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pilot_biotech.yaml",
        help="Path to YAML config.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    config_path = str(project_root / args.config)
    train(config_path)


if __name__ == "__main__":
    main()
