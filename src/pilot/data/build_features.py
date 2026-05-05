"""Build node features, graph adjacency, and targets from raw price data.

Usage:
    python scripts/data/build_features.py --config configs/pilot_biotech.yaml

Reads the close and volume panels from data/raw/, computes features,
builds correlation-based adjacency matrices, and saves everything
to data/processed/ as .pt files for PyTorch Geometric.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from scipy import stats


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def compute_returns(
    close: pd.DataFrame,
    windows: List[int],
) -> Dict[str, pd.DataFrame]:
    """Compute log returns over multiple windows.

    Args:
        close: Close price panel (dates x tickers).
        windows: List of return horizons in trading days.

    Returns:
        Dictionary mapping feature name to DataFrame.
    """
    features = {}
    for w in windows:
        ret = np.log(close / close.shift(w))
        features[f"ret_{w}d"] = ret
    return features


def compute_volatility(
    close: pd.DataFrame,
    window: int,
) -> pd.DataFrame:
    """Compute rolling volatility (std of daily log returns).

    Args:
        close: Close price panel.
        window: Rolling window size.

    Returns:
        DataFrame of rolling volatility.
    """
    daily_ret = np.log(close / close.shift(1))
    return daily_ret.rolling(window).std()


def compute_volume_ratio(
    volume: pd.DataFrame,
    window: int,
) -> pd.DataFrame:
    """Compute volume relative to rolling average.

    Args:
        volume: Volume panel.
        window: Rolling window for average.

    Returns:
        DataFrame of volume ratios.
    """
    avg_vol = volume.rolling(window).mean()
    # Avoid division by zero
    avg_vol = avg_vol.replace(0, np.nan)
    return volume / avg_vol


def compute_rolling_correlation(
    close: pd.DataFrame,
    window: int,
) -> Dict[str, pd.DataFrame]:
    """Compute rolling pairwise correlation of daily returns.

    Args:
        close: Close price panel.
        window: Rolling window in trading days.

    Returns:
        Dictionary mapping date string to correlation matrix (DataFrame).
    """
    daily_ret = np.log(close / close.shift(1)).dropna()
    tickers = daily_ret.columns.tolist()

    corr_matrices = {}
    dates = daily_ret.index[window:]

    for i in range(window, len(daily_ret)):
        date = daily_ret.index[i]
        window_data = daily_ret.iloc[i - window:i]
        corr = window_data.corr()
        corr_matrices[str(date.date())] = corr

    return corr_matrices


def threshold_correlation(
    corr: pd.DataFrame,
    threshold_percentile: float,
) -> np.ndarray:
    """Convert correlation matrix to binary adjacency via percentile threshold.

    Self-loops are removed. Keeps the top (1 - threshold_percentile) fraction
    of absolute correlations as edges.

    Args:
        corr: Correlation matrix (tickers x tickers).
        threshold_percentile: Fraction of edges to keep (e.g. 0.80 keeps top 20%).

    Returns:
        Binary adjacency matrix as numpy array.
    """
    n = len(corr)
    abs_corr = corr.abs().values.copy()
    np.fill_diagonal(abs_corr, 0.0)

    # Get upper triangle values for threshold computation
    upper_vals = abs_corr[np.triu_indices(n, k=1)]
    if len(upper_vals) == 0:
        return np.zeros((n, n), dtype=np.float32)

    cutoff = np.percentile(upper_vals, threshold_percentile * 100)
    adj = (abs_corr >= cutoff).astype(np.float32)
    return adj


def compute_targets(
    close: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    """Compute next-period return ranks as targets.

    Args:
        close: Close price panel.
        horizon: Forward return horizon in days.

    Returns:
        DataFrame of cross-sectional ranks (0 to n_stocks-1), lower is worse.
    """
    fwd_ret = close.shift(-horizon) / close - 1.0
    # Rank cross-sectionally (per row); ties get average rank
    ranks = fwd_ret.rank(axis=1, method="average", na_option="keep")
    # Normalize to [0, 1] using proper axis alignment
    n_stocks = ranks.count(axis=1)
    ranks = ranks.sub(1).div((n_stocks - 1).replace(0, np.nan), axis=0)
    return ranks


def build_and_save(config_path: str) -> None:
    """Main pipeline: load data, compute features/edges/targets, save .pt files.

    Args:
        config_path: Path to YAML config.
    """
    project_root = Path(__file__).resolve().parent.parent.parent
    cfg = load_config(config_path)
    data_cfg = cfg["data"]
    feat_cfg = cfg["features"]
    target_cfg = cfg["target"]

    raw_dir = project_root / data_cfg["raw_dir"]
    processed_dir = project_root / data_cfg["processed_dir"]
    os.makedirs(processed_dir, exist_ok=True)

    # Load panels
    close = pd.read_csv(raw_dir / "close_panel.csv", index_col=0, parse_dates=True)
    volume = pd.read_csv(raw_dir / "volume_panel.csv", index_col=0, parse_dates=True)

    tickers = close.columns.tolist()
    print(f"Loaded {len(tickers)} tickers, {len(close)} trading days.")

    # --- Compute node features ---
    print("Computing node features...")
    feature_dfs = {}

    # Returns
    ret_features = compute_returns(close, feat_cfg["return_windows"])
    feature_dfs.update(ret_features)

    # Volatility
    feature_dfs["volatility"] = compute_volatility(
        close, feat_cfg["volatility_window"]
    )

    # Volume ratio
    feature_dfs["volume_ratio"] = compute_volume_ratio(
        volume, feat_cfg["volume_norm_window"]
    )

    feature_names = list(feature_dfs.keys())
    print(f"  Features: {feature_names}")

    # --- Compute targets ---
    print("Computing targets...")
    target_ranks = compute_targets(close, target_cfg["horizon"])
    # Also store raw forward returns for evaluation
    fwd_returns = close.shift(-target_cfg["horizon"]) / close - 1.0

    # --- Compute correlation-based adjacency ---
    print("Computing rolling correlations (this may take a minute)...")
    corr_matrices = compute_rolling_correlation(
        close, data_cfg["correlation_window"]
    )

    # --- Align all data to common dates ---
    # Find dates where all features, targets, and correlations are available
    feature_dates = set(close.index)
    for name, df in feature_dfs.items():
        feature_dates = feature_dates & set(df.dropna(how="all").index)
    target_dates = set(target_ranks.dropna(how="all").index)
    corr_dates = set(pd.to_datetime(list(corr_matrices.keys())))

    common_dates = sorted(feature_dates & target_dates & corr_dates)
    print(f"  Common dates after alignment: {len(common_dates)}")

    if len(common_dates) < 100:
        print("WARN: Very few common dates. Check data quality.")

    # --- Build tensors ---
    print("Building tensors...")

    # For each date, create: node features [n_stocks, n_features],
    # adjacency [2, n_edges], target [n_stocks]
    snapshots = []
    valid_dates = []

    for date in common_dates:
        date_str = str(date.date())

        # Node features
        feat_list = []
        skip = False
        for name in feature_names:
            vals = feature_dfs[name].loc[date, tickers].values.astype(np.float32)
            if np.all(np.isnan(vals)):
                skip = True
                break
            feat_list.append(vals)

        if skip:
            continue

        x = np.stack(feat_list, axis=1)  # [n_stocks, n_features]

        # Replace NaN with 0 (stocks with missing data)
        nan_mask = np.isnan(x)
        x[nan_mask] = 0.0

        # Z-score normalization per feature (cross-sectional)
        for f_idx in range(x.shape[1]):
            col = x[:, f_idx]
            valid = ~nan_mask[:, f_idx]
            if valid.sum() > 1:
                mu = col[valid].mean()
                sigma = col[valid].std()
                if sigma > 1e-8:
                    x[:, f_idx] = (col - mu) / sigma
                else:
                    x[:, f_idx] = 0.0

        # Target ranks
        y = target_ranks.loc[date, tickers].values.astype(np.float32)
        fwd_ret = fwd_returns.loc[date, tickers].values.astype(np.float32)

        if np.all(np.isnan(y)):
            continue

        # Adjacency from correlation
        if date_str not in corr_matrices:
            continue

        corr_mat = corr_matrices[date_str]
        adj = threshold_correlation(corr_mat, data_cfg["correlation_threshold"])

        # Convert to edge index (COO format)
        edge_src, edge_dst = np.where(adj > 0)
        edge_index = np.stack([edge_src, edge_dst], axis=0)

        snapshots.append({
            "x": torch.tensor(x, dtype=torch.float32),
            "edge_index": torch.tensor(edge_index, dtype=torch.long),
            "y": torch.tensor(y, dtype=torch.float32),
            "fwd_return": torch.tensor(fwd_ret, dtype=torch.float32),
        })
        valid_dates.append(date)

    print(f"  Built {len(snapshots)} valid snapshots.")

    if len(snapshots) == 0:
        print("FAIL: No valid snapshots. Something is wrong with the data.")
        sys.exit(1)

    # --- Save ---
    save_path = processed_dir / "biotech_snapshots.pt"
    torch.save({
        "snapshots": snapshots,
        "dates": [str(d.date()) for d in valid_dates],
        "tickers": tickers,
        "feature_names": feature_names,
    }, save_path)
    print(f"Saved {len(snapshots)} snapshots to {save_path}")

    # Print summary stats
    avg_edges = np.mean([s["edge_index"].shape[1] for s in snapshots])
    print(f"  Average edges per snapshot: {avg_edges:.0f}")
    print(f"  Node feature dim: {snapshots[0]['x'].shape[1]}")
    print(f"  Date range: {valid_dates[0].date()} to {valid_dates[-1].date()}")


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Build features and graph dataset.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pilot_biotech.yaml",
        help="Path to YAML config file.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    config_path = str(project_root / args.config)
    build_and_save(config_path)


if __name__ == "__main__":
    main()
