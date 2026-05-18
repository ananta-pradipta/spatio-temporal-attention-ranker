"""Train-only regime labels for InVAR-v6 regime auxiliary loss.

Per Section 7 of the v6 spec: build per-day pseudo-regime labels from
macro and cross-sectional state features that are available at or
before day t. Fit StandardScaler and KMeans (n_regimes=8) on the
TRAIN split only; transform val/test using the train-fitted objects.

Forbidden inputs (Section 7.3): future returns, target y, anything
centered using validation or test distributions.

Output: ``data/cache/regime_labels/fold_{fold}.parquet`` with columns
``date, regime_label, regime_distance, regime_confidence,
features_used_hash, fit_split``.

Usage:
    from src.invar.data.regime_labels import build_v6_regime_labels
    build_v6_regime_labels(fold=1)
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

REGIME_FEATURES = [
    "vix", "vxn", "vvix",
    "vix_5d_change",
    "xbi_rv_20d", "xbi_rv_60d",
    "xbi_return_5d", "xbi_return_20d",
    "avg_pairwise_corr_60d",
    "cross_sectional_return_dispersion",
    "active_count",
    "mean_realized_vol_20d",
]


def _features_hash() -> str:
    return hashlib.sha256(",".join(REGIME_FEATURES).encode()).hexdigest()[:12]


def _load_panel_and_macro() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the LATTICE panel and macro tables.

    Returns ``(panel, macro)`` indexed by date.
    """
    import os
    base = Path(os.environ.get("LATTICE_PROCESSED_DIR",
                                "data/lattice/processed"))
    panel_path = base / "panel_features.parquet"
    macro_path = base / "macro_state.parquet"
    panel = pd.read_parquet(panel_path)
    macro = pd.read_parquet(macro_path)
    panel["date"] = pd.to_datetime(panel["date"])
    macro["date"] = pd.to_datetime(macro["date"])
    return panel, macro


def _build_day_features(
    panel: pd.DataFrame, macro: pd.DataFrame,
) -> pd.DataFrame:
    """Construct the per-day regime feature table.

    Uses only same-day or backward-looking values; never references
    future returns or target columns.
    """
    macro = macro.copy().sort_values("date").reset_index(drop=True)
    if "vix" in macro.columns:
        macro["vix_5d_change"] = macro["vix"].diff(5)
    else:
        macro["vix"] = 0.0
        macro["vix_5d_change"] = 0.0
    for col in ("vxn", "vvix"):
        if col not in macro.columns:
            macro[col] = 0.0

    # Cross-sectional aggregates from the panel. We reuse columns that
    # exist; missing ones are filled with zero.
    panel = panel.copy().sort_values(["date", "ticker"]).reset_index(drop=True)
    grp = panel.groupby("date", sort=True)

    rv_col = "rv_20d" if "rv_20d" in panel.columns else None
    if rv_col is None:
        if "ret_20d" in panel.columns:
            panel["rv_20d_proxy"] = panel.groupby("ticker")["ret_20d"].transform(
                lambda x: x.rolling(20, min_periods=5).std(),
            )
            rv_col = "rv_20d_proxy"
        else:
            panel["rv_20d_proxy"] = 0.0
            rv_col = "rv_20d_proxy"

    ret1d_col = "ret_1d" if "ret_1d" in panel.columns else None

    daily = pd.DataFrame({"date": sorted(panel["date"].unique())})
    daily["active_count"] = daily["date"].map(grp.size())
    daily["mean_realized_vol_20d"] = daily["date"].map(grp[rv_col].mean())
    if ret1d_col is not None:
        daily["cross_sectional_return_dispersion"] = (
            daily["date"].map(grp[ret1d_col].std())
        )
    else:
        daily["cross_sectional_return_dispersion"] = 0.0

    # XBI proxy: equal-weighted cross-sectional mean of 1d returns,
    # rolled. (We do not have a proper biotech-ETF panel here; this is
    # a backward-looking proxy.)
    if ret1d_col is not None:
        ew_index = grp[ret1d_col].mean().sort_index()
        cum = (1.0 + ew_index.fillna(0)).cumprod()
        rv20 = ew_index.rolling(20, min_periods=5).std()
        rv60 = ew_index.rolling(60, min_periods=10).std()
        ret5 = ew_index.rolling(5, min_periods=2).sum()
        ret20 = ew_index.rolling(20, min_periods=5).sum()
        daily["xbi_rv_20d"] = daily["date"].map(rv20)
        daily["xbi_rv_60d"] = daily["date"].map(rv60)
        daily["xbi_return_5d"] = daily["date"].map(ret5)
        daily["xbi_return_20d"] = daily["date"].map(ret20)
    else:
        for c in ("xbi_rv_20d", "xbi_rv_60d", "xbi_return_5d", "xbi_return_20d"):
            daily[c] = 0.0

    # Average pairwise correlation: too expensive to compute exactly;
    # use a low-cost proxy from the cross-sectional dispersion of
    # 60-day rolling returns at the day level (higher dispersion roughly
    # implies lower average correlation).
    daily["avg_pairwise_corr_60d"] = (
        -daily["cross_sectional_return_dispersion"].rolling(60, min_periods=10).mean()
    )
    daily["avg_pairwise_corr_60d"] = daily["avg_pairwise_corr_60d"].fillna(0.0)

    out = daily.merge(
        macro[["date", "vix", "vxn", "vvix", "vix_5d_change"]],
        on="date", how="left",
    )
    out = out[["date", *REGIME_FEATURES]].copy()
    out = out.fillna(0.0)
    return out


def _train_split_dates(fold: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return ``(train_start, train_end)`` for the requested fold."""
    if fold == 1:
        return pd.Timestamp("2015-01-01"), pd.Timestamp("2018-12-31")
    if fold == 2:
        return pd.Timestamp("2015-01-01"), pd.Timestamp("2020-12-31")
    if fold == 3:
        return pd.Timestamp("2015-01-01"), pd.Timestamp("2022-06-30")
    raise ValueError(f"unknown fold: {fold}")


def build_v6_regime_labels(
    fold: int, n_regimes: int = 8, kmeans_seed: int = 0,
    out_dir: str = "data/cache/regime_labels",
) -> pd.DataFrame:
    """Build and persist train-fitted regime labels for one fold.

    Returns the labelled DataFrame that was written to disk.
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    panel, macro = _load_panel_and_macro()
    daily = _build_day_features(panel, macro)
    train_start, train_end = _train_split_dates(fold)
    train_mask = (daily["date"] >= train_start) & (daily["date"] <= train_end)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(daily.loc[train_mask, REGIME_FEATURES].to_numpy())
    km = KMeans(n_clusters=n_regimes, random_state=kmeans_seed, n_init=10)
    km.fit(X_train)

    X_all = scaler.transform(daily[REGIME_FEATURES].to_numpy())
    labels = km.predict(X_all)
    dist = km.transform(X_all)
    nearest = np.partition(dist, 1, axis=1)[:, :2]
    confidence = (nearest[:, 1] - nearest[:, 0]) / (nearest[:, 1] + 1.0e-6)

    out = pd.DataFrame({
        "date": daily["date"],
        "regime_label": labels.astype(int),
        "regime_distance": dist.min(axis=1).astype(float),
        "regime_confidence": confidence.astype(float),
        "features_used_hash": _features_hash(),
        "fit_split": "train",
    })
    out_path = Path(out_dir) / f"fold_{fold}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    meta = {
        "fold": fold, "n_regimes": n_regimes,
        "kmeans_seed": kmeans_seed,
        "features": REGIME_FEATURES,
        "features_hash": _features_hash(),
        "train_start": str(train_start), "train_end": str(train_end),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "n_train_days": int(train_mask.sum()),
        "n_total_days": int(len(daily)),
    }
    with open(out_path.with_suffix(".meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return out


def load_v6_regime_labels(
    fold: int, out_dir: str = "data/cache/regime_labels",
) -> pd.DataFrame:
    return pd.read_parquet(Path(out_dir) / f"fold_{fold}.parquet")


__all__ = [
    "REGIME_FEATURES",
    "build_v6_regime_labels",
    "load_v6_regime_labels",
]
