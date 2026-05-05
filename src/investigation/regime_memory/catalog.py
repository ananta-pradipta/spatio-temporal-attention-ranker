"""Regime catalog: k-means on train signatures + per-cluster graphs.

Enforced invariants (REM-Audits 2, 3, 4):
  - Clustering uses train-fold days only.
  - Signature z-score statistics come from train-fold only.
  - Catalog state (mu, sd, centroids, labels_train, per-cluster
    top_neighbors) is frozen after build; test-time retrieval is a
    read-only lookup.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans

from src.mtgn.graph.edges import EdgeBuildConfig, build_mechanistic_edges
from src.mtgn.model.utils.patch_construction import precompute_top_neighbors


@dataclass
class RegimeCatalog:
    mu: np.ndarray            # [D] signature means (train-fold)
    sd: np.ndarray            # [D] signature stds  (train-fold)
    centroids: np.ndarray     # [K, D] cluster centroids in z-space
    labels_train: np.ndarray  # [T_train] cluster labels for train days
    top_neighbors: dict[int, np.ndarray]  # cluster idx -> [N, num_neighbors]
    n_per_cluster: dict[int, int]         # cluster idx -> days in train
    K: int
    N: int
    num_neighbors: int


def assign_days_to_cluster(signatures: np.ndarray, catalog: RegimeCatalog) -> np.ndarray:
    """Hard-assign every panel day to its nearest cluster (read-only)."""
    z = (signatures - catalog.mu) / np.maximum(catalog.sd, 1e-6)
    # [T, D] - [K, D] -> [T, K] distances
    d = np.linalg.norm(z[:, None, :] - catalog.centroids[None, :, :], axis=2)
    return np.argmin(d, axis=1).astype(np.int64)


def _cluster_correlation_neighbors(log_returns: np.ndarray, mask: np.ndarray,
                                   cluster_days: np.ndarray, N: int,
                                   num_neighbors: int) -> np.ndarray:
    """Build a [N, num_neighbors] top-N graph from the pairwise correlation of
    ticker log-returns restricted to this cluster's training days.
    """
    if cluster_days.size < 20:
        # Too few days; fall back to uniform (no correlation signal)
        return np.full((N, num_neighbors), -1, dtype=np.int64)
    sub = log_returns[cluster_days]        # [d_c, N]
    sub_m = mask[cluster_days]             # [d_c, N]
    always_active = sub_m.all(axis=0)
    cols = np.where(always_active)[0]
    if cols.size < 2:
        return np.full((N, num_neighbors), -1, dtype=np.int64)
    X = sub[:, cols]                       # [d_c, n_active]
    C = np.abs(np.corrcoef(X, rowvar=False))  # [n_active, n_active]
    np.fill_diagonal(C, -np.inf)
    top = np.full((N, num_neighbors), -1, dtype=np.int64)
    for local_i, global_i in enumerate(cols):
        row = C[local_i]
        valid = np.where(np.isfinite(row))[0]
        if valid.size == 0:
            continue
        k = min(num_neighbors, valid.size)
        local_top = valid[np.argsort(-row[valid])[:k]]
        top[global_i, :k] = cols[local_top]
    return top


def build_catalog(signatures: np.ndarray, tickers: list, dates: list,
                  log_returns: np.ndarray, mask: np.ndarray,
                  train_slice: slice, cfg_start_date: str,
                  cfg_train_end: str, K: int, num_neighbors: int,
                  kmeans_seed: int = 0, single_graph: bool = False) -> RegimeCatalog:
    """Build regime catalog on train-fold days only.

    per-cluster top_neighbors = union of (a) a global mechanistic
    top-`num_neighbors//2` graph (same across clusters; the "structural
    prior") and (b) a cluster-specific correlation top-`num_neighbors//2`
    graph (the "regime-typed statistical structure"). Duplicates
    deduplicated; unfilled slots left as -1. This preserves
    multi-relation spatial structure (mechanistic + correlation).
    """
    import pandas as pd

    sig_train = signatures[train_slice]
    finite = np.all(np.isfinite(sig_train), axis=1)
    # Use only fully-finite train signatures for clustering
    sig_train_f = sig_train[finite]
    mu = sig_train_f.mean(axis=0)
    sd = sig_train_f.std(axis=0).clip(min=1e-6)
    z_train = (sig_train - mu) / sd
    # Impute any non-finite remaining entries with 0 (mean in z-space)
    z_train = np.where(np.isfinite(z_train), z_train, 0.0)

    km = KMeans(n_clusters=K, n_init=10, random_state=kmeans_seed)
    labels_train = km.fit_predict(z_train)
    centroids = km.cluster_centers_

    N = log_returns.shape[1]
    half_k = num_neighbors // 2

    # (a) Global mechanistic graph (train-fold edges). In single-graph mode,
    # compute the FULL top-num_neighbors once and reuse for every cluster;
    # otherwise compute the top-half for the mechanistic half of each
    # cluster's union graph.
    mech_cfg = EdgeBuildConfig(train_start=cfg_start_date, train_end=cfg_train_end)
    ei, ew = build_mechanistic_edges(tickers, mech_cfg, require_nonempty=False)
    if ei.shape[1] == 0:
        raise RuntimeError("mechanistic edge builder returned 0 edges")
    if single_graph:
        mech_full = precompute_top_neighbors(ei, ew, N, num_neighbors)  # [N, num_neighbors]
    else:
        mech_top = precompute_top_neighbors(ei, ew, N, half_k)          # [N, half_k]

    # (b) Per-cluster correlation top-half
    train_start = train_slice.start
    top_neighbors: dict[int, np.ndarray] = {}
    n_per_cluster: dict[int, int] = {}
    for c in range(K):
        if single_graph:
            top_neighbors[c] = mech_full
            n_per_cluster[c] = int((labels_train == c).sum())
            continue
        # Map cluster's train-indexed positions back to global day indices
        cluster_positions = np.where(labels_train == c)[0] + train_start
        cluster_positions = cluster_positions[cluster_positions < train_slice.stop]
        corr_top = _cluster_correlation_neighbors(
            log_returns, mask, cluster_positions, N, half_k
        )

        # Union mech top-half and corr top-half into `num_neighbors` slots
        combined = np.full((N, num_neighbors), -1, dtype=np.int64)
        for i in range(N):
            seen: set[int] = set()
            slot = 0
            # Mechanistic half first (structural prior always active)
            for j in mech_top[i]:
                if j < 0 or j in seen:
                    continue
                seen.add(int(j)); combined[i, slot] = j; slot += 1
                if slot >= num_neighbors:
                    break
            # Correlation half (regime-typed)
            for j in corr_top[i]:
                if slot >= num_neighbors:
                    break
                if j < 0 or j in seen:
                    continue
                seen.add(int(j)); combined[i, slot] = j; slot += 1
            # Pad with mech tail if still short
            if slot < num_neighbors:
                for j in mech_top[i]:
                    if slot >= num_neighbors:
                        break
                    if j < 0 or j in seen:
                        continue
                    seen.add(int(j)); combined[i, slot] = j; slot += 1

        top_neighbors[c] = combined
        n_per_cluster[c] = int((labels_train == c).sum())

    print(f"[catalog] K={K} cluster sizes: {n_per_cluster}")
    if single_graph:
        print(f"[catalog] single-graph mode: mechanistic edges={ei.shape[1]}, top-{num_neighbors} shared by all clusters")
    else:
        print(f"[catalog] mechanistic edges={ei.shape[1]}; half={half_k} mech + half={num_neighbors - half_k} corr per cluster")

    return RegimeCatalog(
        mu=mu, sd=sd, centroids=centroids, labels_train=labels_train,
        top_neighbors=top_neighbors, n_per_cluster=n_per_cluster,
        K=K, N=N, num_neighbors=num_neighbors,
    )


__all__ = ["RegimeCatalog", "build_catalog", "assign_days_to_cluster"]
