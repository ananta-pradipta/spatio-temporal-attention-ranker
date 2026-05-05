"""Event memory bank construction.

Iter 8: identify "event days" in training history (high-stress days
defined by financial criteria) and build a small memory bank of
event-typed regime fingerprints. At inference the gate / transformer
attends to this bank to retrieve historical event context.

Causal: event labels and bank are built from train-fold days only;
test-time retrieval is read-only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from src.investigation.regime_memory.signature import (
    compute_extended_signatures, forward_fill_signatures,
)


@dataclass
class EventMemoryBank:
    """Built once on train fold, then frozen."""
    fingerprint_mu: np.ndarray       # [F_fp] standardization mean (train-only)
    fingerprint_sd: np.ndarray       # [F_fp]
    centroids: np.ndarray            # [K, F_fp] event-day cluster centroids
    K: int


def build_event_fingerprints(log_returns: np.ndarray, mask: np.ndarray,
                             dates: list, risk_df: pd.DataFrame) -> np.ndarray:
    """Per-day fingerprint = [extended 7-dim signature, cs_mean, cs_std, cs_skew_today].
    All causal at day t (use signature at t which is causal by construction; cs
    moments use day-t cross-section, which is allowed since the gate reads
    them at prediction time, not the targets).

    Returns [T, 11] fingerprints (some early-day NaNs from windowed signature).
    """
    T, N = log_returns.shape
    sigs = compute_extended_signatures(log_returns, mask, dates, risk_df)  # [T, 7]
    sigs = forward_fill_signatures(sigs)

    cs_mean = np.full(T, np.nan, dtype=np.float64)
    cs_std = np.full(T, np.nan, dtype=np.float64)
    cs_skew = np.full(T, np.nan, dtype=np.float64)
    for t in range(T):
        m = mask[t]
        if m.sum() < 3:
            continue
        r = log_returns[t, m]
        cs_mean[t] = float(r.mean())
        cs_std[t] = float(r.std())
        cs_skew[t] = float(pd.Series(r).skew())
    cs = np.stack([cs_mean, cs_std, cs_skew], axis=1)  # [T, 3]
    fingerprints = np.concatenate([sigs, cs], axis=1)  # [T, 10]
    # Replace any remaining NaN with 0 (will be standardized away on event days)
    fingerprints = np.where(np.isfinite(fingerprints), fingerprints, 0.0)
    return fingerprints


def select_event_days(fingerprints: np.ndarray, train_slice: slice,
                      stress_quantile: float = 0.75) -> np.ndarray:
    """Boolean mask [T] marking event days. An event day is any train day
    above the `stress_quantile` of the train distribution on at least one of:
      - First-Principal-Component variance share (signature index 4)
      - XBI realized vol 60d (signature index 0)
      - cross-section dispersion (signature index 1)
    """
    T = fingerprints.shape[0]
    event_mask = np.zeros(T, dtype=bool)
    train_fp = fingerprints[train_slice]
    # Indices in fingerprint: sig 0..6 then [cs_mean=7, cs_std=8, cs_skew=9]
    for col in [4, 0, 1]:  # PC1 share, XBI vol, dispersion
        thresh = np.quantile(train_fp[:, col], stress_quantile)
        event_mask |= fingerprints[:, col] >= thresh
    # Restrict to train range
    full_event = np.zeros(T, dtype=bool)
    full_event[train_slice] = event_mask[train_slice]
    return full_event


def build_event_memory(fingerprints: np.ndarray, event_mask: np.ndarray,
                      train_slice: slice, K: int = 8,
                      kmeans_seed: int = 0) -> EventMemoryBank:
    """K-means on event-day fingerprints (train-only). Returns frozen bank."""
    train_fp = fingerprints[train_slice]
    train_event = event_mask[train_slice]
    # Standardize using train-fold statistics
    mu = train_fp.mean(axis=0)
    sd = train_fp.std(axis=0).clip(min=1e-6)
    z = (train_fp - mu) / sd
    z_events = z[train_event]
    if z_events.shape[0] < K:
        # Too few events; fall back to fewer clusters
        K = max(2, z_events.shape[0])
    km = KMeans(n_clusters=K, n_init=10, random_state=kmeans_seed)
    km.fit(z_events)
    print(f"[event-mem] {z_events.shape[0]} train event days  → K={K} clusters")
    return EventMemoryBank(
        fingerprint_mu=mu, fingerprint_sd=sd,
        centroids=km.cluster_centers_, K=K,
    )


def event_similarity_weights(fingerprint_t: np.ndarray, bank: EventMemoryBank,
                             temperature: float = 1.0) -> np.ndarray:
    """At day t, return softmax similarity weights over the K event clusters.
    fingerprint_t is the raw (un-z-scored) day-t fingerprint."""
    z = (fingerprint_t - bank.fingerprint_mu) / bank.fingerprint_sd
    d2 = ((z[None, :] - bank.centroids) ** 2).sum(axis=1)  # [K]
    return np.exp(-d2 / temperature) / np.exp(-d2 / temperature).sum()


__all__ = ["build_event_fingerprints", "select_event_days", "build_event_memory",
           "event_similarity_weights", "EventMemoryBank"]
