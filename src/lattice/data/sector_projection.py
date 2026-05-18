"""Frozen one-shot sector projection for the Phase 5b novelty key.

Per Phase 5b spec section 5.2 and the user's ambiguity-4 resolution:
a fresh ``torch.nn.Linear(11, 1)`` initialised with ``xavier_uniform_`` at
``torch.manual_seed(0)``, frozen (``requires_grad_(False)``), saved to
``data/lattice/processed/sector_projection.pt`` and reloaded for every
fold. Pinned across folds so the projection is part of the data pipeline,
not the model state.

Avoids reusing the cohort embedding's sector axis: that axis is learned
and would couple the bank key to the model's training-time representation,
creating a subtle leakage pathway (retrieval similarity between train and
test entries would implicitly condition on the model's learned weights
rather than on a fixed projection of GICS).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn


SECTOR_PROJECTION_PATH = Path("data/lattice/processed/sector_projection.pt")
SECTOR_PROJECTION_SEED = 0
N_GICS_SECTORS = 11


def build_or_load_sector_projection(
    path: Path = SECTOR_PROJECTION_PATH,
    seed: int = SECTOR_PROJECTION_SEED,
) -> torch.Tensor:
    """Load or build the frozen sector projection weight matrix.

    Returns:
        Weight tensor of shape ``(1, n_sectors)``. Apply by
        ``proj @ one_hot(sector_id)``.
    """
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=False)
    torch.manual_seed(seed)
    layer = nn.Linear(N_GICS_SECTORS, 1, bias=False)
    nn.init.xavier_uniform_(layer.weight)
    weight = layer.weight.detach().clone()
    weight.requires_grad_(False)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(weight, path)
    return weight


def project_sector(weight: torch.Tensor, sector_ids: np.ndarray) -> np.ndarray:
    """Apply the projection to a vector of integer sector ids.

    Args:
        weight: ``(1, n_sectors)`` weight tensor from build_or_load.
        sector_ids: integer sector ids in [0, n_sectors). Negative entries
            (no sector) project to 0.

    Returns:
        ``(len(sector_ids),)`` projected scalars.
    """
    out = np.zeros(len(sector_ids), dtype=np.float32)
    w = weight.cpu().numpy().reshape(-1)
    for i, sid in enumerate(sector_ids):
        if 0 <= sid < N_GICS_SECTORS:
            out[i] = float(w[sid])
    return out


__all__ = [
    "SECTOR_PROJECTION_PATH",
    "SECTOR_PROJECTION_SEED",
    "N_GICS_SECTORS",
    "build_or_load_sector_projection",
    "project_sector",
]
