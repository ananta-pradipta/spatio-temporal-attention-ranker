"""TGN memory module built on top of PyG primitives.

Thin wrapper over `torch_geometric.nn.TGNMemory` with two small additions
on top of the PyG reference:

  * explicit `detach_between_batches()` for the Rossi et al. (2020)
    discipline that prevents backprop flowing through the full history;
  * a helper that returns raw memory snapshots `s_i(t)` (the values we
    later store in the MTGN episodic KV cache), without extra projections.

PyG's `TGNMemory` already handles:
  - the message store / aggregation (last-message and mean aggregators),
  - the memory update via a GRU cell,
  - `last_update` tracking for time encoding.

This wrapper is intentionally thin: PyG is authoritative for the
Rossi-ordering edge cases (reset, detach, last_update) that are easy
to get wrong; only the novel MTGN components (episodic store, dual
attention) are implemented from scratch.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor
from torch.nn import GRUCell


try:
    from torch_geometric.nn.models.tgn import (
        IdentityMessage,
        LastAggregator,
        TGNMemory,
        TimeEncoder,
    )
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "torch_geometric is required. Install with `pip install torch-geometric`."
    ) from e


@dataclass
class MemoryConfig:
    num_nodes: int
    raw_msg_dim: int
    memory_dim: int = 128
    time_dim: int = 32


def build_tgn_memory(cfg: MemoryConfig) -> TGNMemory:
    """Construct a PyG TGNMemory with the MTGN default hyperparameters."""
    return TGNMemory(
        num_nodes=cfg.num_nodes,
        raw_msg_dim=cfg.raw_msg_dim,
        memory_dim=cfg.memory_dim,
        time_dim=cfg.time_dim,
        message_module=IdentityMessage(
            raw_msg_dim=cfg.raw_msg_dim, memory_dim=cfg.memory_dim, time_dim=cfg.time_dim
        ),
        aggregator_module=LastAggregator(),
    )


def detach_between_batches(memory: TGNMemory) -> None:
    """Detach memory tensors from the autograd graph between training batches.

    PyG's `reset_state()` clears the store but also zeroes memory; we only
    want to detach. The implementation mirrors PyG's internal detachment
    pattern used at the end of each message-passing step.
    """
    memory.memory.detach_()
    memory.last_update.detach_()


def snapshot(memory: TGNMemory, node_ids: Tensor) -> Tensor:
    """Return the raw memory vectors `s_i(t)` for the given node ids.

    Use these as key-value entries when writing to the MTGN episodic store.
    Projections (W_q, W_k, W_v) are applied at QUERY time, not write time,
    per the gating-policy addendum Section 3 (Option B).
    """
    mem, _ = memory(node_ids)
    return mem


__all__ = [
    "MemoryConfig",
    "build_tgn_memory",
    "detach_between_batches",
    "snapshot",
    "TGNMemory",
    "TimeEncoder",
    "GRUCell",
]
