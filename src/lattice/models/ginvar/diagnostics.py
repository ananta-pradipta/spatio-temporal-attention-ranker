"""G-InVAR per-batch diagnostics (graph weights, attention entropy, etc.).

Stub; populated alongside the regime-aware graph blender in step 8.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GraphDiagnosticRow:
    """One day's worth of graph + attention diagnostics."""

    date: str
    fold: int
    seed: int
    w_corr: float = 0.0
    w_sector: float = 0.0
    w_factor: float = 0.0
    w_social: float = 0.0
    w_beta: float = 0.0
    avg_degree: float = 0.0
    avg_graph_entropy: float = 0.0
    avg_attention_entropy: float = 0.0


__all__ = ["GraphDiagnosticRow"]
