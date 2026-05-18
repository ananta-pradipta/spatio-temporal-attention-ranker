"""G-InVAR: Graph-Guided Inverted VARiate Attention Ranker.

Universal cross-sectional equity ranker on the LATTICE S&P 500 panel.
Treats each stock as a variate token (iTransformer-style), summarizes
its 20-day feature history through a shared MLP, and learns cross-stock
interactions through inverted attention BIASED by point-in-time
financial graph priors.

Per the spec, graph priors are NOT a replacement for attention; they
are an economically-meaningful prior over the attention logits.

See docs/ginvar_design.md for the design rationale and the implementation
spec (paper title "Graph-Guided Inverted Attention for Spatio-Temporal
Equity Ranking").
"""
from src.lattice.models.ginvar.model import GInVAR, GInVARConfig
from src.lattice.models.ginvar.losses import cs_zscored_mse_loss

__all__ = ["GInVAR", "GInVARConfig", "cs_zscored_mse_loss"]
