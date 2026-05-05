"""Episodic Factor Repricing (EFR) for EFR-DGraph-epiSTAR.

Per spec ``efr_dgraph_epistar_implementation_prompt.md`` Sections
4 + 5 + 6. Three components:

1. EpisodicFactorRepricingMemory: per-training-day ridge OLS coefficients
   theta_tau = argmin ||z(y_tau) - B_tau theta||^2 + eta||theta||^2,
   stored alongside the regime key for that day. Retrieval is top-K
   regime-cosine plus reliability weighting.

2. FactorRepricingGate: deterministic + learnable lambda_efr in
   [0.05, 0.30] based on stress + rebound.

3. FactorRepricingHead: applies score_total = z(score_ow) +
   lambda_efr * z(score_factor) per day.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class EFRConfig:
    """Hyperparameters for EFR."""

    n_retrieved_days: int = 16
    ridge_eta: float = 0.25
    retrieval_temperature: float = 0.20
    horizon_days: int = 5
    embargo_days: int = 5
    min_lambda: float = 0.05
    max_lambda: float = 0.30


def ridge_solve(b_train: np.ndarray, y_z: np.ndarray, eta: float) -> np.ndarray:
    """Ridge OLS theta = (B^T B + eta I)^-1 B^T y.

    Args:
        b_train: [N, K] factor exposures (cross-sectionally standardised).
        y_z: [N] cross-sectionally z-scored target.
        eta: ridge regularisation.

    Returns:
        theta: [K] coefficient vector.
    """
    n, k = b_train.shape
    if n < k + 2 or y_z.std() < 1e-9:
        return np.zeros(k, dtype=np.float32)
    bt_b = b_train.T @ b_train + eta * np.eye(k, dtype=np.float64)
    try:
        theta = np.linalg.solve(bt_b, b_train.T @ y_z)
    except np.linalg.LinAlgError:
        return np.zeros(k, dtype=np.float32)
    return theta.astype(np.float32)


class EpisodicFactorRepricingMemory(nn.Module):
    """Stores (date, regime_key, theta_tau, factor_ic, n_active) per
    training day; retrieves top-K regime-similar entries at query time."""

    def __init__(self, cfg: EFRConfig, regime_dim: int, n_factors: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.regime_dim = regime_dim
        self.n_factors = n_factors
        self.register_buffer("regime_keys", torch.zeros(0, regime_dim))
        self.register_buffer("thetas", torch.zeros(0, n_factors))
        self.register_buffer("factor_ics", torch.zeros(0))
        self.register_buffer("n_active_arr", torch.zeros(0, dtype=torch.long))
        self.register_buffer("day_indices", torch.zeros(0, dtype=torch.long))

    def populate(
        self,
        train_idx: np.ndarray,
        factor_exposures: np.ndarray,    # [T, N, K]
        regime_keys_arr: np.ndarray,     # [T, R]
        y_true: np.ndarray,              # [T, N]
        loss_mask: np.ndarray,           # [T, N]
    ) -> None:
        """Build per-training-day theta + factor_ic + regime_key index."""
        keys: list = []; thetas: list = []; ics: list = []
        ns: list = []; days: list = []
        for t in train_idx:
            m = loss_mask[t]
            if m.sum() < self.n_factors + 2:
                continue
            b = factor_exposures[t, m]                     # [n_active, K]
            y = y_true[t, m]
            if y.std() < 1e-9:
                continue
            y_z = (y - y.mean()) / (y.std() + 1e-9)
            theta = ridge_solve(b.astype(np.float64), y_z.astype(np.float64), self.cfg.ridge_eta)
            # Factor IC = corr(B @ theta, y_z).
            score_factor = b @ theta
            if score_factor.std() < 1e-9:
                continue
            ic = float(np.corrcoef(score_factor, y_z)[0, 1])
            keys.append(regime_keys_arr[t])
            thetas.append(theta)
            ics.append(ic)
            ns.append(int(m.sum()))
            days.append(int(t))
        if not keys:
            print("[EFR] WARN no training days qualified for memory")
            return
        keys_a = np.asarray(keys, dtype=np.float32)
        thetas_a = np.asarray(thetas, dtype=np.float32)
        ics_a = np.asarray(ics, dtype=np.float32)
        ns_a = np.asarray(ns, dtype=np.int64)
        days_a = np.asarray(days, dtype=np.int64)
        self.regime_keys.data = torch.from_numpy(keys_a)
        self.thetas.data = torch.from_numpy(thetas_a)
        self.factor_ics.data = torch.from_numpy(ics_a)
        self.n_active_arr.data = torch.from_numpy(ns_a)
        self.day_indices.data = torch.from_numpy(days_a)

    def retrieve_theta(
        self, regime_key_t: Tensor, query_day_idx: int,
        use_positive_factor_ic_only: bool = True,
    ) -> tuple[Tensor, dict]:
        """Top-K regime-cosine + reliability-weighted theta + confidence.

        Per DOW-EFR spec Section 6: the retrieval also returns a
        scalar `confidence` in [0, 1] computed from
        weighted-mean-reliability + weighted-mean-similarity +
        theta-agreement.

        When `use_positive_factor_ic_only=True` (DOW-EFR default), the
        reliability is `clip(max(factor_ic, 0), 0, 0.10)`. Negatively
        correlated historical portfolios get reliability 0.
        """
        cfg = self.cfg
        device = regime_key_t.device
        zero_theta = torch.zeros(self.n_factors, device=device)
        zero_conf = torch.zeros((), device=device)
        empty_diag = {
            "n_eligible": 0, "top1_sim": zero_conf,
            "retrieved_days": torch.zeros(cfg.n_retrieved_days, dtype=torch.long, device=device) - 1,
            "confidence": zero_conf,
            "mean_reliability": zero_conf,
            "mean_similarity": zero_conf,
            "theta_agreement": zero_conf,
            "retrieved_factor_ics": torch.zeros(cfg.n_retrieved_days, device=device),
            "retrieved_weights": torch.zeros(cfg.n_retrieved_days, device=device),
        }
        if self.regime_keys.numel() == 0:
            return zero_theta, empty_diag
        cutoff = query_day_idx - cfg.horizon_days - cfg.embargo_days
        eligible_mask = self.day_indices < cutoff
        if eligible_mask.sum() == 0:
            return zero_theta, empty_diag

        keys = self.regime_keys[eligible_mask].to(device)
        thetas = self.thetas[eligible_mask].to(device)
        ics = self.factor_ics[eligible_mask].to(device)
        days = self.day_indices[eligible_mask].to(device)

        q = regime_key_t / (regime_key_t.norm(p=2) + 1e-8)
        k_norm = keys / (keys.norm(p=2, dim=-1, keepdim=True) + 1e-8)
        sims = (k_norm @ q).squeeze(-1)
        m = min(cfg.n_retrieved_days, sims.shape[0])
        idx = torch.topk(sims, k=m, largest=True).indices
        sel_thetas = thetas[idx]
        sel_sims = sims[idx]
        sel_ics = ics[idx]
        sel_days = days[idx]

        if use_positive_factor_ic_only:
            rel = torch.clamp(torch.clamp(sel_ics, min=0.0), max=0.10)
        else:
            rel = torch.clamp(sel_ics.abs(), min=0.01, max=0.10)

        # If all reliabilities are zero, return theta=0 and confidence=0.
        if float(rel.sum().item()) < 1e-6:
            theta_t = zero_theta
            w = torch.zeros(m, device=device)
            mean_rel = zero_conf
            mean_sim = sel_sims.mean().detach()
            agreement = zero_conf
            confidence = zero_conf
        else:
            w_raw = torch.exp(sel_sims / cfg.retrieval_temperature) * rel
            w = w_raw / (w_raw.sum() + 1e-8)
            theta_t = (w.unsqueeze(-1) * sel_thetas).sum(dim=0)
            mean_rel = (w * rel).sum().detach()
            mean_sim = (w * sel_sims).sum().detach()
            # Pairwise cosine agreement among retrieved thetas.
            theta_norm = sel_thetas / (sel_thetas.norm(p=2, dim=-1, keepdim=True) + 1e-8)
            cos_mat = theta_norm @ theta_norm.T
            n_off = m * (m - 1)
            if n_off > 0:
                off_diag_sum = float(cos_mat.sum().item()) - float(cos_mat.diag().sum().item())
                mean_pairwise_cos = off_diag_sum / float(n_off)
            else:
                mean_pairwise_cos = 0.0
            agreement = torch.tensor(
                max(0.0, mean_pairwise_cos), dtype=torch.float32, device=device,
            )
            confidence_logit = (
                3.0 * mean_rel + 1.0 * mean_sim + 1.0 * agreement - 1.5
            )
            confidence = torch.sigmoid(confidence_logit).clamp(0.0, 1.0)

        if sel_days.shape[0] < cfg.n_retrieved_days:
            pad = cfg.n_retrieved_days - sel_days.shape[0]
            sel_days = torch.cat(
                [sel_days, torch.full((pad,), -1, dtype=torch.long, device=device)],
                dim=0,
            )
            rel = torch.cat([rel, torch.zeros(pad, device=device)], dim=0)
            w = torch.cat([w, torch.zeros(pad, device=device)], dim=0)
            sel_ics = torch.cat([sel_ics, torch.zeros(pad, device=device)], dim=0)

        return theta_t, {
            "n_eligible": int(eligible_mask.sum().item()),
            "top1_sim": sel_sims[0].detach() if sel_sims.numel() > 0 else zero_conf,
            "retrieved_days": sel_days,
            "confidence": confidence,
            "mean_reliability": mean_rel,
            "mean_similarity": mean_sim,
            "theta_agreement": agreement,
            "retrieved_factor_ics": sel_ics.detach(),
            "retrieved_weights": w.detach(),
        }


class FactorRepricingGate(nn.Module):
    """Deterministic + learnable scalar lambda_efr in [min, max]."""

    def __init__(
        self, regime_dim: int = 12, min_lambda: float = 0.05, max_lambda: float = 0.30,
    ) -> None:
        super().__init__()
        self.min_lambda = min_lambda
        self.max_lambda = max_lambda
        # Deterministic init per spec: a=0.5 (stress), b=0.7 (rebound), c=-1.0.
        # We learn small refinements but keep the deterministic backbone.
        self.a = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor(0.7, dtype=torch.float32))
        self.c = nn.Parameter(torch.tensor(-1.0, dtype=torch.float32))

    @staticmethod
    def stress_and_rebound(regime_features_t: Tensor) -> tuple[Tensor, Tensor]:
        """Per spec Section 5. regime_features_t is [12]:
        0=pc1_share_21d, 1=avg_pairwise_corr_60d, 2=dispersion_5d,
        3=market_return_5d, 4=xbi_ret_5d, 5=xbi_ret_20d,
        ...
        """
        pc1 = regime_features_t[0]
        avg_corr = regime_features_t[1]
        disp = regime_features_t[2]
        mkt_5d = regime_features_t[3]
        xbi_5d = regime_features_t[4]
        xbi_20d = regime_features_t[5]
        stress = pc1 + avg_corr + disp + xbi_20d.abs()
        rebound = mkt_5d + xbi_5d - xbi_20d
        return stress, rebound

    def forward(self, regime_features_t: Tensor) -> Tensor:
        stress, rebound = self.stress_and_rebound(regime_features_t)
        raw = self.a * stress + self.b * rebound + self.c
        s = torch.sigmoid(raw)
        return self.min_lambda + (self.max_lambda - self.min_lambda) * s


def cs_zscore(x: Tensor, mask: Tensor, eps: float = 1e-6) -> Tensor:
    """Cross-sectional z-score over masked entries; zeros elsewhere."""
    m = mask.to(x.dtype)
    n = m.sum().clamp(min=1.0)
    mu = (x * m).sum() / n
    var = ((x - mu) * m).pow(2).sum() / n
    sd = var.clamp(min=1e-12).sqrt()
    z = (x - mu) / sd.clamp(min=eps)
    return z * m


__all__ = [
    "EFRConfig",
    "EpisodicFactorRepricingMemory",
    "FactorRepricingGate",
    "ridge_solve",
    "cs_zscore",
]
