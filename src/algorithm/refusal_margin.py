"""Refusal boundary margin computation helpers."""

from __future__ import annotations

import math
from typing import Iterable

import torch


def _resolve_anchor_tau(values: Iterable[float], mode: str, fixed_tau: float, quantile: float) -> float:
    mode_norm = str(mode).lower().strip()
    if mode_norm == "fixed":
        return float(fixed_tau)
    if mode_norm != "quantile":
        raise ValueError(f"Unsupported tau mode: {mode}")

    safe_vals = []
    for v in values:
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(fv):
            safe_vals.append(abs(fv))
    if not safe_vals:
        return float(fixed_tau)

    q = min(max(float(quantile), 0.0), 1.0)
    t = torch.tensor(safe_vals, dtype=torch.float32)
    tau_val = float(torch.quantile(t, q=q).item())
    if tau_val <= 0.0:
        return float(fixed_tau)
    return tau_val


def aggregate_anchor_scores(
    scores: Iterable[float],
    method: str = "logsumexp",
    tau_mode: str = "fixed",
    tau_fixed: float = 1.0,
    tau_quantile: float = 0.2,
) -> float:
    """Aggregate anchor scores into one scalar.

    Supported methods:
        - logsumexp
        - mean
        - max
    """
    values = list(scores)
    if not values:
        return 0.0

    method_norm = method.lower().strip()
    t = torch.tensor(values, dtype=torch.float32)

    if method_norm == "logsumexp":
        tau_val = _resolve_anchor_tau(values, mode=tau_mode, fixed_tau=tau_fixed, quantile=tau_quantile)
        if tau_val <= 0.0:
            tau_val = 1.0
        scaled = t / float(tau_val)
        return float((float(tau_val) * (torch.logsumexp(scaled, dim=0) - math.log(float(len(values))))).item())
    if method_norm == "mean":
        return float(t.mean().item())
    if method_norm == "max":
        return float(t.max().item())

    raise ValueError(f"Unsupported aggregation method: {method}")


def compute_margin(
    refusal_scores: Iterable[float],
    compliance_scores: Iterable[float],
    aggregation: str = "logsumexp",
    tau_mode: str = "fixed",
    tau_fixed: float = 1.0,
    tau_quantile: float = 0.2,
) -> float:
    """Compute refusal margin m_q(z)=S_ref_q(z)-S_comp_q(z)."""
    s_ref = aggregate_anchor_scores(
        refusal_scores,
        method=aggregation,
        tau_mode=tau_mode,
        tau_fixed=tau_fixed,
        tau_quantile=tau_quantile,
    )
    s_comp = aggregate_anchor_scores(
        compliance_scores,
        method=aggregation,
        tau_mode=tau_mode,
        tau_fixed=tau_fixed,
        tau_quantile=tau_quantile,
    )
    return s_ref - s_comp
