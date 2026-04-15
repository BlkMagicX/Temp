"""Refusal boundary margin computation helpers."""

from __future__ import annotations

from typing import Iterable

import torch


def aggregate_anchor_scores(scores: Iterable[float], method: str = "logsumexp") -> float:
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
        return float(torch.logsumexp(t, dim=0).item())
    if method_norm == "mean":
        return float(t.mean().item())
    if method_norm == "max":
        return float(t.max().item())

    raise ValueError(f"Unsupported aggregation method: {method}")


def compute_margin(
    refusal_scores: Iterable[float],
    compliance_scores: Iterable[float],
    aggregation: str = "logsumexp",
) -> float:
    """Compute refusal margin m_q(z)=S_ref_q(z)-S_comp_q(z)."""
    s_ref = aggregate_anchor_scores(refusal_scores, method=aggregation)
    s_comp = aggregate_anchor_scores(compliance_scores, method=aggregation)
    return s_ref - s_comp
