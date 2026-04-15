"""Boundary drift and flip indicator metrics."""

from __future__ import annotations

from typing import Iterable, List, Literal

import numpy as np


def boundary_drift(m_q: float, m_fp: float) -> float:
    """Compute delta_q(z)=m_q(z)-m_fp(z)."""
    return float(m_q - m_fp)


def flip_indicator(m_fp: float, m_q: float) -> int:
    """Compute F_q(z)=1[m_fp(z)>0 and m_q(z)<0]."""
    return 1 if (m_fp > 0.0 and m_q < 0.0) else 0


def resolve_boundary_tau(
    m_fp_values: Iterable[float],
    mode: Literal["fixed", "quantile"] = "fixed",
    fixed_tau: float = 1.0,
    quantile: float = 0.2,
) -> float:
    """Resolve tau using fixed threshold or positive-margin quantile."""
    if mode == "fixed":
        return float(fixed_tau)

    if mode != "quantile":
        raise ValueError(f"Unsupported tau mode: {mode}")

    positives = [float(x) for x in m_fp_values if float(x) > 0.0]
    if not positives:
        return float(fixed_tau)

    q = min(max(float(quantile), 0.0), 1.0)
    return float(np.quantile(np.array(positives, dtype=np.float32), q=q))


def select_boundary_near_safe(m_fp_values: Iterable[float], tau: float) -> List[bool]:
    """Compute membership in B_safe_tau={z:0<m_fp(z)<=tau}."""
    out: List[bool] = []
    for m in m_fp_values:
        val = float(m)
        out.append(bool(0.0 < val <= float(tau)))
    return out
