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


def classify_boundary_band(
    m_fp: float,
    ultra_near_upper: float = 0.05,
    sub_near_upper: float = 0.10,
    non_near_lower: float = 10.0,
) -> str:
    """Classify one sample into boundary-distance bands by m_fp.

    Bands:
      - 0 < m_fp <= ultra_near_upper: ultra_near
      - ultra_near_upper < m_fp <= sub_near_upper: sub_near
      - m_fp > non_near_lower: non_near
      - (sub_near_upper, non_near_lower]: mid_range (gap bucket)
      - m_fp <= 0: non_positive
    """
    val = float(m_fp)
    if val <= 0.0:
        return "non_positive"
    if val <= float(ultra_near_upper):
        return "ultra_near"
    if val <= float(sub_near_upper):
        return "sub_near"
    if val > float(non_near_lower):
        return "non_near"
    return "mid_range"


def classify_boundary_band_tau_scaled(
    m_fp: float,
    tau: float,
    ultra_near_upper: float = 0.05,
    sub_near_upper: float = 0.10,
    non_near_lower: float = 10.0,
) -> str:
    """Classify one sample using tau-scaled band thresholds.

    Effective thresholds are:
      - ultra_near: 0 < m_fp <= ultra_near_upper * tau
      - sub_near: ultra_near_upper * tau < m_fp <= sub_near_upper * tau
      - non_near: m_fp > non_near_lower * tau
      - mid_range: between sub_near_upper * tau and non_near_lower * tau
      - non_positive: m_fp <= 0
    """
    tau_val = float(tau)
    if tau_val <= 0.0:
        # Fall back to absolute thresholds if tau is invalid.
        return classify_boundary_band(
            m_fp=m_fp,
            ultra_near_upper=ultra_near_upper,
            sub_near_upper=sub_near_upper,
            non_near_lower=non_near_lower,
        )

    return classify_boundary_band(
        m_fp=m_fp,
        ultra_near_upper=float(ultra_near_upper) * tau_val,
        sub_near_upper=float(sub_near_upper) * tau_val,
        non_near_lower=float(non_near_lower) * tau_val,
    )


def classify_boundary_bands(
    m_fp_values: Iterable[float],
    ultra_near_upper: float = 0.05,
    sub_near_upper: float = 0.10,
    non_near_lower: float = 10.0,
) -> List[str]:
    """Batch version of boundary band classification."""
    out: List[str] = []
    for m in m_fp_values:
        out.append(
            classify_boundary_band(
                m_fp=float(m),
                ultra_near_upper=ultra_near_upper,
                sub_near_upper=sub_near_upper,
                non_near_lower=non_near_lower,
            )
        )
    return out


def classify_boundary_bands_tau_scaled(
    m_fp_values: Iterable[float],
    tau: float,
    ultra_near_upper: float = 0.05,
    sub_near_upper: float = 0.10,
    non_near_lower: float = 10.0,
) -> List[str]:
    """Batch version of tau-scaled boundary band classification."""
    out: List[str] = []
    for m in m_fp_values:
        out.append(
            classify_boundary_band_tau_scaled(
                m_fp=float(m),
                tau=float(tau),
                ultra_near_upper=ultra_near_upper,
                sub_near_upper=sub_near_upper,
                non_near_lower=non_near_lower,
            )
        )
    return out


def select_boundary_near_safe(m_fp_values: Iterable[float], tau: float) -> List[bool]:
    """Compute membership in B_safe_tau={z:0<m_fp(z)<=tau}."""
    out: List[bool] = []
    for m in m_fp_values:
        val = float(m)
        out.append(bool(0.0 < val <= float(tau)))
    return out
