"""Directional boundary calibration defense (minimal runnable baseline)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class DirectionalBoundaryCalibrationConfig:
    """Config for lightweight quantization-aware boundary calibration."""

    mode: str = "approximate"  # exact | approximate
    lambda_proj: float = 0.0
    lambda_delta: float = 1.0
    lambda_bias: float = 0.0
    risk_margin_threshold: float = 1.0
    only_high_risk: bool = True


class DirectionalBoundaryCalibration:
    """Minimal defense module for quantized safety-margin restoration.

    Exact hidden-state editing may not be available in current wrappers.
    Therefore this module provides an output-side approximate calibration:

        m_hat_q = m_q + b(z)

    where b(z) depends on risk indicators such as negative drift and optional
    dangerous-direction score proxy.
    """

    def __init__(self, config: DirectionalBoundaryCalibrationConfig) -> None:
        self.config = config

    def _is_high_risk(
        self,
        m_q: float,
        delta_q: float,
        dangerous_score: Optional[float],
    ) -> bool:
        if not self.config.only_high_risk:
            return True

        cond_margin = float(m_q) <= float(self.config.risk_margin_threshold)
        cond_delta = float(delta_q) < 0.0
        cond_dir = dangerous_score is not None and float(dangerous_score) > 0.0
        return bool(cond_margin or cond_delta or cond_dir)

    def calibration_bias(
        self,
        m_q: float,
        delta_q: float,
        dangerous_score: Optional[float] = None,
    ) -> float:
        if not self._is_high_risk(m_q=m_q, delta_q=delta_q, dangerous_score=dangerous_score):
            return 0.0

        # Restore margin when quantization causes negative drift.
        neg_delta = max(0.0, -float(delta_q))
        dir_term = max(0.0, float(dangerous_score)) if dangerous_score is not None else 0.0

        bias = float(self.config.lambda_bias)
        bias += float(self.config.lambda_delta) * neg_delta
        bias += float(self.config.lambda_proj) * dir_term
        return float(bias)

    def calibrate_margin(
        self,
        m_q: float,
        delta_q: float,
        dangerous_score: Optional[float] = None,
    ) -> float:
        b = self.calibration_bias(m_q=m_q, delta_q=delta_q, dangerous_score=dangerous_score)
        return float(m_q + b)

    def calibrate_rows(self, rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for row in rows:
            m_q = row.get("m_q")
            delta_q = row.get("delta_q")
            if m_q is None or delta_q is None:
                out.append(dict(row))
                continue

            dangerous_score = row.get("dangerous_score")
            m_hat = self.calibrate_margin(
                m_q=float(m_q),
                delta_q=float(delta_q),
                dangerous_score=(float(dangerous_score) if dangerous_score is not None else None),
            )

            new_row = dict(row)
            new_row["m_q_defended"] = float(m_hat)
            new_row["margin_restoration"] = float(m_hat - float(m_q))
            out.append(new_row)
        return out
