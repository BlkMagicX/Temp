"""Core mathematical modules for boundary drift analysis."""

from .anchor_scorer import AnchorScorer, AnchorScoringConfig
from .refusal_margin import compute_margin
from .boundary_metrics import (
    boundary_drift,
    flip_indicator,
    resolve_boundary_tau,
    select_boundary_near_safe,
)
from .sensitivity import (
    compute_margin_gradient,
    compute_kappa,
    compute_eta,
    compute_rho,
)
from .survival import (
    FirstLayerSurvivalConfig,
    build_small_delta,
    compute_survival_rate,
)

__all__ = [
    "AnchorScorer",
    "AnchorScoringConfig",
    "compute_margin",
    "boundary_drift",
    "flip_indicator",
    "resolve_boundary_tau",
    "select_boundary_near_safe",
    "compute_margin_gradient",
    "compute_kappa",
    "compute_eta",
    "compute_rho",
    "FirstLayerSurvivalConfig",
    "build_small_delta",
    "compute_survival_rate",
]
