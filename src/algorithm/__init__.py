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
from .representation_drift import (
    RepresentationDriftConfig,
    extract_layer_representations,
    compute_sample_layer_drift,
    compute_representation_drift_dataset,
    summarize_drift_rows,
)
from .dangerous_direction import (
    DangerousDirectionConfig,
    fit_dangerous_directions,
    compute_alignment_and_push,
    save_dangerous_directions,
    load_dangerous_directions,
)
from .landscape import (
    LandscapeScanConfig,
    make_linear_grid,
    make_alpha_grid_from_projection_quantiles,
    scan_landscape_1d,
    scan_landscape_2d,
    scan_hidden_direction_landscape_1d,
    estimate_derivative_and_curvature,
)
from .qcsd_direction import (
    load_drift_subspace,
    compute_margin_gradient_surrogate,
    compute_qcsd_direction,
    compare_directions,
    apply_qcsd_projection_correction,
)
from .pressure_test import (
    PressureAttackConfig,
    apply_pair_prompt,
    apply_typography_overlay,
    run_attack_family_on_sample,
)
from .defense_boundary_calibration import (
    DirectionalBoundaryCalibrationConfig,
    DirectionalBoundaryCalibration,
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
    "RepresentationDriftConfig",
    "extract_layer_representations",
    "compute_sample_layer_drift",
    "compute_representation_drift_dataset",
    "summarize_drift_rows",
    "DangerousDirectionConfig",
    "fit_dangerous_directions",
    "compute_alignment_and_push",
    "save_dangerous_directions",
    "load_dangerous_directions",
    "LandscapeScanConfig",
    "make_linear_grid",
    "make_alpha_grid_from_projection_quantiles",
    "scan_landscape_1d",
    "scan_landscape_2d",
    "scan_hidden_direction_landscape_1d",
    "estimate_derivative_and_curvature",
    "load_drift_subspace",
    "compute_margin_gradient_surrogate",
    "compute_qcsd_direction",
    "compare_directions",
    "apply_qcsd_projection_correction",
    "PressureAttackConfig",
    "apply_pair_prompt",
    "apply_typography_overlay",
    "run_attack_family_on_sample",
    "DirectionalBoundaryCalibrationConfig",
    "DirectionalBoundaryCalibration",
]
