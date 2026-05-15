"""Margin landscape analysis utilities.

This module supports:
- 1D scan m(z; alpha)
- optional 2D scan m(z; alpha, beta)
- finite-difference derivative/curvature estimates around alpha=0

Important:
- `mode="exact"` requires backend support for hidden-state intervention.
- Current default implementation uses `mode="approximate"` and scans in input
  space along surrogate directions.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .refusal_margin import compute_margin


@dataclass
class LandscapeScanConfig:
    mode: str = "approximate"  # exact | approximate
    alpha_max: float = 0.05
    alpha_steps: int = 21
    beta_max: float = 0.0
    beta_steps: int = 0
    aggregation: str = "logsumexp"


def make_linear_grid(max_abs: float, n: int) -> List[float]:
    if n <= 1:
        return [0.0]
    return np.linspace(-float(max_abs), float(max_abs), int(n), dtype=np.float32).tolist()


def make_alpha_grid_from_projection_quantiles(
    projections: Sequence[float],
    n_steps: int,
    q_low: float = 0.05,
    q_high: float = 0.95,
    fallback: Optional[Tuple[float, float]] = None,
) -> List[float]:
    """Build alpha grid from projection quantiles.

    Args:
        projections: scalar projection list p_l(z)=<d_l(z),u>.
        n_steps: number of alpha steps.
        q_low/q_high: quantile range used as [alpha_min, alpha_max].
        fallback: optional manual range when projections are unavailable.
    """
    n = max(2, int(n_steps))
    vals = np.asarray([float(x) for x in projections], dtype=np.float32) if projections else np.asarray([], dtype=np.float32)
    if vals.size > 0:
        lo = float(np.quantile(vals, q=float(q_low)))
        hi = float(np.quantile(vals, q=float(q_high)))
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            return np.linspace(lo, hi, n, dtype=np.float32).tolist()

    if fallback is None:
        return make_linear_grid(max_abs=0.05, n=n)

    lo, hi = float(fallback[0]), float(fallback[1])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return make_linear_grid(max_abs=0.05, n=n)
    return np.linspace(lo, hi, n, dtype=np.float32).tolist()


def _compute_margin_for_tensor(
    model: Any,
    image_tensor: torch.Tensor,
    prompt: str,
    refusal_anchors: Sequence[str],
    compliance_anchors: Sequence[str],
    aggregation: str,
    model_key: str,
) -> float:
    # Use wrapper's score interface to preserve existing scoring behavior.
    all_targets = [str(x) for x in list(refusal_anchors) + list(compliance_anchors)]
    batch_fn = getattr(model, "score_sequence_loglikelihood_batch", None)

    if batch_fn is not None:
        all_scores = [float(x) for x in batch_fn(image=image_tensor, prompt=prompt, target_texts=all_targets)]
        if len(all_scores) != len(all_targets):
            raise RuntimeError(
                "score_sequence_loglikelihood_batch returned unexpected length: " f"got={len(all_scores)}, expected={len(all_targets)}"
            )
        n_ref = len(refusal_anchors)
        refusal_scores = all_scores[:n_ref]
        compliance_scores = all_scores[n_ref:]
    else:
        refusal_scores = []
        for a in refusal_anchors:
            refusal_scores.append(float(model.score_sequence_loglikelihood(image=image_tensor, prompt=prompt, target_text=a)))
        compliance_scores = []
        for a in compliance_anchors:
            compliance_scores.append(float(model.score_sequence_loglikelihood(image=image_tensor, prompt=prompt, target_text=a)))

    del model_key
    return float(compute_margin(refusal_scores, compliance_scores, aggregation=aggregation))


def _compute_margin_for_tensor_exact_hidden(
    model: Any,
    image_tensor: torch.Tensor,
    prompt: str,
    refusal_anchors: Sequence[str],
    compliance_anchors: Sequence[str],
    aggregation: str,
    model_key: str,
    hidden_layer_index: int,
    delta_hidden: torch.Tensor,
    hidden_token_scope: str,
) -> float:
    score_fn = getattr(model, "score_sequence_loglikelihood_with_hidden_intervention", None)
    score_batch_fn = getattr(model, "score_sequence_loglikelihood_batch_with_hidden_intervention", None)
    if score_fn is None and score_batch_fn is None:
        raise RuntimeError(
            "Exact hidden-state intervention requires wrapper method "
            "`score_sequence_loglikelihood_with_hidden_intervention(...)`, but current model does not provide it."
        )

    all_targets = [str(x) for x in list(refusal_anchors) + list(compliance_anchors)]
    if score_batch_fn is not None:
        all_scores = [
            float(x)
            for x in score_batch_fn(
                image=image_tensor,
                prompt=prompt,
                target_texts=all_targets,
                layer_index=int(hidden_layer_index),
                delta_hidden=delta_hidden,
                token_scope=hidden_token_scope,
            )
        ]
        if len(all_scores) != len(all_targets):
            raise RuntimeError(
                "score_sequence_loglikelihood_batch_with_hidden_intervention returned unexpected length: "
                f"got={len(all_scores)}, expected={len(all_targets)}"
            )
        n_ref = len(refusal_anchors)
        refusal_scores = all_scores[:n_ref]
        compliance_scores = all_scores[n_ref:]
    else:
        refusal_scores = []
        for a in refusal_anchors:
            refusal_scores.append(
                float(
                    score_fn(
                        image=image_tensor,
                        prompt=prompt,
                        target_text=a,
                        layer_index=int(hidden_layer_index),
                        delta_hidden=delta_hidden,
                        token_scope=hidden_token_scope,
                    )
                )
            )
        compliance_scores = []
        for a in compliance_anchors:
            compliance_scores.append(
                float(
                    score_fn(
                        image=image_tensor,
                        prompt=prompt,
                        target_text=a,
                        layer_index=int(hidden_layer_index),
                        delta_hidden=delta_hidden,
                        token_scope=hidden_token_scope,
                    )
                )
            )

    del model_key
    return float(compute_margin(refusal_scores, compliance_scores, aggregation=aggregation))


def scan_landscape_1d(
    model: Any,
    base_image_tensor: torch.Tensor,
    prompt: str,
    direction_u: torch.Tensor,
    refusal_anchors: Sequence[str],
    compliance_anchors: Sequence[str],
    alphas: Iterable[float],
    aggregation: str = "logsumexp",
    model_key: str = "unknown",
    mode: str = "approximate",
    hidden_layer_index: Optional[int] = None,
    hidden_token_scope: str = "all",
    fp_model: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Scan 1D margin landscape along one direction.

    In current project this is implemented in approximate input-space mode.
    If `fp_model` is provided, also computes `margin_fp` at each point.

    Performance note: in `mode='exact'` the input image stays constant across all
    alpha steps (only the hidden-state delta changes). When the model wrapper
    exposes `prepare_score_inputs(...)` / `score_with_prepared(...)`, this
    function reuses the prepared inputs across the alpha loop to avoid running
    the processor (image preprocessing + tokenization) on every step.
    """
    mode_norm = str(mode).lower().strip()
    if mode_norm not in {"approximate", "exact"}:
        raise ValueError("mode must be one of: approximate/exact")

    x0 = base_image_tensor.detach()
    u = direction_u.detach()
    if float(u.norm(p=2).item()) > 0:
        u = u / u.norm(p=2)

    all_targets = [str(x) for x in list(refusal_anchors) + list(compliance_anchors)]
    n_ref = len(refusal_anchors)

    use_prepared_q = mode_norm == "exact" and hasattr(model, "prepare_score_inputs") and hasattr(model, "score_with_prepared")
    use_prepared_fp = (
        mode_norm == "exact" and fp_model is not None and hasattr(fp_model, "prepare_score_inputs") and hasattr(fp_model, "score_with_prepared")
    )

    prepared_q: Optional[Dict[str, Any]] = None
    prepared_fp: Optional[Dict[str, Any]] = None
    if use_prepared_q:
        prepared_q = model.prepare_score_inputs(image=x0, prompt=prompt, target_texts=all_targets)
    if use_prepared_fp:
        prepared_fp = fp_model.prepare_score_inputs(image=x0, prompt=prompt, target_texts=all_targets)

    out: List[Dict[str, Any]] = []
    for alpha in alphas:
        if mode_norm == "exact":
            if hidden_layer_index is None:
                raise ValueError("hidden_layer_index is required when mode='exact'")
            delta_hidden = float(alpha) * u
            intervention = (int(hidden_layer_index), delta_hidden, hidden_token_scope)

            if prepared_q is not None:
                scores_q = model.score_with_prepared(prepared=prepared_q, hidden_intervention=intervention)
                margin = float(compute_margin(scores_q[:n_ref], scores_q[n_ref:], aggregation=aggregation))
            else:
                margin = _compute_margin_for_tensor_exact_hidden(
                    model=model,
                    image_tensor=x0,
                    prompt=prompt,
                    refusal_anchors=refusal_anchors,
                    compliance_anchors=compliance_anchors,
                    aggregation=aggregation,
                    model_key=model_key,
                    hidden_layer_index=int(hidden_layer_index),
                    delta_hidden=delta_hidden,
                    hidden_token_scope=hidden_token_scope,
                )

            if fp_model is not None:
                if prepared_fp is not None:
                    scores_fp = fp_model.score_with_prepared(prepared=prepared_fp, hidden_intervention=intervention)
                    margin_fp = float(compute_margin(scores_fp[:n_ref], scores_fp[n_ref:], aggregation=aggregation))
                else:
                    margin_fp = _compute_margin_for_tensor_exact_hidden(
                        model=fp_model,
                        image_tensor=x0,
                        prompt=prompt,
                        refusal_anchors=refusal_anchors,
                        compliance_anchors=compliance_anchors,
                        aggregation=aggregation,
                        model_key="fp",
                        hidden_layer_index=int(hidden_layer_index),
                        delta_hidden=delta_hidden,
                        hidden_token_scope=hidden_token_scope,
                    )
            else:
                margin_fp = None
        else:
            x = torch.clamp(x0 + float(alpha) * u, 0.0, 1.0)
            margin = _compute_margin_for_tensor(
                model=model,
                image_tensor=x,
                prompt=prompt,
                refusal_anchors=refusal_anchors,
                compliance_anchors=compliance_anchors,
                aggregation=aggregation,
                model_key=model_key,
            )
            if fp_model is not None:
                margin_fp = _compute_margin_for_tensor(
                    model=fp_model,
                    image_tensor=x,
                    prompt=prompt,
                    refusal_anchors=refusal_anchors,
                    compliance_anchors=compliance_anchors,
                    aggregation=aggregation,
                    model_key="fp",
                )
            else:
                margin_fp = None
        item: Dict[str, Any] = {
            "alpha": float(alpha),
            "margin": float(margin),
            "mode": mode_norm,
        }
        if margin_fp is not None:
            item["margin_fp"] = float(margin_fp)
        out.append(item)
    return out


def scan_landscape_2d(
    model: Any,
    base_image_tensor: torch.Tensor,
    prompt: str,
    direction_u: torch.Tensor,
    direction_v: torch.Tensor,
    refusal_anchors: Sequence[str],
    compliance_anchors: Sequence[str],
    alphas: Iterable[float],
    betas: Iterable[float],
    aggregation: str = "logsumexp",
    model_key: str = "unknown",
    mode: str = "approximate",
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    hidden_layer_index: Optional[int] = None,
    hidden_token_scope: str = "all",
    fp_model: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Scan 2D margin landscape in approximate input space.

    If `fp_model` is provided, also computes `margin_fp` at each point.

    Performance note: identical fast path to `scan_landscape_1d` is used here for
    `mode='exact'` — image stays constant across the (alpha, beta) grid, so the
    processor outputs are prepared once per call and reused on every grid point.
    """
    mode_norm = str(mode).lower().strip()
    if mode_norm not in {"approximate", "exact"}:
        raise ValueError("mode must be one of: approximate/exact")

    x0 = base_image_tensor.detach()
    u = direction_u.detach()
    v = direction_v.detach()
    if float(u.norm(p=2).item()) > 0:
        u = u / u.norm(p=2)
    if float(v.norm(p=2).item()) > 0:
        v = v / v.norm(p=2)

    alpha_list = [float(a) for a in alphas]
    beta_list = [float(b) for b in betas]
    n_rows = len(alpha_list)
    n_cols = len(beta_list)
    total_points = max(1, n_rows * n_cols)
    t0 = time.perf_counter()

    all_targets = [str(x) for x in list(refusal_anchors) + list(compliance_anchors)]
    n_ref = len(refusal_anchors)

    use_prepared_q = mode_norm == "exact" and hasattr(model, "prepare_score_inputs") and hasattr(model, "score_with_prepared")
    use_prepared_fp = (
        mode_norm == "exact" and fp_model is not None and hasattr(fp_model, "prepare_score_inputs") and hasattr(fp_model, "score_with_prepared")
    )

    prepared_q: Optional[Dict[str, Any]] = None
    prepared_fp: Optional[Dict[str, Any]] = None
    if use_prepared_q:
        prepared_q = model.prepare_score_inputs(image=x0, prompt=prompt, target_texts=all_targets)
    if use_prepared_fp:
        prepared_fp = fp_model.prepare_score_inputs(image=x0, prompt=prompt, target_texts=all_targets)

    out: List[Dict[str, Any]] = []
    point_index = 0
    for row_idx, alpha in enumerate(alpha_list):
        for col_idx, beta in enumerate(beta_list):
            if mode_norm == "exact":
                if hidden_layer_index is None:
                    raise ValueError("hidden_layer_index is required when mode='exact'")
                delta_hidden = float(alpha) * u + float(beta) * v
                intervention = (int(hidden_layer_index), delta_hidden, hidden_token_scope)

                if prepared_q is not None:
                    scores_q = model.score_with_prepared(prepared=prepared_q, hidden_intervention=intervention)
                    margin = float(compute_margin(scores_q[:n_ref], scores_q[n_ref:], aggregation=aggregation))
                else:
                    margin = _compute_margin_for_tensor_exact_hidden(
                        model=model,
                        image_tensor=x0,
                        prompt=prompt,
                        refusal_anchors=refusal_anchors,
                        compliance_anchors=compliance_anchors,
                        aggregation=aggregation,
                        model_key=model_key,
                        hidden_layer_index=int(hidden_layer_index),
                        delta_hidden=delta_hidden,
                        hidden_token_scope=hidden_token_scope,
                    )

                if fp_model is not None:
                    if prepared_fp is not None:
                        scores_fp = fp_model.score_with_prepared(prepared=prepared_fp, hidden_intervention=intervention)
                        margin_fp = float(compute_margin(scores_fp[:n_ref], scores_fp[n_ref:], aggregation=aggregation))
                    else:
                        margin_fp = _compute_margin_for_tensor_exact_hidden(
                            model=fp_model,
                            image_tensor=x0,
                            prompt=prompt,
                            refusal_anchors=refusal_anchors,
                            compliance_anchors=compliance_anchors,
                            aggregation=aggregation,
                            model_key="fp",
                            hidden_layer_index=int(hidden_layer_index),
                            delta_hidden=delta_hidden,
                            hidden_token_scope=hidden_token_scope,
                        )
                else:
                    margin_fp = None
            else:
                x = torch.clamp(x0 + float(alpha) * u + float(beta) * v, 0.0, 1.0)
                margin = _compute_margin_for_tensor(
                    model=model,
                    image_tensor=x,
                    prompt=prompt,
                    refusal_anchors=refusal_anchors,
                    compliance_anchors=compliance_anchors,
                    aggregation=aggregation,
                    model_key=model_key,
                )
                if fp_model is not None:
                    margin_fp = _compute_margin_for_tensor(
                        model=fp_model,
                        image_tensor=x,
                        prompt=prompt,
                        refusal_anchors=refusal_anchors,
                        compliance_anchors=compliance_anchors,
                        aggregation=aggregation,
                        model_key="fp",
                    )
                else:
                    margin_fp = None
            item: Dict[str, Any] = {
                "alpha": float(alpha),
                "beta": float(beta),
                "margin": float(margin),
                "mode": mode_norm,
            }
            if margin_fp is not None:
                item["margin_fp"] = float(margin_fp)
            out.append(item)
            point_index += 1
            if progress_callback is not None:
                elapsed = time.perf_counter() - t0
                eta = None
                if point_index > 0:
                    eta = (elapsed / float(point_index)) * float(total_points - point_index)
                progress_callback(
                    {
                        "row_index": int(row_idx),
                        "col_index": int(col_idx),
                        "n_rows": int(n_rows),
                        "n_cols": int(n_cols),
                        "point_index": int(point_index),
                        "total_points": int(total_points),
                        "alpha": float(alpha),
                        "beta": float(beta),
                        "margin": float(margin),
                        "elapsed": float(elapsed),
                        "eta": (float(eta) if eta is not None else None),
                    }
                )
    return out


def estimate_derivative_and_curvature(scan_1d: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """Estimate first and second derivatives around alpha=0 by finite differences."""
    if not scan_1d:
        return {"directional_derivative": 0.0, "curvature": 0.0}

    points = sorted(scan_1d, key=lambda x: float(x["alpha"]))
    alphas = [float(p["alpha"]) for p in points]
    margins = [float(p["margin"]) for p in points]

    # find nearest around zero
    idx_zero = min(range(len(alphas)), key=lambda i: abs(alphas[i]))

    # derivative: central diff if possible
    if 0 < idx_zero < len(alphas) - 1:
        a_prev, a_next = alphas[idx_zero - 1], alphas[idx_zero + 1]
        m_prev, m_next = margins[idx_zero - 1], margins[idx_zero + 1]
        deriv = (m_next - m_prev) / (a_next - a_prev + 1e-12)
    elif idx_zero + 1 < len(alphas):
        deriv = (margins[idx_zero + 1] - margins[idx_zero]) / (alphas[idx_zero + 1] - alphas[idx_zero] + 1e-12)
    elif idx_zero - 1 >= 0:
        deriv = (margins[idx_zero] - margins[idx_zero - 1]) / (alphas[idx_zero] - alphas[idx_zero - 1] + 1e-12)
    else:
        deriv = 0.0

    # curvature: central second difference if possible
    if 0 < idx_zero < len(alphas) - 1:
        a0 = alphas[idx_zero]
        ap = alphas[idx_zero + 1]
        am = alphas[idx_zero - 1]
        h1 = a0 - am
        h2 = ap - a0
        m0 = margins[idx_zero]
        mp = margins[idx_zero + 1]
        mm = margins[idx_zero - 1]
        # generalized uneven-grid second derivative
        curvature = 2.0 * ((mp - m0) / (h2 + 1e-12) - (m0 - mm) / (h1 + 1e-12)) / (h1 + h2 + 1e-12)
    else:
        curvature = 0.0

    return {
        "directional_derivative": float(deriv),
        "curvature": float(curvature),
    }


def scan_hidden_direction_landscape_1d(
    fp_model: Any,
    q_model: Any,
    base_image_tensor: torch.Tensor,
    prompt: str,
    hidden_layer_index: int,
    direction_u: torch.Tensor,
    refusal_anchors: Sequence[str],
    compliance_anchors: Sequence[str],
    alphas: Iterable[float],
    aggregation: str = "logsumexp",
    hidden_token_scope: str = "all",
) -> List[Dict[str, Any]]:
    """Exact hidden-space 1D scan that reports fp/q margins and centered drops."""
    u = direction_u.detach().float()
    if u.dim() != 1:
        u = u.reshape(-1)
    if float(u.norm(p=2).item()) > 0:
        u = u / u.norm(p=2)

    x0 = base_image_tensor.detach()
    alpha_list = [float(a) for a in alphas]
    if not alpha_list:
        alpha_list = [0.0]

    margin_fp0: Optional[float] = None
    margin_q0: Optional[float] = None
    out: List[Dict[str, Any]] = []

    for alpha in alpha_list:
        delta_hidden = float(alpha) * u
        margin_q = _compute_margin_for_tensor_exact_hidden(
            model=q_model,
            image_tensor=x0,
            prompt=prompt,
            refusal_anchors=refusal_anchors,
            compliance_anchors=compliance_anchors,
            aggregation=aggregation,
            model_key="q",
            hidden_layer_index=int(hidden_layer_index),
            delta_hidden=delta_hidden,
            hidden_token_scope=hidden_token_scope,
        )
        margin_fp = _compute_margin_for_tensor_exact_hidden(
            model=fp_model,
            image_tensor=x0,
            prompt=prompt,
            refusal_anchors=refusal_anchors,
            compliance_anchors=compliance_anchors,
            aggregation=aggregation,
            model_key="fp",
            hidden_layer_index=int(hidden_layer_index),
            delta_hidden=delta_hidden,
            hidden_token_scope=hidden_token_scope,
        )

        if margin_q0 is None or abs(alpha) <= 1e-12:
            margin_q0 = float(margin_q)
        if margin_fp0 is None or abs(alpha) <= 1e-12:
            margin_fp0 = float(margin_fp)

        out.append(
            {
                "alpha": float(alpha),
                "margin_fp": float(margin_fp),
                "margin_q": float(margin_q),
                "gap_q_minus_fp": float(margin_q - margin_fp),
                "drop_fp": float(margin_fp - float(margin_fp0)),
                "drop_q": float(margin_q - float(margin_q0)),
            }
        )

    return out
