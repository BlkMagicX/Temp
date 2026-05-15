"""Representation drift extraction utilities.

This module provides minimal-intrusion hidden-state drift extraction for
post-training quantization analysis.

Design notes:
- Uses existing model wrappers and forward path (no model rewrite).
- Works when wrappers expose `model` and `preprocess_example`.
- For non-torch backends (for example vLLM), extraction is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image


@dataclass
class RepresentationDriftConfig:
    """Configuration for representation drift extraction."""

    layer_indices: Sequence[int]
    pooling: str = "mean"  # mean | cls | last_token
    max_samples: Optional[int] = None


def load_image_pil(sample: Any) -> Image.Image:
    """Load sample image as RGB PIL image.

    Expected sample fields:
    - image_path (absolute path in this project)
    """
    image_path = getattr(sample, "image_path", None)
    if image_path is None and isinstance(sample, dict):
        image_path = sample.get("image_path")
    if image_path is None:
        raise ValueError("Sample has no image_path")

    p = Path(str(image_path)).resolve()
    return Image.open(p).convert("RGB")


@torch.inference_mode()
def extract_layer_representations(
    model_wrapper: Any,
    image: Image.Image,
    prompt: str,
    layer_indices: Sequence[int],
    pooling: str = "mean",
) -> Dict[int, np.ndarray]:
    """Extract pooled hidden representations from selected layers.

    Returns:
        Dict[layer_index, vector_np]

    Raises:
        RuntimeError if backend does not expose hidden states.
    """
    model_wrapper.load_model()

    model = getattr(model_wrapper, "model", None)
    if model is None:
        raise RuntimeError("Representation extraction requires torch model backend; current backend has no model attribute")

    model_inputs = model_wrapper.preprocess_example(image=image, prompt=prompt)
    model_inputs["output_hidden_states"] = True
    model_inputs["return_dict"] = True

    outputs = model(**model_inputs)
    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is None:
        raise RuntimeError("Model outputs do not contain hidden_states")

    n_layers = len(hidden_states)
    reps: Dict[int, np.ndarray] = {}

    pooling_norm = str(pooling).lower().strip()
    for requested_idx in layer_indices:
        idx = int(requested_idx)
        if idx < 0:
            idx = n_layers + idx
        if idx < 0 or idx >= n_layers:
            continue

        h = hidden_states[idx]  # [B, T, H]
        if h.dim() != 3:
            continue

        if pooling_norm == "cls":
            vec = h[:, 0, :]
        elif pooling_norm == "last_token":
            vec = h[:, -1, :]
        else:
            vec = h.mean(dim=1)

        reps[int(requested_idx)] = vec[0].detach().float().cpu().numpy()

    return reps


def _l2_norm(vec: np.ndarray) -> float:
    return float(np.linalg.norm(vec, ord=2))


def compute_sample_layer_drift(
    fp_reps: Dict[int, np.ndarray],
    q_reps: Dict[int, np.ndarray],
    eps: float = 1e-12,
) -> List[Dict[str, Any]]:
    """Compute per-layer drift stats for one sample."""
    rows: List[Dict[str, Any]] = []
    keys = sorted(set(fp_reps.keys()) & set(q_reps.keys()))
    for li in keys:
        fp = fp_reps[li]
        q = q_reps[li]
        d = q - fp

        fp_norm = _l2_norm(fp)
        d_norm = _l2_norm(d)
        r = d_norm / (fp_norm + float(eps))

        rows.append(
            {
                "layer_index": int(li),
                "fp_norm_l2": fp_norm,
                "drift_norm_l2": d_norm,
                "normalized_drift": float(r),
                "drift_vector": d,
                "fp_vector": fp,
                "q_vector": q,
            }
        )
    return rows


def compute_representation_drift_dataset(
    samples: Sequence[Any],
    fp_model: Any,
    q_model: Any,
    cfg: RepresentationDriftConfig,
    eps: float = 1e-12,
) -> Dict[str, Any]:
    """Compute representation drift for a sample set.

    Returns dictionary contains:
    - rows: list[dict] with per-sample per-layer stats
    - drifts_by_layer: dict[layer_index, np.ndarray [N, D]]
    - sample_ids_by_layer: dict[layer_index, list[str]]
    """
    all_rows: List[Dict[str, Any]] = []
    drifts_by_layer: Dict[int, List[np.ndarray]] = {}
    sample_ids_by_layer: Dict[int, List[str]] = {}

    max_samples = cfg.max_samples if cfg.max_samples is not None else len(samples)

    for i, sample in enumerate(samples[: int(max_samples)]):
        sample_id = getattr(sample, "sample_id", None)
        if sample_id is None and isinstance(sample, dict):
            sample_id = str(sample.get("sample_id", f"sample_{i}"))
        sample_id = str(sample_id)

        prompt = getattr(sample, "prompt", None)
        if prompt is None and isinstance(sample, dict):
            prompt = sample.get("prompt")
        if prompt is None:
            continue

        image = load_image_pil(sample)

        try:
            fp_reps = extract_layer_representations(
                model_wrapper=fp_model,
                image=image,
                prompt=str(prompt),
                layer_indices=cfg.layer_indices,
                pooling=cfg.pooling,
            )
            q_reps = extract_layer_representations(
                model_wrapper=q_model,
                image=image,
                prompt=str(prompt),
                layer_indices=cfg.layer_indices,
                pooling=cfg.pooling,
            )
        except RuntimeError:
            # Backend does not support hidden-state extraction.
            continue

        layer_rows = compute_sample_layer_drift(fp_reps=fp_reps, q_reps=q_reps, eps=eps)
        for row in layer_rows:
            li = int(row["layer_index"])
            drift_vec = row.pop("drift_vector")
            row.pop("fp_vector", None)
            row.pop("q_vector", None)

            drifts_by_layer.setdefault(li, []).append(drift_vec)
            sample_ids_by_layer.setdefault(li, []).append(sample_id)

            out_row = {
                "sample_id": sample_id,
                **row,
            }
            all_rows.append(out_row)

    stacked_drifts: Dict[int, np.ndarray] = {}
    for li, vecs in drifts_by_layer.items():
        if not vecs:
            continue
        stacked_drifts[li] = np.stack(vecs, axis=0)

    return {
        "rows": all_rows,
        "drifts_by_layer": stacked_drifts,
        "sample_ids_by_layer": sample_ids_by_layer,
    }


def summarize_drift_rows(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Aggregate per-layer drift rows into summary rows."""
    by_layer: Dict[int, List[Dict[str, Any]]] = {}
    for r in rows:
        li = int(r["layer_index"])
        by_layer.setdefault(li, []).append(r)

    out: List[Dict[str, Any]] = []
    for li, items in sorted(by_layer.items()):
        d_vals = [float(x["drift_norm_l2"]) for x in items]
        n_vals = [float(x["normalized_drift"]) for x in items]
        fp_vals = [float(x["fp_norm_l2"]) for x in items]
        out.append(
            {
                "layer_index": li,
                "n_samples": len(items),
                "drift_norm_l2_mean": float(np.mean(d_vals)) if d_vals else 0.0,
                "drift_norm_l2_std": float(np.std(d_vals)) if d_vals else 0.0,
                "normalized_drift_mean": float(np.mean(n_vals)) if n_vals else 0.0,
                "normalized_drift_std": float(np.std(n_vals)) if n_vals else 0.0,
                "fp_norm_l2_mean": float(np.mean(fp_vals)) if fp_vals else 0.0,
            }
        )
    return out
