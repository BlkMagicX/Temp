"""Dangerous direction extraction from representation drifts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch


@dataclass
class DangerousDirectionConfig:
    """Configuration for dangerous-direction extraction."""

    top_k: int = 3
    center: bool = True
    max_samples_per_layer: Optional[int] = None


def fit_dangerous_directions(
    drifts_by_layer: Dict[int, np.ndarray],
    cfg: DangerousDirectionConfig,
) -> Dict[int, Dict[str, Any]]:
    """Fit principal drift directions per layer using SVD/PCA.

    Args:
        drifts_by_layer: layer -> [N, D] drift matrix
        cfg: extraction config

    Returns:
        layer -> {
            "mean": [D],
            "directions": [K, D],
            "explained_variance_ratio": [K],
            "singular_values": [K],
            "n_samples": int,
        }
    """
    out: Dict[int, Dict[str, Any]] = {}

    for layer_idx, mat in sorted(drifts_by_layer.items()):
        if mat is None or mat.size == 0:
            continue

        x = np.asarray(mat, dtype=np.float32)
        if x.ndim != 2 or x.shape[0] < 2:
            continue

        if cfg.max_samples_per_layer is not None and x.shape[0] > int(cfg.max_samples_per_layer):
            x = x[: int(cfg.max_samples_per_layer)]

        mu = np.mean(x, axis=0) if cfg.center else np.zeros((x.shape[1],), dtype=np.float32)
        xc = x - mu

        # SVD: xc = U * S * Vt
        _, s, vt = np.linalg.svd(xc, full_matrices=False)

        k = min(int(cfg.top_k), vt.shape[0])
        dirs = vt[:k]

        s2 = s**2
        denom = float(np.sum(s2)) + 1e-12
        evr = (s2[:k] / denom).astype(np.float32)

        out[int(layer_idx)] = {
            "mean": mu.astype(np.float32),
            "directions": dirs.astype(np.float32),
            "explained_variance_ratio": evr,
            "singular_values": s[:k].astype(np.float32),
            "n_samples": int(x.shape[0]),
        }

    return out


def compute_alignment_and_push(
    drift_vec: np.ndarray,
    direction_vec: np.ndarray,
    eps: float = 1e-12,
) -> Dict[str, float]:
    """Compute alignment and push scalar for one drift vector.

    alignment = cos(d, u)
    push = <d, u>
    """
    d = np.asarray(drift_vec, dtype=np.float32)
    u = np.asarray(direction_vec, dtype=np.float32)

    d_norm = float(np.linalg.norm(d, ord=2))
    u_norm = float(np.linalg.norm(u, ord=2))
    denom = d_norm * u_norm + float(eps)
    if denom <= 0.0:
        return {"alignment": 0.0, "push": 0.0}

    push = float(np.dot(d, u))
    align = float(push / denom)
    return {
        "alignment": align,
        "push": push,
    }


def save_dangerous_directions(
    directions_by_layer: Dict[int, Dict[str, Any]],
    output_dir: str | Path,
    prefix: str = "dangerous_directions",
) -> List[Path]:
    """Save per-layer dangerous directions to disk (.pt files)."""
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: List[Path] = []
    for layer_idx, payload in sorted(directions_by_layer.items()):
        path = out_dir / f"{prefix}.layer_{layer_idx}.pt"
        torch_payload = {
            "layer_index": int(layer_idx),
            "mean": torch.from_numpy(np.asarray(payload["mean"], dtype=np.float32)),
            "directions": torch.from_numpy(np.asarray(payload["directions"], dtype=np.float32)),
            "explained_variance_ratio": torch.from_numpy(np.asarray(payload["explained_variance_ratio"], dtype=np.float32)),
            "singular_values": torch.from_numpy(np.asarray(payload["singular_values"], dtype=np.float32)),
            "n_samples": int(payload.get("n_samples", 0)),
        }
        torch.save(torch_payload, path)
        paths.append(path)

    return paths


def load_dangerous_directions(path: str | Path) -> Dict[str, Any]:
    """Load one saved dangerous-direction file."""
    payload = torch.load(Path(path).resolve(), map_location="cpu")
    return {
        "layer_index": int(payload["layer_index"]),
        "mean": payload["mean"].detach().cpu().numpy(),
        "directions": payload["directions"].detach().cpu().numpy(),
        "explained_variance_ratio": payload["explained_variance_ratio"].detach().cpu().numpy(),
        "singular_values": payload["singular_values"].detach().cpu().numpy(),
        "n_samples": int(payload.get("n_samples", 0)),
    }
