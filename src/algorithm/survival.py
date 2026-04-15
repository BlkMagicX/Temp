"""First-layer perturbation survival metric (approximate implementation)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch


@dataclass
class FirstLayerSurvivalConfig:
    """Configuration for SR metric and tiny perturbation generation."""

    delta_type: str = "random"
    delta_budget: float = 1.0 / 255.0
    eps: float = 1e-8


def _tensor_to_pil(image_tensor: torch.Tensor) -> Any:
    from PIL import Image
    import numpy as np

    x = image_tensor
    if x.dim() == 4:
        if x.shape[0] != 1:
            raise ValueError("Only batch size 1 is supported")
        x = x[0]
    x = torch.clamp(x.detach().cpu(), 0.0, 1.0)
    arr = (x.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _extract_observable_phi(model: Any, image_tensor: torch.Tensor, prompt: str) -> torch.Tensor:
    """Extract an observable representation close to early visual features.

    Approximation note:
        We use processor output pixel_values as a practical proxy for phi(z)
        because current backends do not expose true first-layer pre/post-quant
        tensors in a unified way.
    """
    if not hasattr(model, "preprocess_example"):
        raise RuntimeError("Model must implement preprocess_example for SR approximation")

    image = _tensor_to_pil(image_tensor)
    batch = model.preprocess_example(image=image, prompt=prompt)
    pixel_values = batch.get("pixel_values")
    if pixel_values is None:
        raise RuntimeError("preprocess_example did not return pixel_values")
    return pixel_values.detach().float()


def _parse_activation_bits(precision_name: str, default_bits: int = 16) -> int:
    norm = precision_name.lower().strip()
    if "a" not in norm:
        return default_bits

    try:
        after = norm.split("a", 1)[1]
        digits = ""
        for ch in after:
            if ch.isdigit():
                digits += ch
            else:
                break
        if digits:
            return max(1, int(digits))
    except Exception:
        return default_bits
    return default_bits


def _fake_quantize_activation(x: torch.Tensor, bits: int) -> torch.Tensor:
    """Uniform fake quantization used as Q_q(.) approximation for SR."""
    if bits >= 16:
        return x

    x_min = x.min()
    x_max = x.max()
    span = (x_max - x_min).clamp(min=1e-8)
    levels = float((2**bits) - 1)
    scaled = (x - x_min) / span
    q = torch.round(scaled * levels) / levels
    return q * span + x_min


def build_small_delta(
    image_tensor: torch.Tensor,
    cfg: FirstLayerSurvivalConfig,
    margin_grad: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Build a tiny perturbation delta for SR evaluation."""
    budget = float(cfg.delta_budget)
    if cfg.delta_type == "random":
        return torch.empty_like(image_tensor).uniform_(-budget, budget)

    if cfg.delta_type == "sign_noise":
        noise = torch.sign(torch.randn_like(image_tensor))
        return noise * budget

    if cfg.delta_type == "fgsm":
        if margin_grad is None:
            raise ValueError("FGSM delta requires margin_grad")
        return torch.sign(margin_grad) * budget

    raise ValueError(f"Unsupported delta_type: {cfg.delta_type}")


def compute_survival_rate(
    model: Any,
    precision_name: str,
    image_tensor: torch.Tensor,
    prompt: str,
    delta: torch.Tensor,
    eps: float = 1e-8,
) -> float:
    """Compute SR_q(z,delta) with approximate first-layer observable tensors.

    SR_q = ||Q_q(phi(z+delta)) - Q_q(phi(z))||_2 / (||phi(z+delta)-phi(z)||_2 + eps)
    """
    x = torch.clamp(image_tensor.detach(), 0.0, 1.0)
    x_pert = torch.clamp(x + delta.detach(), 0.0, 1.0)

    phi_x = _extract_observable_phi(model, x, prompt)
    phi_xp = _extract_observable_phi(model, x_pert, prompt)

    bits = _parse_activation_bits(precision_name)
    q_phi_x = _fake_quantize_activation(phi_x, bits=bits)
    q_phi_xp = _fake_quantize_activation(phi_xp, bits=bits)

    numerator = (q_phi_xp - q_phi_x).view(-1).norm(p=2)
    denominator = (phi_xp - phi_x).view(-1).norm(p=2) + float(eps)

    return float((numerator / denominator).item())
