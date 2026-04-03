"""Visual-only PGD attack for VLM white-box optimization."""

from __future__ import annotations

import time
import importlib
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from PIL import Image

from .base_attack import AttackOutput, BaseAttack

_loss_module = importlib.import_module("src.attacks.losses")
compose_attack_loss = getattr(_loss_module, "compose_attack_loss")


class VisualPGDAttack(BaseAttack):
    """PGD-style visual attack for VLM jailbreak/harmful generation.

    Key design:
      - Only image pixels are optimized.
      - Prompt text is passed through unchanged on every iteration.
      - Loss uses VLM token-level objective rather than classifier logits.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        self.steps = int(cfg.get("steps", 20))
        self.alpha = float(cfg.get("alpha", 1.0 / 255.0))
        self.eps = float(cfg.get("eps", 8.0 / 255.0))
        self.norm = str(cfg.get("norm", "linf")).lower()
        self.random_start = bool(cfg.get("random_start", True))

        self.use_targeted_loss = bool(cfg.get("use_targeted_loss", True))
        self.use_negative_refusal = bool(cfg.get("use_negative_refusal", False))
        self.negative_refusal_weight = float(cfg.get("negative_refusal_weight", 0.0))

        if self.norm not in {"linf", "l2"}:
            raise ValueError("`norm` must be one of {'linf', 'l2'}")

    def attack(self, sample: Any, model: Any) -> AttackOutput:
        """Generate adversarial perturbation for one sample.

        Args:
            sample: Sample object/dict containing `image_path`, `user_prompt`,
                optional `target_text`.
            model: White-box VLM wrapper exposing `forward_for_loss(...)` and
                `get_device()`.

        Returns:
            `AttackOutput` with adversarial image tensor, delta, losses, norms.
        """
        prompt = self._get_sample_field(sample, "user_prompt")
        target_text = self._get_sample_field(sample, "target_text", default=None)

        # IMPORTANT: we never modify prompt text inside the attack loop.
        # This guarantees pure visual perturbation attack setting.
        x_orig = self._load_sample_image_tensor(sample, device=model.get_device())

        if self.random_start:
            delta = self._random_init_delta_like(x_orig)
        else:
            delta = torch.zeros_like(x_orig)

        delta = self.project(delta, x_orig).detach().requires_grad_(True)

        loss_history = []
        step_losses = []
        start_t = time.perf_counter()

        for step_idx in range(self.steps):
            x_adv = torch.clamp(x_orig + delta, 0.0, 1.0)

            # Keep gradient path explicitly through image tensor.
            outputs = model.forward_for_loss(
                image_tensor=x_adv,
                prompt=prompt,
                target_text=target_text,
            )

            loss_terms = self.build_loss(
                model_outputs=outputs,
                prompt=prompt,
                target_text=target_text,
                step_idx=step_idx,
            )
            total_loss = loss_terms["total_loss"]

            grad = torch.autograd.grad(total_loss, delta, retain_graph=False, create_graph=False)[0]

            if self.norm == "linf":
                delta = delta.detach() + self.alpha * grad.sign()
            else:
                g = grad.detach()
                g_flat = g.view(g.shape[0], -1)
                g_norm = g_flat.norm(p=2, dim=1).view(-1, 1, 1, 1).clamp(min=1e-12)
                delta = delta.detach() + self.alpha * (g / g_norm)

            delta = self.project(delta, x_orig).detach().requires_grad_(True)

            loss_history.append(float(total_loss.detach().item()))
            step_losses.append({k: float(v.detach().item()) for k, v in loss_terms.items()})

        runtime = float(time.perf_counter() - start_t)
        x_adv_final = torch.clamp(x_orig + delta.detach(), 0.0, 1.0)
        final_delta = (x_adv_final - x_orig).detach()

        final_delta_linf = float(final_delta.abs().max().item())
        final_delta_l2 = float(final_delta.view(final_delta.shape[0], -1).norm(p=2, dim=1).mean().item())

        return AttackOutput(
            adv_image=self.tensor_to_pil(x_adv_final),
            adv_image_tensor=x_adv_final.detach(),
            delta=final_delta,
            loss_history=loss_history,
            step_losses=step_losses,
            final_delta_linf=final_delta_linf,
            final_delta_l2=final_delta_l2,
            attack_runtime=runtime,
            meta={
                "attack_name": "visual_pgd",
                "steps": self.steps,
                "alpha": self.alpha,
                "eps": self.eps,
                "norm": self.norm,
            },
        )

    def build_loss(
        self,
        model_outputs: Dict[str, Any],
        prompt: str,
        target_text: Optional[str],
        step_idx: int,
    ) -> Dict[str, torch.Tensor]:
        """Build loss dictionary for one attack step.

        We currently rely on token-level targeted objective and keep refusal term
        as an optional extension hook.
        """
        del prompt, target_text, step_idx
        return compose_attack_loss(
            model_outputs=model_outputs,
            use_targeted_loss=self.use_targeted_loss,
            use_negative_refusal=self.use_negative_refusal,
            negative_refusal_weight=self.negative_refusal_weight,
        )

    def project(self, delta: torch.Tensor, x_orig: torch.Tensor) -> torch.Tensor:
        """Project perturbation into constraint set and legal image range."""
        if self.norm == "linf":
            delta_proj = torch.clamp(delta, min=-self.eps, max=self.eps)
        else:
            # l2-ball projection
            d = delta.view(delta.shape[0], -1)
            d_norm = d.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
            factor = torch.clamp(self.eps / d_norm, max=1.0)
            delta_proj = (d * factor).view_as(delta)

        x_adv = torch.clamp(x_orig + delta_proj, 0.0, 1.0)
        return x_adv - x_orig

    @staticmethod
    def tensor_to_pil(tensor_bchw: torch.Tensor) -> Image.Image:
        """Convert BCHW/BCHW(1) float tensor in [0,1] to PIL RGB image."""
        if tensor_bchw.dim() == 4:
            if tensor_bchw.shape[0] != 1:
                raise ValueError("tensor_to_pil expects batch size 1")
            tensor = tensor_bchw[0]
        elif tensor_bchw.dim() == 3:
            tensor = tensor_bchw
        else:
            raise ValueError("Expected CHW or BCHW tensor")

        tensor = torch.clamp(tensor.detach().cpu(), 0.0, 1.0)
        array = (tensor * 255.0).byte().permute(1, 2, 0).numpy()
        return Image.fromarray(array, mode="RGB")

    def _load_sample_image_tensor(self, sample: Any, device: torch.device) -> torch.Tensor:
        """Load sample image and convert to BCHW tensor in [0, 1]."""
        if isinstance(sample, dict):
            image_path = sample["image_path"]
        else:
            image_path = sample.image_path

        if hasattr(sample, "resolve_image_path"):
            resolved = sample.resolve_image_path()
        else:
            resolved = Path(image_path).resolve()

        image = Image.open(resolved).convert("RGB")
        tensor = self._pil_to_unit_tensor(image).unsqueeze(0).to(device)
        return tensor

    def _random_init_delta_like(self, x_orig: torch.Tensor) -> torch.Tensor:
        """Initialize random perturbation inside the configured threat model."""
        if self.norm == "linf":
            return torch.empty_like(x_orig).uniform_(-self.eps, self.eps)

        # l2 random init: sample from normal then project to l2 ball.
        d = torch.randn_like(x_orig)
        d_flat = d.view(d.shape[0], -1)
        d_norm = d_flat.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
        unit = d_flat / d_norm
        radius = torch.rand(d.shape[0], 1, device=x_orig.device)
        scaled = unit * (radius * self.eps)
        return scaled.view_as(x_orig)

    @staticmethod
    def _pil_to_unit_tensor(image: Image.Image) -> torch.Tensor:
        """Convert PIL RGB image to CHW float tensor in [0, 1]."""
        import numpy as np

        img = image.convert("RGB")
        arr = np.asarray(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        return tensor

    @staticmethod
    def _get_sample_field(sample: Any, key: str, default: Any = None) -> Any:
        """Get field from sample object or dict without mutating sample."""
        if isinstance(sample, dict):
            return sample.get(key, default)
        return getattr(sample, key, default)
