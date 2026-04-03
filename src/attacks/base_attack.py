"""Base abstractions for visual-only white-box attacks on VLMs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch


@dataclass
class AttackOutput:
    """Container for attack artifacts.

    Fields are intentionally explicit to simplify downstream saving/logging.
    """

    adv_image: Any
    adv_image_tensor: torch.Tensor
    delta: torch.Tensor
    loss_history: List[float] = field(default_factory=list)
    step_losses: List[Dict[str, float]] = field(default_factory=list)
    final_delta_linf: float = 0.0
    final_delta_l2: float = 0.0
    attack_runtime: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)


class BaseAttack(ABC):
    """Abstract interface for visual-only attacks.

    This interface is designed for VLM generation/token-loss settings, not
    classification-only pipelines.
    """

    @abstractmethod
    def attack(self, sample: Any, model: Any) -> AttackOutput:
        """Run attack for one sample on one white-box model.

        Implementations must optimize only image pixels and must not modify the
        user prompt text.
        """

    @abstractmethod
    def build_loss(
        self,
        model_outputs: Dict[str, Any],
        prompt: str,
        target_text: Optional[str],
        step_idx: int,
    ) -> Dict[str, torch.Tensor]:
        """Build differentiable loss terms from model outputs.

        Returns a dictionary containing at least `total_loss`.
        """

    @abstractmethod
    def project(self, delta: torch.Tensor, x_orig: torch.Tensor) -> torch.Tensor:
        """Project perturbation onto constraint set and valid pixel domain."""
