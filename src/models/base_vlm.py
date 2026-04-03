"""Unified abstract interface for Vision-Language Models used in this project."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch


class BaseVLM(ABC):
    """Abstract base class for all VLM backends.

    The interface explicitly separates:
    1) Text generation (`generate_response`) for normal evaluation.
    2) Differentiable forward (`forward_for_loss`) for white-box attacks.
    """

    @abstractmethod
    def load_model(self) -> None:
        """Load model/processor/tokenizer resources into memory.

        Implementations should be idempotent so repeated calls are safe.
        """

    @abstractmethod
    def preprocess_example(self, image: Any, prompt: str) -> Dict[str, Any]:
        """Preprocess one multimodal example for generation.

        Args:
            image: Input image object, usually PIL image or torch tensor.
            prompt: User prompt text.

        Returns:
            A dictionary containing model-ready inputs.
        """

    @abstractmethod
    def generate_response(
        self,
        image: Any,
        prompt: str,
        generation_config: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate model text response for one image-prompt pair.

        Args:
            image: Input image object.
            prompt: User prompt text.
            generation_config: Optional generation arguments.

        Returns:
            Generated text string.
        """

    @abstractmethod
    def forward_for_loss(
        self,
        image_tensor: torch.Tensor,
        prompt: str,
        target_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run differentiable forward pass used by attacks.

        Args:
            image_tensor: Preprocessed image tensor expected by model visual encoder,
                typically shape `(B, C, H, W)`. This tensor should support gradient
                flow (e.g., `requires_grad=True`) for white-box attacks.
            prompt: Prompt text used to build multimodal context.
            target_text: Optional attack target text. If provided, labels are built
                from this target; otherwise labels are derived from the prompt path
                according to backend policy.

        Returns:
            Dictionary with at least:
            - `loss`: scalar loss tensor or `None`
            - `logits`: model logits tensor
            - `input_ids`: text token ids
            - `labels`: supervised labels or `None`
            Additional backend-specific fields are allowed.
        """

    @abstractmethod
    def get_device(self) -> torch.device:
        """Return current device of the loaded model."""
