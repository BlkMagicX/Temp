"""Abstract interface for quantized model loading backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch


class BaseQuantBackend(ABC):
    """Base abstraction for quantized backend integrations.

    A backend is responsible for instantiating a quantized model from user
    configuration and declaring whether it supports differentiable forward.
    """

    @abstractmethod
    def load_quantized_model(
        self,
        model_cls: Any,
        model_path: str,
        quant_model_path: Optional[str],
        device_map: Optional[Any],
        torch_dtype: Optional[torch.dtype],
        extra_config: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Load quantized model instance.

        Args:
            model_cls: Base HF model class used for loading.
            model_path: Original base model path/id.
            quant_model_path: Quantized checkpoint path/id.
            device_map: Device mapping passed to loader when supported.
            torch_dtype: Torch dtype hint.
            extra_config: Backend-specific options.
        """

    @abstractmethod
    def get_precision_name(self) -> str:
        """Return backend precision name, such as `w8a8` or `w4a16`."""

    @abstractmethod
    def is_differentiable(self) -> bool:
        """Whether backend supports gradient-based white-box optimization."""
