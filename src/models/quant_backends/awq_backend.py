"""AWQ backend interface layer (first runnable version)."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .base_quant_backend import BaseQuantBackend


class AWQBackend(BaseQuantBackend):
    """Quant backend adapter for AWQ-style quantized checkpoints.

    Design:
      - Treat AWQ as an inference-only backend.
      - Load an AWQ-exported checkpoint directly from `quant_model_path`.
      - Return the loaded HF model instance.
      - Processor is expected to be loaded by the wrapper from model/quant path.

    Notes:
      - This implementation does NOT fabricate a custom AWQ runtime.
      - It assumes the AWQ checkpoint can be loaded via HF `from_pretrained`.
      - If the target checkpoint requires a specialized runtime not supported by
        the installed environment, the real loader will fail with a clear error.
    """

    def __init__(self, precision_name: str = "w4a16") -> None:
        self._precision_name = precision_name

    def load_quantized_model(
        self,
        model_cls: Any,
        model_path: str,
        quant_model_path: Optional[str],
        device_map: Optional[Any],
        torch_dtype: Optional[Any],
        extra_config: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Load AWQ quantized model.

        Args:
            model_cls:
                The HF model class passed from wrapper, e.g.
                Qwen2VLForConditionalGeneration.
            model_path:
                Original full-precision model path/id. Used as fallback for
                processor/tokenizer resolution if needed.
            quant_model_path:
                Path or HF repo id of AWQ checkpoint.
            device_map:
                Device map for loading.
            torch_dtype:
                Optional dtype hint.
            extra_config:
                Optional backend config dictionary.

        Returns:
            Loaded quantized model instance.

        Raises:
            ValueError:
                If quant_model_path is missing.
            RuntimeError:
                If AWQ checkpoint loading fails.
        """
        extra_config = extra_config or {}

        if not quant_model_path:
            raise ValueError("AWQBackend requires `quant_model_path` for loading.")

        load_kwargs: Dict[str, Any] = {}

        if device_map is not None:
            load_kwargs["device_map"] = device_map

        # Some AWQ checkpoints can be loaded directly with HF. Keep dtype optional.
        if torch_dtype is not None:
            load_kwargs["torch_dtype"] = torch_dtype

        model = model_cls.from_pretrained(
            quant_model_path,
            **load_kwargs,
        )
        return model

    def get_precision_name(self) -> str:
        """Return configured precision label."""
        return self._precision_name

    def is_differentiable(self) -> bool:
        """AWQ eval path is treated as non-differentiable by default."""
        return False
