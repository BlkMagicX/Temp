"""GPTQ backend interface layer (strict, real loading only)."""

from __future__ import annotations

from typing import Any, Dict, Optional

from transformers import AutoConfig

from .base_quant_backend import BaseQuantBackend


class GPTQBackend(BaseQuantBackend):
    """Quant backend adapter for GPTQ-style quantized checkpoints.

    Design:
      - Treat GPTQ as an inference-only backend.
      - Load a GPTQ-exported checkpoint directly from `quant_model_path`.
      - Fail fast if checkpoint metadata does not look like GPTQ.

    Notes:
      - This backend does not provide any placeholder loading.
      - If your environment cannot load GPTQ checkpoints, it will raise a clear error.
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
        """Load GPTQ quantized model.

        Args:
            model_cls: HF model class from wrapper.
            model_path: Base model id (unused here, kept for interface parity).
            quant_model_path: GPTQ checkpoint path or HF repo id.
            device_map: Device map for loading.
            torch_dtype: Optional dtype hint.
            extra_config: Backend-specific options.

        Raises:
            ValueError: If quant_model_path is missing.
            RuntimeError: If checkpoint does not look like GPTQ or loading fails.
        """
        extra_config = extra_config or {}

        if not quant_model_path:
            raise ValueError("GPTQBackend requires `quant_model_path` for loading.")

        strict_gptq_check = bool(extra_config.get("strict_gptq_check", True))
        allow_non_gptq_checkpoint = bool(extra_config.get("allow_non_gptq_checkpoint", False))

        if strict_gptq_check:
            cfg = AutoConfig.from_pretrained(quant_model_path)
            qcfg = getattr(cfg, "quantization_config", None)

            quant_method = None
            if isinstance(qcfg, dict):
                quant_method = qcfg.get("quant_method")
            elif qcfg is not None:
                quant_method = getattr(qcfg, "quant_method", None)

            is_gptq = str(quant_method or "").lower() == "gptq"
            if not is_gptq and not allow_non_gptq_checkpoint:
                raise RuntimeError(
                    "Configured quant_model_path does not look like a GPTQ checkpoint. "
                    f"quant_method={quant_method!r}, quant_model_path={quant_model_path}. "
                    "Set quant_backend_config.allow_non_gptq_checkpoint=true only if you know this is expected."
                )

        load_kwargs: Dict[str, Any] = {
            "low_cpu_mem_usage": True,
        }

        if device_map is not None:
            load_kwargs["device_map"] = device_map

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
        """GPTQ eval path is treated as non-differentiable by default."""
        return False
