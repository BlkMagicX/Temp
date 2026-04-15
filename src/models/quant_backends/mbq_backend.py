"""MBQ backend interface layer (first version placeholder)."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .base_quant_backend import BaseQuantBackend


class MBQBackend(BaseQuantBackend):
    """Quant backend adapter for MBQ-style quantized checkpoints.

    Current status:
      - Interface and parameter validation are implemented.
      - Real MBQ load path is TODO and intentionally not fabricated.
      - Optional placeholder fallback can load by HF `from_pretrained` for
        structure testing only (not real MBQ quant inference).
    """

    def __init__(self, precision_name: str = "w8a8") -> None:
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
        """Load MBQ quantized model.

        Required parameters (for real backend implementation):
          - `quant_model_path`: path/id to MBQ-exported checkpoint
          - backend-specific MBQ runtime package and loader API

        Placeholder behavior:
          - If `extra_config.enable_placeholder_load` is True, fallback to
            generic HF loading from `quant_model_path or model_path`.
          - Otherwise raises `NotImplementedError` with clear TODO guidance.
        """
        cfg = extra_config or {}

        if not quant_model_path:
            raise ValueError("MBQBackend requires `quant_model_path` for real quantized loading. " "Please provide path/id of MBQ-exported weights.")

        if cfg.get("enable_placeholder_load", False):
            load_path = quant_model_path or model_path
            return model_cls.from_pretrained(load_path, device_map=device_map, torch_dtype=torch_dtype)

        raise NotImplementedError(
            "MBQ backend real loading is TODO.\n"
            "Needed next steps:\n"
            "1) Integrate MBQ runtime package and loader entrypoint.\n"
            "2) Map `quant_model_path` and quant config to loader args.\n"
            "3) Verify generation parity and memory footprint."
        )

    def get_precision_name(self) -> str:
        """Return configured precision label."""
        return self._precision_name

    def is_differentiable(self) -> bool:
        """MBQ path is considered non-differentiable by default."""
        return False
