"""Qwen2.5-VL wrapper implementing the unified VLM interface."""

from __future__ import annotations

from typing import Any, Dict

from transformers import AutoProcessor
from transformers import Qwen2_5_VLForConditionalGeneration

from .qwen2vl_wrapper import Qwen2VLWrapper


class Qwen25VLWrapper(Qwen2VLWrapper):
    """Unified wrapper for Qwen2.5-VL models.

    This wrapper reuses the Qwen2-VL pipeline logic and only switches the
    underlying model class to `Qwen2_5_VLForConditionalGeneration`.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        cfg = dict(config)
        if not cfg.get("model_path") and not cfg.get("model_name_or_path"):
            cfg["model_path"] = "Qwen/Qwen2.5-VL-7B-Instruct"
        super().__init__(cfg)

    def load_model(self) -> None:
        """Load processor and Qwen2.5-VL model according to `precision_mode`."""
        if self.model is not None and self.processor is not None:
            return

        self.processor = AutoProcessor.from_pretrained(self.processor_path)

        if self.backend_type == "bf16" or self.precision_mode == "bf16":
            load_kwargs: Dict[str, Any] = {}
            if self.torch_dtype is not None:
                load_kwargs["torch_dtype"] = self.torch_dtype
            if self.device_map is not None:
                load_kwargs["device_map"] = self.device_map

            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(self.model_path, **load_kwargs)
            if self.device_map is None:
                self.model.to(self.device)
            self.model.eval()
            return

        self.model = self._load_quantized_model()
        if self.device_map is None:
            self.model.to(self.device)
            if self.torch_dtype is not None:
                self.model.to(dtype=self.torch_dtype)
        self.model.eval()

    def _load_quantized_model(self):
        """Load quantized Qwen2.5-VL model via selected backend adapter."""
        self.quant_backend = self._build_quant_backend()
        return self.quant_backend.load_quantized_model(
            model_cls=Qwen2_5_VLForConditionalGeneration,
            model_path=self.model_path,
            quant_model_path=self.quant_model_path,
            device_map=self.device_map,
            torch_dtype=self.torch_dtype,
            extra_config=self.quant_backend_config,
        )
