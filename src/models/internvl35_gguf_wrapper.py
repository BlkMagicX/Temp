"""InternVL3.5 GGUF wrapper using llama.cpp backend."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from huggingface_hub import hf_hub_download

from .base_vlm import BaseVLM


class InternVL35GGUFWrapper(BaseVLM):
    """GGUF-backed InternVL3.5 wrapper using llama.cpp via llama-cpp-python.

    This backend supports text generation for evaluation and does not support
    differentiable forward passes.
    """

    DEFAULT_REPO_ID = "bartowski/OpenGVLab_InternVL3_5-1B-GGUF"
    DEFAULT_MMPROJ_FILENAME = "mmproj-OpenGVLab_InternVL3_5-1B-f16.gguf"

    PRECISION_TO_FILENAME = {
        "bf16": "OpenGVLab_InternVL3_5-1B-bf16.gguf",
        "q4": "OpenGVLab_InternVL3_5-1B-Q4_K_L.gguf",
        "q3": "OpenGVLab_InternVL3_5-1B-Q3_K_L.gguf",
    }

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.repo_id = config.get("repo_id", self.DEFAULT_REPO_ID)
        self.precision_mode = str(config.get("precision_mode", "q4")).lower()
        self.gguf_filename = config.get("gguf_filename") or self.PRECISION_TO_FILENAME.get(self.precision_mode)
        self.mmproj_filename = config.get("mmproj_filename", self.DEFAULT_MMPROJ_FILENAME)
        self.device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.require_differentiable = bool(config.get("require_differentiable", False))
        self.chat_handler_name = config.get("chat_handler", "Llava15ChatHandler")
        self.llama_cpp_config = dict(config.get("llama_cpp_config", {}))
        self.generation_defaults = dict(config.get("generation_defaults", {}))

        self.llm: Optional[Any] = None

        if self.require_differentiable:
            raise ValueError("InternVL35GGUFWrapper does not support require_differentiable=True")
        if not self.gguf_filename:
            raise ValueError("Missing gguf_filename for InternVL35GGUFWrapper")

    def load_model(self) -> None:
        """Load GGUF model via llama.cpp."""
        if self.llm is not None:
            return

        model_path = self._download_file(self.gguf_filename)
        mmproj_path = self._download_file(self.mmproj_filename) if self.mmproj_filename else None

        self.llm = self._build_llama(model_path=model_path, mmproj_path=mmproj_path)

    def preprocess_example(self, image: Any, prompt: str) -> Dict[str, Any]:
        """Return a lightweight payload used by generate_response."""
        return {"image": image, "prompt": prompt}

    def generate_response(
        self,
        image: Any,
        prompt: str,
        generation_config: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate response for one image + prompt pair."""
        self._ensure_loaded()

        assert self.llm is not None

        gen_cfg = {"max_new_tokens": 128, "temperature": 0.0}
        gen_cfg.update(self.generation_defaults)
        if generation_config:
            gen_cfg.update(generation_config)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        image_payload = self._ensure_pil_image(image)
        if image_payload is not None:
            messages[0]["content"].append({"type": "image", "image": image_payload})

        response = self._create_chat_completion(messages=messages, image=image_payload, gen_cfg=gen_cfg)
        return response.strip()

    def forward_for_loss(
        self,
        image_tensor: torch.Tensor,
        prompt: str,
        target_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """GGUF backend does not support differentiable forward."""
        raise RuntimeError("InternVL35GGUFWrapper does not support forward_for_loss")

    def get_device(self) -> torch.device:
        """Return configured device."""
        return self.device

    def _ensure_loaded(self) -> None:
        if self.llm is None:
            self.load_model()

    def _download_file(self, filename: str) -> str:
        return hf_hub_download(repo_id=self.repo_id, filename=filename)

    def _build_llama(self, model_path: str, mmproj_path: Optional[str]) -> Any:
        try:
            from llama_cpp import Llama
            from llama_cpp import llama_chat_format
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("llama-cpp-python is required for GGUF inference") from exc

        chat_handler = None
        if mmproj_path:
            handler_cls = getattr(llama_chat_format, self.chat_handler_name, None)
            if handler_cls is None:
                raise RuntimeError(f"chat_handler {self.chat_handler_name} not found in llama_cpp.llama_chat_format")
            chat_handler = handler_cls(clip_model_path=mmproj_path)

        cfg = {
            "n_gpu_layers": self.llama_cpp_config.get("n_gpu_layers", -1),
            "n_ctx": self.llama_cpp_config.get("n_ctx", 4096),
            "n_batch": self.llama_cpp_config.get("n_batch", 512),
            "n_ubatch": self.llama_cpp_config.get("n_ubatch", 64),
            "offload_kqv": self.llama_cpp_config.get("offload_kqv", True),
            "verbose": self.llama_cpp_config.get("verbose", False),
        }

        for key in ("tensor_split", "rope_freq_base", "rope_freq_scale", "cache_type"):
            if key in self.llama_cpp_config:
                cfg[key] = self.llama_cpp_config[key]

        return Llama(model_path=model_path, chat_handler=chat_handler, **cfg)

    def _create_chat_completion(self, messages: Any, image: Any, gen_cfg: Dict[str, Any]) -> str:
        assert self.llm is not None

        params = {
            "messages": messages,
            "max_tokens": gen_cfg.get("max_new_tokens") or gen_cfg.get("max_tokens"),
            "temperature": gen_cfg.get("temperature", 0.0),
            "top_p": gen_cfg.get("top_p"),
            "top_k": gen_cfg.get("top_k"),
            "repeat_penalty": gen_cfg.get("repetition_penalty"),
            "stop": gen_cfg.get("stop"),
        }

        # Filter None values to avoid llama.cpp validation issues.
        params = {k: v for k, v in params.items() if v is not None}

        # Try image-aware call first, fall back to text-only if not supported.
        try:
            result = self.llm.create_chat_completion(**params, images=[image] if image is not None else None)
        except TypeError:
            result = self.llm.create_chat_completion(**params)

        choices = result.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        return str(message.get("content", ""))

    @staticmethod
    def _ensure_pil_image(image: Any) -> Any:
        if image is None:
            return None
        if isinstance(image, torch.Tensor):
            from PIL import Image
            import numpy as np

            if image.dim() == 4 and image.shape[0] == 1:
                image = image[0]
            image = torch.clamp(image.detach().cpu(), 0.0, 1.0)
            array = (image.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
            return Image.fromarray(array, mode="RGB")
        return image
