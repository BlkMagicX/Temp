"""Qwen2-VL wrapper implementing the unified VLM interface."""

from __future__ import annotations

import importlib
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from .base_vlm import BaseVLM


class Qwen2VLWrapper(BaseVLM):
    """Unified wrapper for `Qwen/Qwen2-VL-7B-Instruct`.

    Notes:
                - `bf16` path is implemented and runnable.
                - quantized paths are routed through backend adapters (`mbq` / `awq`).
                - MBQ/AWQ backend internals are currently explicit placeholders unless
                    real runtime loaders are integrated.
    """

    SUPPORTED_PRECISION = {"bf16", "w3a16", "w8a8", "w4a4", "w4a16", "w8a16"}
    SUPPORTED_BACKEND_TYPES = {"bf16", "mbq", "awq", "gptq", "vllm"}

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize wrapper from config.

        Expected config fields (flat or nested by caller):
            - model_path / model_name_or_path: HF model id/path
            - precision_mode: one of {bf16, , w4a16}
            - backend_type: one of {bf16, mbq, awq}
            - quant_model_path: optional quantized checkpoint path
            - device / device_map: optional runtime placement
            - torch_dtype: optional, e.g. "bfloat16", "float16"
            - require_differentiable: whether white-box gradient path is required
        """
        self.config = config
        self.model_path = config.get("model_path") or config.get("model_name_or_path", "Qwen/Qwen2-VL-7B-Instruct")
        self.quant_model_path = config.get("quant_model_path")
        self.processor_path = config.get("processor_path", self.model_path)
        self.precision_mode = str(config.get("precision_mode", "bf16")).lower()
        self.backend_type = str(config.get("backend_type", "bf16")).lower()
        self.device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.device_map = config.get("device_map", None)
        self.torch_dtype = self._parse_torch_dtype(config.get("torch_dtype", "bfloat16"))
        self.require_differentiable = bool(config.get("require_differentiable", False))
        self.quant_backend_config = dict(config.get("quant_backend_config", {}))
        self.vllm_config = dict(config.get("vllm_config", {}))

        if self.precision_mode not in self.SUPPORTED_PRECISION:
            raise ValueError(f"Unsupported precision_mode={self.precision_mode}. " f"Expected one of {sorted(self.SUPPORTED_PRECISION)}")
        if self.backend_type not in self.SUPPORTED_BACKEND_TYPES:
            raise ValueError(f"Unsupported backend_type={self.backend_type}. " f"Expected one of {sorted(self.SUPPORTED_BACKEND_TYPES)}")
        if self.precision_mode == "bf16" and self.backend_type != "bf16":
            raise ValueError("precision_mode=bf16 requires backend_type=bf16")
        if self.precision_mode in {"", "w3a16", "w4a16"} and self.backend_type == "bf16":
            raise ValueError(f"precision_mode={self.precision_mode} requires quant backend_type (mbq/awq/gptq), not bf16")

        self.model: Optional[Qwen2VLForConditionalGeneration] = None
        self.processor: Optional[AutoProcessor] = None
        self.quant_backend: Optional[Any] = None
        self.vllm_llm: Optional[Any] = None

    def load_model(self) -> None:
        """Load processor and model according to `precision_mode`."""
        if self.backend_type == "vllm":
            if self.vllm_llm is not None:
                return
            self._load_vllm_model()
            return

        if self.model is not None and self.processor is not None:
            return

        self.processor = AutoProcessor.from_pretrained(self.processor_path)

        if self.backend_type == "bf16" or self.precision_mode == "bf16":
            load_kwargs: Dict[str, Any] = {}
            if self.torch_dtype is not None:
                load_kwargs["torch_dtype"] = self.torch_dtype
            if self.device_map is not None:
                load_kwargs["device_map"] = self.device_map

            self.model = Qwen2VLForConditionalGeneration.from_pretrained(self.model_path, **load_kwargs)
            if self.device_map is None:
                self.model.to(self.device)
            self.model.eval()
            return

        self.model = self._load_quantized_model()
        if self.device_map is None:
            self.model.to(self.device)
            if self.torch_dtype is not None:
                try:
                    self.model.to(dtype=self.torch_dtype)
                except Exception as exc:  # noqa: BLE001
                    raise RuntimeError(
                        "Failed to cast quantized model to requested torch_dtype. "
                        "Try setting device_map=None and a compatible torch_dtype (e.g. float16)."
                    ) from exc
        self.model.eval()

    def preprocess_example(self, image: Any, prompt: str) -> Dict[str, Any]:
        """Build model input tensors for one image + prompt example.

        This path is intended for generation/evaluation and is decoupled from
        attack-specific differentiable preprocessing.
        """
        self._ensure_loaded()
        assert self.processor is not None

        chat_text = self._build_user_chat_text(prompt, add_generation_prompt=True)
        inputs = self.processor(
            text=[chat_text],
            images=[image],
            return_tensors="pt",
        )
        return self._align_batch_for_model(inputs)

    @torch.inference_mode()
    def generate_response(
        self,
        image: Any,
        prompt: str,
        generation_config: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate response text for one image + prompt pair."""
        if self.backend_type == "vllm":
            return self._generate_response_vllm(image=image, prompt=prompt, generation_config=generation_config)

        self._ensure_loaded()
        assert self.model is not None and self.processor is not None

        model_inputs = self.preprocess_example(image=image, prompt=prompt)
        gen_cfg = {"max_new_tokens": 128, "do_sample": False}
        if generation_config:
            gen_cfg.update(generation_config)

        output_ids = self.model.generate(**model_inputs, **gen_cfg)
        input_length = model_inputs["input_ids"].shape[-1]
        generated_ids = output_ids[:, input_length:]
        text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return text.strip()

    @torch.inference_mode()
    def score_sequence_loglikelihood(
        self,
        image: Any,
        prompt: str,
        target_text: str,
    ) -> float:
        """Teacher-forced sequence log-likelihood for inference-only scoring.

        This path is intentionally non-differentiable and can be used on
        non-differentiable quantized backends (for example GPTQ).
        """
        self._ensure_loaded()
        assert self.model is not None and self.processor is not None

        if self.backend_type == "vllm":
            raise RuntimeError("score_sequence_loglikelihood is not supported for backend_type=vllm")

        if isinstance(image, torch.Tensor):
            image = self._tensor_to_pil(image)
        if hasattr(image, "convert"):
            image = image.convert("RGB")

        prefix_text = self._build_user_chat_text(prompt, add_generation_prompt=True)
        prefix_inputs = self.processor(
            text=[prefix_text],
            images=[image],
            return_tensors="pt",
        )

        full_text = prefix_text + target_text
        model_inputs = self.processor(
            text=[full_text],
            images=[image],
            return_tensors="pt",
        )

        prefix_len = int(prefix_inputs["input_ids"].shape[-1])
        labels = model_inputs["input_ids"].clone()
        labels[:, :prefix_len] = -100
        valid_tokens = int((labels != -100).sum().item())

        model_inputs["labels"] = labels
        model_inputs["return_dict"] = True
        model_inputs = self._align_batch_for_model(model_inputs)

        outputs = self.model(**model_inputs)
        loss = outputs.loss
        if loss is None or valid_tokens <= 0:
            return 0.0
        return float((-loss.detach() * valid_tokens).item())

    def forward_for_loss(
        self,
        image_tensor: torch.Tensor,
        prompt: str,
        target_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run differentiable forward pass for attack optimization.

        Args:
            image_tensor: Tensor used as visual input in gradient-based attacks.
                Recommended shape is `(B, C, H, W)` and `requires_grad=True`.
            prompt: User-side prompt.
            target_text: Optional target string for supervised attack objective.
                If None, `labels` is set to None and model loss may be None.

        Returns:
            Dict containing at least `loss`, `logits`, `input_ids`, `labels`.

        Important:
            This method is separate from generation and is designed to preserve
            gradient flow from loss to `image_tensor`.
        """
        self._ensure_loaded()
        assert self.model is not None and self.processor is not None

        if self.backend_type == "vllm":
            raise RuntimeError("forward_for_loss is not supported for backend_type=vllm")

        if self.quant_backend is not None and not self.quant_backend.is_differentiable():
            raise RuntimeError(
                "forward_for_loss requires differentiable backend, but current quant backend "
                f"`{self.backend_type}` is marked non-differentiable. "
                "Use bf16 attack model or implement differentiable quant backend."
            )
        if self.require_differentiable and self.quant_backend is not None and not self.quant_backend.is_differentiable():
            raise RuntimeError("This model is configured with require_differentiable=True but backend is non-differentiable.")

        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        if image_tensor.dim() != 4:
            raise ValueError("image_tensor must be CHW or BCHW tensor")
        if image_tensor.shape[0] != 1:
            raise ValueError("forward_for_loss currently supports batch size 1")

        image_tensor = image_tensor.to(self.device)

        prefix_text = self._build_user_chat_text(prompt, add_generation_prompt=True)
        image_pil = self._tensor_to_pil(image_tensor.detach())

        prefix_inputs = self.processor(
            text=[prefix_text],
            images=[image_pil],
            return_tensors="pt",
        )

        labels = None
        if target_text is not None:
            full_text = prefix_text + target_text
            model_inputs = self.processor(
                text=[full_text],
                images=[image_pil],
                return_tensors="pt",
            )
            prefix_len = int(prefix_inputs["input_ids"].shape[-1])
            labels = model_inputs["input_ids"].clone()
            labels[:, :prefix_len] = -100
        else:
            model_inputs = prefix_inputs

        input_ids = model_inputs["input_ids"].to(self.device)
        attention_mask = model_inputs["attention_mask"].to(self.device)
        mm_token_type_ids = model_inputs.get("mm_token_type_ids")
        if mm_token_type_ids is not None:
            mm_token_type_ids = mm_token_type_ids.to(self.device)

        image_grid_thw = model_inputs.get("image_grid_thw")
        if image_grid_thw is None:
            raise RuntimeError("Qwen2-VL forward requires `image_grid_thw`, but processor did not return it.")
        image_grid_thw = image_grid_thw.to(self.device)

        pixel_values = self._build_differentiable_pixel_values(
            image_tensor=image_tensor,
            image_grid_thw=image_grid_thw,
        )

        if labels is not None:
            labels = labels.to(self.device)

        forward_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "labels": labels,
            "return_dict": True,
        }
        if mm_token_type_ids is not None:
            forward_inputs["mm_token_type_ids"] = mm_token_type_ids

        forward_inputs = self._align_batch_for_model(forward_inputs)

        outputs = self.model(**forward_inputs)

        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

    def _get_input_device(self) -> torch.device:
        """Best-effort device for input tensors under device_map setups."""
        try:
            emb = self.model.get_input_embeddings()
            return emb.weight.device
        except Exception:
            return next(self.model.parameters()).device

    def _get_vision_dtype(self) -> torch.dtype:
        """Best-effort dtype for vision branch tensors."""
        try:
            return self.model.visual.patch_embed.proj.weight.dtype
        except Exception:
            # print(next(self.model.parameters()).dtype)
            return next(self.model.parameters()).dtype

    def _align_batch_for_model(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Align tensor device/dtype to match model requirements.

        - `pixel_values` cast to vision dtype.
        - Other tensors moved to input device without dtype change.
        """
        if self.model is None:
            return batch

        input_device = self._get_input_device()
        vision_dtype = self._get_vision_dtype()
        aligned: Dict[str, Any] = {}
        for k, v in batch.items():
            if not isinstance(v, torch.Tensor):
                aligned[k] = v
                continue
            if k == "pixel_values":
                aligned[k] = v.to(device=input_device, dtype=vision_dtype, non_blocking=True)
            else:
                aligned[k] = v.to(device=input_device, non_blocking=True)
        return aligned

    def _build_differentiable_pixel_values(
        self,
        image_tensor: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        """Convert BCHW tensor to Qwen2-VL visual patch tokens differentiably.

        Qwen2-VL expects `pixel_values` shaped as `(num_patches, 3*2*14*14)`.
        This function keeps gradient path from model loss to `image_tensor`.
        """
        if image_tensor.shape[0] != 1:
            raise ValueError("Only batch size 1 is supported in current implementation")

        patch_size = 14
        temporal_patch = 2

        t_grid = int(image_grid_thw[0, 0].item())
        h_grid = int(image_grid_thw[0, 1].item())
        w_grid = int(image_grid_thw[0, 2].item())

        target_h = h_grid * patch_size
        target_w = w_grid * patch_size

        x = F.interpolate(image_tensor, size=(target_h, target_w), mode="bilinear", align_corners=False)

        image_processor = getattr(self.processor, "image_processor", None)
        if image_processor is not None:
            mean = torch.tensor(getattr(image_processor, "image_mean", [0.5, 0.5, 0.5]), device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
            std = torch.tensor(getattr(image_processor, "image_std", [0.5, 0.5, 0.5]), device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
            x = (x - mean) / std

        num_frames = max(1, t_grid) * temporal_patch
        x = x.unsqueeze(2).repeat(1, 1, num_frames, 1, 1)
        x = x.view(1, 3, t_grid, temporal_patch, target_h, target_w)
        x = x.view(1, 3, t_grid, temporal_patch, h_grid, patch_size, w_grid, patch_size)
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
        pixel_values = x.view(-1, 3 * temporal_patch * patch_size * patch_size)
        return pixel_values

    @staticmethod
    def _tensor_to_pil(image_tensor: torch.Tensor) -> Any:
        """Convert BCHW/BCHW(1) tensor in [0, 1] to RGB PIL image."""
        from PIL import Image
        import numpy as np

        if image_tensor.dim() == 4:
            if image_tensor.shape[0] != 1:
                raise ValueError("Expected batch size 1")
            chw = image_tensor[0]
        else:
            chw = image_tensor

        chw = torch.clamp(chw.detach().cpu(), 0.0, 1.0)
        array = (chw.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        return Image.fromarray(array, mode="RGB")

    def get_device(self) -> torch.device:
        """Return model device."""
        return self.device

    def _load_vllm_model(self) -> None:
        """Load vLLM model for multimodal generation-only inference."""
        try:
            from vllm import LLM
        except ImportError as exc:  # pragma: no cover
            raise ImportError("backend_type=vllm requires installing vllm") from exc

        model_id = self.quant_model_path or self.model_path
        max_model_len = self.vllm_config.get("max_model_len", self.config.get("max_model_len", 4096))
        max_num_seqs = self.vllm_config.get("max_num_seqs", self.config.get("max_num_seqs", 2))

        if self.processor is None:
            self.processor = AutoProcessor.from_pretrained(self.processor_path)

        self.vllm_llm = LLM(
            model=model_id,
            max_model_len=int(max_model_len),
            max_num_seqs=int(max_num_seqs),
        )

    def _build_vllm_prompt(self, prompt: str) -> str:
        """Build default chat prompt for vLLM multimodal call."""
        if self.processor is not None and hasattr(self.processor, "apply_chat_template"):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            return self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        return f"<|user|>\n<|image_1|>\n{prompt}<|end|>\n<|assistant|>\n"

    def _generate_response_vllm(
        self,
        image: Any,
        prompt: str,
        generation_config: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate response with vLLM backend."""
        self._ensure_loaded()
        assert self.vllm_llm is not None

        try:
            from vllm import SamplingParams
        except ImportError as exc:  # pragma: no cover
            raise ImportError("backend_type=vllm requires installing vllm") from exc

        gen_cfg = {"max_new_tokens": 128, "do_sample": False, "temperature": 0.0}
        if generation_config:
            gen_cfg.update(generation_config)

        max_tokens = int(gen_cfg.get("max_tokens", gen_cfg.get("max_new_tokens", 128)))
        temperature = float(gen_cfg.get("temperature", 0.0))
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
        )

        prompt_text = self._build_vllm_prompt(prompt)
        if isinstance(image, torch.Tensor):
            image = self._tensor_to_pil(image)
        if hasattr(image, "convert"):
            image = image.convert("RGB")

        inputs = {
            "prompt": prompt_text,
            "multi_modal_data": {"image": [image]},
        }

        outputs = self.vllm_llm.generate(inputs, sampling_params)
        if not outputs:
            return ""
        out0 = outputs[0]
        if not getattr(out0, "outputs", None):
            return ""
        return str(out0.outputs[0].text).strip()

    def _load_quantized_model(self) -> Qwen2VLForConditionalGeneration:
        """Load quantized model via selected backend adapter."""
        self.quant_backend = self._build_quant_backend()
        return self.quant_backend.load_quantized_model(
            model_cls=Qwen2VLForConditionalGeneration,
            model_path=self.model_path,
            quant_model_path=self.quant_model_path,
            device_map=self.device_map,
            torch_dtype=self.torch_dtype,
            extra_config=self.quant_backend_config,
        )

    def _build_quant_backend(self) -> Any:
        """Construct quant backend from configuration.

        Mapping:
            - backend_type=mbq -> `MBQBackend`
            - backend_type=awq -> `AWQBackend`
            - backend_type=gptq -> `GPTQBackend`
        """
        if self.backend_type == "mbq":
            mbq_module = importlib.import_module("src.models.quant_backends.mbq_backend")
            mbq_cls = getattr(mbq_module, "MBQBackend")
            return mbq_cls(precision_name=self.precision_mode)
        if self.backend_type == "awq":
            awq_module = importlib.import_module("src.models.quant_backends.awq_backend")
            awq_cls = getattr(awq_module, "AWQBackend")
            return awq_cls(precision_name=self.precision_mode)
        if self.backend_type == "gptq":
            gptq_module = importlib.import_module("src.models.quant_backends.gptq_backend")
            gptq_cls = getattr(gptq_module, "GPTQBackend")
            return gptq_cls(precision_name=self.precision_mode)

        raise ValueError(f"Quantized precision requested with unsupported backend_type={self.backend_type}. " "Expected 'mbq', 'awq', or 'gptq'.")

    def _ensure_loaded(self) -> None:
        """Lazy-load model and processor if needed."""
        if self.backend_type == "vllm":
            if self.vllm_llm is None:
                self.load_model()
            return

        if self.model is None or self.processor is None:
            self.load_model()

    @staticmethod
    def _parse_torch_dtype(torch_dtype_cfg: Any) -> Optional[torch.dtype]:
        """Parse torch dtype config into `torch.dtype`.

        Supported strings: `bfloat16`, `float16`, `float32`.
        """
        if torch_dtype_cfg is None:
            return None
        if isinstance(torch_dtype_cfg, str) and torch_dtype_cfg.lower() in {"auto", "none"}:
            return None
        if isinstance(torch_dtype_cfg, torch.dtype):
            return torch_dtype_cfg

        value = str(torch_dtype_cfg).lower()
        mapping = {
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        if value not in mapping:
            raise ValueError(f"Unsupported torch_dtype config: {torch_dtype_cfg}")
        return mapping[value]

    def _build_user_chat_text(self, prompt: str, add_generation_prompt: bool) -> str:
        """Build chat-formatted text string with one image slot and user prompt."""
        assert self.processor is not None

        if hasattr(self.processor, "apply_chat_template"):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            return self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )

        return prompt
