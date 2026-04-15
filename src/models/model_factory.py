"""Factory for constructing VLM model wrappers from config."""

from __future__ import annotations

import importlib
from typing import Any, Dict

from .base_vlm import BaseVLM


def create_vlm(model_config: Dict[str, Any]) -> BaseVLM:
    """Create and return a VLM instance based on config.

    Supported examples:
        {"name": "qwen2vl", "precision_mode": "bf16", ...}
        {"model_type": "qwen2-vl", "precision_mode": "w8a8", ...}

    Args:
        model_config: Model configuration dictionary.

    Returns:
        A `BaseVLM` implementation.

    Raises:
        ValueError: If model type is missing or unsupported.
    """
    model_type = model_config.get("name") or model_config.get("model_type") or model_config.get("type")

    if not model_type:
        raise ValueError("Model config must include one of: name/model_type/type")

    norm_type = str(model_type).lower().replace("_", "-")

    # Compatibility guard: if config name says qwen2vl but model path points to
    # Qwen2.5-VL, route to Qwen2.5 wrapper to avoid shape mismatch at load time.
    model_path = str(model_config.get("model_path") or model_config.get("model_name_or_path") or "").lower()
    if norm_type in {"qwen2vl", "qwen2-vl", "qwen2-vl-7b-instruct"} and "qwen2.5-vl" in model_path:
        norm_type = "qwen25vl"

    if norm_type in {"qwen2vl", "qwen2-vl", "qwen2-vl-7b-instruct"}:
        module = importlib.import_module("src.models.qwen2vl_wrapper")
        wrapper_cls = getattr(module, "Qwen2VLWrapper")
        return wrapper_cls(model_config)

    if norm_type in {
        "qwen25vl",
        "qwen2.5-vl",
        "qwen2-5-vl",
        "qwen2.5-vl-7b-instruct",
        "qwen2.5-vl-3b-instruct",
    }:
        module = importlib.import_module("src.models.qwen25vl_wrapper")
        wrapper_cls = getattr(module, "Qwen25VLWrapper")
        return wrapper_cls(model_config)

    raise ValueError(f"Unsupported model type: {model_type}")


def create_vlm_from_root_config(config: Dict[str, Any]) -> BaseVLM:
    """Create VLM instance from root experiment config.

    This keeps model-creation logic out of `main` and pipeline scripts.

    Args:
        config: Root config dictionary that may contain `model` section.

    Returns:
        A `BaseVLM` implementation.
    """
    model_cfg = config.get("model", config)
    return create_vlm(model_cfg)
