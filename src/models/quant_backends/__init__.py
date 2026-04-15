"""Quant backend adapters for quantized VLM loading."""

from .base_quant_backend import BaseQuantBackend
from .gptq_backend import GPTQBackend

__all__ = ["BaseQuantBackend", "GPTQBackend"]
