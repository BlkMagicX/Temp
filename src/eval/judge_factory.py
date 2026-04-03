"""Factory for constructing pluggable judges from config."""

from __future__ import annotations

from typing import Any, Dict

from .base_judge import BaseJudge
from .harmbench_judge import HarmBenchJudge
from .qwen_api_caption_judge import QwenApiCaptionJudge
from .gpt_api_caption_judge import GPT4oApiCaptionJudge
from .rule_based_judge import RuleBasedJudge


def create_judge(judge_config: Dict[str, Any]) -> BaseJudge:
    """Create judge instance from config.

    Supported values:
        - rule_based
        - harmbench
        - qwen_caption
        - qwen_api_caption
    """
    judge_type = judge_config.get("type") or judge_config.get("name") or "rule_based"
    norm_type = str(judge_type).lower().replace("_", "-")

    if norm_type in {"rule-based", "rulebased", "rule"}:
        return RuleBasedJudge(
            refusal_keywords=judge_config.get("refusal_keywords"),
            caption_overlap_threshold=float(judge_config.get("caption_overlap_threshold", 0.3)),
        )

    if norm_type in {"harmbench", "harmbench-classifier"}:
        return HarmBenchJudge(
            classifier_path=judge_config.get("classifier_path"),
            device=judge_config.get("device", "cuda"),
            lazy_load=bool(judge_config.get("lazy_load", True)),
        )

    if norm_type in {"qwen-api-caption", "qwen_api_caption"}:
        return QwenApiCaptionJudge(
            model_name=judge_config.get("model_name", "qwen-plus"),
            api_key=judge_config.get("api_key", "sk-49bc01e65b8840c89d4bd190cb7f9c3e"),
            base_url=judge_config.get("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            attack_mode=judge_config.get("attack_mode", "target"),
            caption_success_threshold=float(judge_config.get("caption_success_threshold", 0.5)),
            refusal_keywords=judge_config.get("refusal_keywords"),
            max_tokens=int(judge_config.get("max_tokens", 32)),
            temperature=float(judge_config.get("temperature", 0.0)),
            timeout=float(judge_config.get("timeout", 60.0)),
        )

    if norm_type in {"gpt-api-caption", "gpt_api_caption"}:
        return GPT4oApiCaptionJudge(
            model_name=judge_config.get("model_name", "gpt-4o"),
            api_key=judge_config.get("api_key", ""),
            base_url=judge_config.get("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            caption_success_threshold=float(judge_config.get("caption_success_threshold", 0.5)),
            refusal_keywords=judge_config.get("refusal_keywords"),
            max_tokens=int(judge_config.get("max_tokens", 32)),
            temperature=float(judge_config.get("temperature", 0.0)),
            timeout=float(judge_config.get("timeout", 60.0)),
        )

    raise ValueError(f"Unsupported judge type: {judge_type}")
