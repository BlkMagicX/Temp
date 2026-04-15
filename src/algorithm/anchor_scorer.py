"""Teacher-forced anchor sequence scoring utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


@dataclass
class AnchorScoringConfig:
    """Configuration for refusal/compliance anchor scoring."""

    refusal_anchors: List[str] = field(default_factory=list)
    compliance_anchors: List[str] = field(default_factory=list)
    aggregation: str = "logsumexp"


class AnchorScorer:
    """Compute anchor-sequence scores with teacher forcing.

    For one anchor sequence a=(a_1,...,a_K), this module estimates
    s_q(z; a) = sum_t log P_q(a_t | z, a_<t)
    through the model's differentiable teacher-forced forward path.
    """

    def __init__(self, config: AnchorScoringConfig) -> None:
        self.config = config
        self._cache: Dict[Tuple[str, str, str], float] = {}

    def score_anchor(
        self,
        model: Any,
        image_tensor: torch.Tensor,
        prompt: str,
        anchor: str,
        cache_prefix: Optional[str] = None,
        model_key: str = "unknown",
    ) -> float:
        """Score one anchor text under teacher forcing.

        Args:
            model: VLM wrapper exposing forward_for_loss.
            image_tensor: BCHW image tensor in [0, 1].
            prompt: User prompt string.
            anchor: Anchor response text to score.
            cache_prefix: Optional sample-level cache prefix.
            model_key: Human-readable model/precision key.
        """
        cache_key = None
        if cache_prefix is not None:
            cache_key = (cache_prefix, model_key, anchor)
            if cache_key in self._cache:
                return self._cache[cache_key]

        loglik = self._teacher_forced_log_likelihood(
            model=model,
            image_tensor=image_tensor,
            prompt=prompt,
            target_text=anchor,
        )

        if cache_key is not None:
            self._cache[cache_key] = loglik
        return loglik

    def score_anchor_set(
        self,
        model: Any,
        image_tensor: torch.Tensor,
        prompt: str,
        anchors: Sequence[str],
        cache_prefix: Optional[str] = None,
        model_key: str = "unknown",
    ) -> List[float]:
        """Score a list of anchors and return per-anchor scores."""
        scores: List[float] = []
        for anchor in anchors:
            scores.append(
                self.score_anchor(
                    model=model,
                    image_tensor=image_tensor,
                    prompt=prompt,
                    anchor=anchor,
                    cache_prefix=cache_prefix,
                    model_key=model_key,
                )
            )
        return scores

    def score_refusal(
        self,
        model: Any,
        image_tensor: torch.Tensor,
        prompt: str,
        cache_prefix: Optional[str] = None,
        model_key: str = "unknown",
    ) -> List[float]:
        """Score refusal anchors."""
        return self.score_anchor_set(
            model=model,
            image_tensor=image_tensor,
            prompt=prompt,
            anchors=self.config.refusal_anchors,
            cache_prefix=cache_prefix,
            model_key=model_key,
        )

    def score_compliance(
        self,
        model: Any,
        image_tensor: torch.Tensor,
        prompt: str,
        cache_prefix: Optional[str] = None,
        model_key: str = "unknown",
    ) -> List[float]:
        """Score compliance anchors."""
        return self.score_anchor_set(
            model=model,
            image_tensor=image_tensor,
            prompt=prompt,
            anchors=self.config.compliance_anchors,
            cache_prefix=cache_prefix,
            model_key=model_key,
        )

    @staticmethod
    def _teacher_forced_log_likelihood(
        model: Any,
        image_tensor: torch.Tensor,
        prompt: str,
        target_text: str,
    ) -> float:
        """Estimate sequence log-likelihood from teacher-forced forward outputs."""
        if hasattr(model, "score_sequence_loglikelihood"):
            return float(model.score_sequence_loglikelihood(image=image_tensor, prompt=prompt, target_text=target_text))

        # Scoring margins does not require gradient graphs; inference_mode
        # significantly reduces activation memory for large VLMs.
        with torch.inference_mode():
            outputs = model.forward_for_loss(
                image_tensor=image_tensor,
                prompt=prompt,
                target_text=target_text,
            )

        labels = outputs.get("labels")
        logits = outputs.get("logits")
        loss = outputs.get("loss")

        if labels is None:
            raise RuntimeError("forward_for_loss must return labels for teacher-forced scoring")

        valid_tokens = int((labels != -100).sum().detach().item())
        if valid_tokens <= 0:
            return 0.0

        if loss is not None:
            return float((-loss.detach() * valid_tokens).item())

        if logits is None:
            raise RuntimeError("forward_for_loss must return loss or logits for teacher-forced scoring")

        vocab_size = logits.shape[-1]
        ce = F.cross_entropy(
            logits.view(-1, vocab_size),
            labels.view(-1),
            ignore_index=-100,
            reduction="sum",
        )
        return float((-ce.detach()).item())
