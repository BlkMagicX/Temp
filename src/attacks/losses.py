"""Loss functions for visual jailbreak attacks in VLM settings."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F


def targeted_token_level_loss(model_outputs: Dict[str, Any]) -> torch.Tensor:
    """Compute targeted token-level objective.

    Priority:
      1) Use model-provided `loss` directly when available.
      2) Otherwise compute token-level CE from `logits` and `labels`.

    Returns:
        A scalar tensor to maximize target tendency (minimization objective).
    """
    if model_outputs.get("loss") is not None:
        return model_outputs["loss"]

    logits = model_outputs.get("logits")
    labels = model_outputs.get("labels")
    if logits is None or labels is None:
        raise ValueError("targeted_token_level_loss requires either `loss` or (`logits`, `labels`).")

    vocab_size = logits.shape[-1]
    return F.cross_entropy(
        logits.view(-1, vocab_size),
        labels.view(-1),
        ignore_index=-100,
    )


def negative_refusal_loss(
    model_outputs: Dict[str, Any],
    refusal_target_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Placeholder for differentiable anti-refusal objective.

    This term is intentionally left as an extension hook. A robust
    implementation should estimate/refine refusal-token probabilities and push
    them down.

    Raises:
        NotImplementedError: Explicitly marks pending research implementation.
    """
    raise NotImplementedError("negative_refusal_loss is a placeholder. " "Please implement refusal-token probability suppression logic.")


def compose_attack_loss(
    model_outputs: Dict[str, Any],
    use_targeted_loss: bool = True,
    use_negative_refusal: bool = False,
    negative_refusal_weight: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """Compose attack loss from configurable terms.

    Returns:
        Dictionary containing `total_loss` and each component term.
    """
    if not use_targeted_loss:
        raise ValueError("At least targeted loss should be enabled for current setup.")

    targeted = targeted_token_level_loss(model_outputs)
    total = targeted

    terms: Dict[str, torch.Tensor] = {
        "targeted_loss": targeted,
    }

    if use_negative_refusal:
        refusal_term = negative_refusal_loss(model_outputs)
        total = total + negative_refusal_weight * refusal_term
        terms["negative_refusal_loss"] = refusal_term

    terms["total_loss"] = total
    return terms
