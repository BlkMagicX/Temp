"""Gradient-based local boundary sensitivity metrics."""

from __future__ import annotations

from typing import Any, Sequence

import torch


def _aggregate_tensor(scores: Sequence[torch.Tensor], method: str) -> torch.Tensor:
    if not scores:
        raise ValueError("scores must not be empty")

    stack = torch.stack(list(scores), dim=0)
    method_norm = method.lower().strip()
    if method_norm == "logsumexp":
        return torch.logsumexp(stack, dim=0)
    if method_norm == "mean":
        return stack.mean(dim=0)
    if method_norm == "max":
        return stack.max(dim=0).values
    raise ValueError(f"Unsupported aggregation method: {method}")


def _teacher_forced_loglik_tensor(
    model: Any,
    image_tensor: torch.Tensor,
    prompt: str,
    anchor: str,
) -> torch.Tensor:
    outputs = model.forward_for_loss(
        image_tensor=image_tensor,
        prompt=prompt,
        target_text=anchor,
    )
    loss = outputs.get("loss")
    labels = outputs.get("labels")
    if loss is None or labels is None:
        raise RuntimeError("Gradient metrics require forward_for_loss to return loss and labels")

    token_count = (labels != -100).sum().clamp(min=1).to(dtype=loss.dtype)
    return -loss * token_count


def compute_margin_gradient(
    model: Any,
    image_tensor: torch.Tensor,
    prompt: str,
    refusal_anchors: Sequence[str],
    compliance_anchors: Sequence[str],
    aggregation: str = "logsumexp",
) -> torch.Tensor:
    """Compute gradient of margin wrt image tensor as a practical proxy."""
    if not refusal_anchors or not compliance_anchors:
        raise ValueError("Both refusal_anchors and compliance_anchors must be non-empty")

    x = image_tensor.detach().clone().requires_grad_(True)

    ref_terms = [_teacher_forced_loglik_tensor(model, x, prompt, a) for a in refusal_anchors]
    comp_terms = [_teacher_forced_loglik_tensor(model, x, prompt, a) for a in compliance_anchors]
    margin = _aggregate_tensor(ref_terms, aggregation) - _aggregate_tensor(comp_terms, aggregation)

    grad = torch.autograd.grad(margin, x, retain_graph=False, create_graph=False)[0]
    return grad.detach()


def compute_kappa(grad: torch.Tensor) -> float:
    """kappa_q(z)=||grad m_q(z)||_2."""
    return float(grad.view(-1).norm(p=2).item())


def compute_eta(grad_q: torch.Tensor, grad_fp: torch.Tensor) -> float:
    """eta_q(z)=||grad m_q(z)-grad m_fp(z)||_2."""
    return float((grad_q - grad_fp).view(-1).norm(p=2).item())


def compute_rho(grad_q: torch.Tensor, grad_fp: torch.Tensor, eps: float = 1e-12) -> float:
    """rho_q(z)=cos(grad m_q(z), grad m_fp(z))."""
    gq = grad_q.view(-1)
    gf = grad_fp.view(-1)
    denom = gq.norm(p=2) * gf.norm(p=2) + eps
    if float(denom.item()) <= 0.0:
        return 0.0
    return float(torch.dot(gq, gf).div(denom).item())
