"""Gradient-based local boundary sensitivity metrics."""

from __future__ import annotations

import math
from typing import Any, Sequence

import torch


def _aggregate_tensor(scores: Sequence[torch.Tensor], method: str) -> torch.Tensor:
    if not scores:
        raise ValueError("scores must not be empty")

    stack = torch.stack(list(scores), dim=0)
    method_norm = method.lower().strip()
    if method_norm == "logsumexp":
        return torch.logsumexp(stack, dim=0) - math.log(float(stack.shape[0]))
    if method_norm == "mean":
        return stack.mean(dim=0)
    if method_norm == "max":
        return stack.max(dim=0).values
    raise ValueError(f"Unsupported aggregation method: {method}")


def _aggregate_grads(
    values: Sequence[float],
    grads: Sequence[torch.Tensor],
    method: str,
    tau: float = 1.0,
) -> torch.Tensor:
    if not values or not grads:
        raise ValueError("values and grads must not be empty")
    if len(values) != len(grads):
        raise ValueError("values and grads must have the same length")

    method_norm = method.lower().strip()
    if method_norm == "mean":
        out = grads[0].clone()
        for g in grads[1:]:
            out = out + g
        return out / float(len(grads))

    if method_norm == "max":
        val_tensor = torch.tensor(values, dtype=torch.float32)
        idx = int(torch.argmax(val_tensor).item())
        return grads[idx]

    if method_norm == "logsumexp":
        val_tensor = torch.tensor(values, device=grads[0].device, dtype=grads[0].dtype)
        if float(tau) != 1.0:
            val_tensor = val_tensor / float(tau)
        weights = torch.softmax(val_tensor, dim=0)
        out = torch.zeros_like(grads[0])
        for i, g in enumerate(grads):
            out = out + weights[i] * g
        return out

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
    if int(token_count.item()) <= 0:
        return loss.detach().new_zeros(())
    # Length-normalized log-likelihood per token.
    return -loss


def compute_margin_gradient(
    model: Any,
    image_tensor: torch.Tensor,
    prompt: str,
    refusal_anchors: Sequence[str],
    compliance_anchors: Sequence[str],
    aggregation: str = "logsumexp",
    aggregation_tau: float = 1.0,
) -> torch.Tensor:
    """Compute gradient of margin wrt image tensor as a practical proxy.

    Memory note:
    - This implementation computes per-anchor gradients sequentially, then
      aggregates them, instead of building all anchor graphs at once.
    - Peak VRAM is therefore much lower for multi-anchor settings.
    """
    if not refusal_anchors or not compliance_anchors:
        raise ValueError("Both refusal_anchors and compliance_anchors must be non-empty")

    x = image_tensor.detach().clone().requires_grad_(True)

    ref_values: list[float] = []
    ref_grads: list[torch.Tensor] = []
    for a in refusal_anchors:
        term = _teacher_forced_loglik_tensor(model, x, prompt, a)
        g = torch.autograd.grad(term, x, retain_graph=False, create_graph=False)[0]
        ref_values.append(float(term.detach().item()))
        ref_grads.append(g.detach())

    comp_values: list[float] = []
    comp_grads: list[torch.Tensor] = []
    for a in compliance_anchors:
        term = _teacher_forced_loglik_tensor(model, x, prompt, a)
        g = torch.autograd.grad(term, x, retain_graph=False, create_graph=False)[0]
        comp_values.append(float(term.detach().item()))
        comp_grads.append(g.detach())

    ref_grad = _aggregate_grads(ref_values, ref_grads, aggregation, tau=aggregation_tau)
    comp_grad = _aggregate_grads(comp_values, comp_grads, aggregation, tau=aggregation_tau)
    return (ref_grad - comp_grad).detach()


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
