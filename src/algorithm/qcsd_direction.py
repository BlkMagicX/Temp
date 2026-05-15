"""QCSD direction extraction utilities.

This module implements Quantization-Constrained Safety Descent Direction (QCSD):
1) Learn quantization drift subspace U_l from saved SVD/PCA directions.
2) Compute surrogate margin gradient on differentiable fp/bf16 backend.
3) Project negative gradient onto U_l and normalize.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch


def _aggregate_scores_tensor(scores: torch.Tensor, method: str = "logsumexp") -> torch.Tensor:
    if scores.numel() == 0:
        return torch.zeros((), dtype=torch.float32, device=scores.device)

    method_norm = str(method).lower().strip()
    if method_norm == "logsumexp":
        return torch.logsumexp(scores, dim=0) - math.log(float(scores.shape[0]))
    if method_norm == "mean":
        return scores.mean(dim=0)
    if method_norm == "max":
        return scores.max(dim=0).values
    raise ValueError(f"Unsupported aggregation method: {method}")


def _sequence_loglikelihood_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if logits.dim() != 3 or labels.dim() != 2:
        raise ValueError("Expected logits [B,T,V] and labels [B,T]")
    if int(logits.shape[0]) != int(labels.shape[0]) or int(logits.shape[1]) != int(labels.shape[1]):
        raise ValueError(f"logits/labels shape mismatch: logits={tuple(logits.shape)}, labels={tuple(labels.shape)}")

    if int(logits.shape[1]) <= 1:
        return torch.zeros((int(logits.shape[0]),), device=logits.device, dtype=logits.dtype)

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    valid_mask = shift_labels != -100
    safe_labels = shift_labels.masked_fill(~valid_mask, 0)

    token_nll = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        safe_labels.view(-1),
        reduction="none",
    )
    token_nll = token_nll.view_as(shift_labels)
    token_nll = token_nll * valid_mask.to(dtype=token_nll.dtype)
    token_counts = valid_mask.to(dtype=token_nll.dtype).sum(dim=1).clamp(min=1)
    return -token_nll.sum(dim=1) / token_counts


def _normalize_batch_vector(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    if x.dim() == 1:
        x = x.unsqueeze(0)
    flat = x.view(int(x.shape[0]), -1)
    n = torch.linalg.norm(flat, ord=2, dim=1, keepdim=True)
    out = flat / (n + float(eps))
    return out.view_as(x)


def _resolve_layer_index(layer_index: int, n_layers: int) -> int:
    idx = int(layer_index)
    if idx < 0:
        idx = int(n_layers) + idx
    if idx < 0 or idx >= int(n_layers):
        raise ValueError(f"layer_index out of range: {layer_index}, n_layers={n_layers}")
    return idx


def _pool_hidden_tensor(hidden: torch.Tensor, token_scope: str) -> torch.Tensor:
    scope = str(token_scope).lower().strip()
    if hidden.dim() != 3:
        raise ValueError(f"Expected hidden tensor [B,T,H], got ndim={hidden.dim()}")

    if scope == "last":
        return hidden[:, -1, :]
    if scope == "first":
        return hidden[:, 0, :]
    if scope == "mean":
        return hidden.mean(dim=1)
    if scope == "all":
        # Sum token gradients to match applying the same delta to all tokens.
        return hidden.sum(dim=1)
    raise ValueError("token_scope must be one of: last/first/mean/all")


def load_drift_subspace(
    layer_index: int,
    direction_file: str | Path,
    top_k: int,
) -> torch.Tensor:
    """Load and orthonormalize drift subspace basis U_l.

    Returns:
        U_l with shape [D, K], columns are orthonormal basis vectors.

    """
    payload = torch.load(Path(direction_file).resolve(), map_location="cpu")
    file_layer = payload.get("layer_index", None) if isinstance(payload, dict) else None
    if file_layer is not None and int(file_layer) != int(layer_index):
        raise ValueError("direction file layer mismatch: " f"expected={int(layer_index)}, file={int(file_layer)}")

    dirs = payload.get("directions")

    if not isinstance(dirs, torch.Tensor):
        dirs = torch.as_tensor(dirs, dtype=torch.float32)
    dirs = dirs.detach().float().cpu()

    k = min(int(top_k), int(dirs.shape[0]))

    basis = dirs[:k].transpose(0, 1).contiguous()  # [D,K]
    q, _ = torch.linalg.qr(basis, mode="reduced")
    return q.detach().float()


def compute_margin_gradient_surrogate(
    model_fp: Any,
    batch: Dict[str, Any],
    layer_index: int,
) -> torch.Tensor:
    """Compute surrogate gradient g_l(z)=d m_fp / d h_l on differentiable backend.

    Required batch keys:
      - image
      - prompt
      - refusal_anchors
      - compliance_anchors

    Optional batch keys:
      - aggregation (default: logsumexp)
      - token_scope (default: last)
    """
    backend_type = str(getattr(model_fp, "backend_type", "bf16")).lower()
    if backend_type in {"gptq", "vllm"}:
        raise RuntimeError("compute_margin_gradient_surrogate requires differentiable fp/bf16 backend")

    model_fp.load_model()
    model = getattr(model_fp, "model", None)
    processor = getattr(model_fp, "processor", None)
    if model is None or processor is None:
        raise RuntimeError("model_fp must expose torch model and processor")

    image = batch["image"]
    prompt = str(batch["prompt"])
    refusal_anchors = [str(x) for x in list(batch["refusal_anchors"])]
    compliance_anchors = [str(x) for x in list(batch["compliance_anchors"])]
    aggregation = str(batch.get("aggregation", "logsumexp"))
    token_scope = str(batch.get("token_scope", "all"))

    if not refusal_anchors or not compliance_anchors:
        raise ValueError("refusal_anchors and compliance_anchors must be non-empty")

    build_chat = getattr(model_fp, "_build_user_chat_text", None)
    align_batch = getattr(model_fp, "_align_batch_for_model", None)
    if build_chat is None or align_batch is None:
        raise RuntimeError("model_fp wrapper does not expose expected preprocessing helpers")

    if isinstance(image, torch.Tensor):
        to_pil = getattr(model_fp, "_tensor_to_pil", None)
        if to_pil is None:
            raise RuntimeError("model_fp wrapper cannot convert tensor image to PIL")
        image = to_pil(image)
    if hasattr(image, "convert"):
        image = image.convert("RGB")

    prefix_text = build_chat(prompt, True)
    prefix_inputs = processor(
        text=[prefix_text],
        images=[image],
        return_tensors="pt",
    )

    resolve_layers = getattr(model_fp, "_resolve_text_decoder_layers", None)
    normalize_layer = getattr(model_fp, "_normalize_layer_index", None)
    if callable(resolve_layers):
        layers = list(resolve_layers())
    else:
        core = getattr(model, "model", None)
        language_model = getattr(core, "language_model", None)
        layers = list(getattr(language_model, "layers", []))
    if not layers:
        raise RuntimeError("Failed to locate decoder layers for hidden gradient extraction")

    if callable(normalize_layer):
        target_idx = int(normalize_layer(int(layer_index), len(layers)))
    else:
        target_idx = _resolve_layer_index(layer_index=int(layer_index), n_layers=len(layers))

    def _prepare_single_target_inputs(target_text: str) -> Dict[str, Any]:
        full_text = prefix_text + str(target_text)
        try:
            model_inputs = processor(
                text=[full_text],
                images=[image],
                return_tensors="pt",
                padding=True,
            )
        except TypeError:
            model_inputs = processor(
                text=[full_text],
                images=[image],
                return_tensors="pt",
            )

        prefix_len = int(prefix_inputs["input_ids"].shape[-1])
        labels = model_inputs["input_ids"].clone()
        labels[:, :prefix_len] = -100

        attn_mask = model_inputs.get("attention_mask")
        if isinstance(attn_mask, torch.Tensor):
            labels = labels.masked_fill(attn_mask == 0, -100)

        model_inputs["labels"] = labels
        model_inputs["return_dict"] = True
        return align_batch(model_inputs)

    def _aggregate_grads(values: Sequence[float], grads: Sequence[torch.Tensor], method: str) -> torch.Tensor:
        if not values or not grads:
            raise ValueError("values and grads must be non-empty")
        if len(values) != len(grads):
            raise ValueError("values and grads size mismatch")

        method_norm = str(method).lower().strip()
        if method_norm == "mean":
            out = torch.zeros_like(grads[0])
            for g in grads:
                out = out + g
            return out / float(len(grads))
        if method_norm == "max":
            idx = int(torch.argmax(torch.tensor(values, dtype=torch.float32)).item())
            return grads[idx]
        if method_norm == "logsumexp":
            v = torch.tensor(values, dtype=grads[0].dtype, device=grads[0].device)
            w = torch.softmax(v, dim=0)
            out = torch.zeros_like(grads[0])
            for i, g in enumerate(grads):
                out = out + w[i] * g
            return out
        raise ValueError(f"Unsupported aggregation method: {method}")

    def _single_anchor_grad(target_text: str) -> tuple[float, torch.Tensor]:
        captured: Dict[str, torch.Tensor] = {}

        def _hook(_module: Any, _inputs: Any, output: Any) -> Any:
            hidden = output[0] if isinstance(output, tuple) else output
            if isinstance(hidden, torch.Tensor):
                captured["h"] = hidden
            return output

        handle = layers[target_idx].register_forward_hook(_hook)
        try:
            model_inputs = _prepare_single_target_inputs(target_text=target_text)
            labels_aligned = model_inputs["labels"]

            outputs = model(**model_inputs)
            logits = outputs.logits
            h = captured.get("h")
            if h is None:
                raise RuntimeError("Failed to capture target-layer hidden state")
            h.retain_grad()

            seq_ll = _sequence_loglikelihood_from_logits(logits=logits, labels=labels_aligned)
            if int(seq_ll.shape[0]) != 1:
                raise RuntimeError(f"Unexpected sequence score batch size: {int(seq_ll.shape[0])}")
            term = seq_ll[0]
            grad_h = torch.autograd.grad(
                term,
                h,
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )[0]
            if grad_h is None:
                raise RuntimeError("Failed to compute hidden-state gradient for anchor")

            pooled = _pool_hidden_tensor(grad_h.detach(), token_scope=token_scope)
            if pooled.dim() == 2:
                pooled = pooled[0]
            return float(term.detach().item()), pooled.detach()
        finally:
            handle.remove()

    ref_values: List[float] = []
    ref_grads: List[torch.Tensor] = []
    for a in refusal_anchors:
        value, grad = _single_anchor_grad(target_text=a)
        ref_values.append(value)
        ref_grads.append(grad)

    comp_values: List[float] = []
    comp_grads: List[torch.Tensor] = []
    for a in compliance_anchors:
        value, grad = _single_anchor_grad(target_text=a)
        comp_values.append(value)
        comp_grads.append(grad)

    ref_grad = _aggregate_grads(ref_values, ref_grads, aggregation)
    comp_grad = _aggregate_grads(comp_values, comp_grads, aggregation)
    return (ref_grad - comp_grad).detach()


def compute_qcsd_direction(
    gradient: torch.Tensor,
    U_l: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Compute QCSD direction from gradient and drift subspace basis."""
    if not isinstance(gradient, torch.Tensor):
        gradient = torch.as_tensor(gradient, dtype=torch.float32)
    if not isinstance(U_l, torch.Tensor):
        U_l = torch.as_tensor(U_l, dtype=torch.float32)

    g = gradient.detach().float()
    U = U_l.detach().float().to(device=g.device)
    if U.dim() != 2:
        raise ValueError(f"U_l must be 2D [D,K], got shape={tuple(U.shape)}")

    single = g.dim() == 1
    g2 = g.unsqueeze(0) if single else g.reshape(int(g.shape[0]), -1)

    d = int(g2.shape[1])
    if int(U.shape[0]) != d:
        raise ValueError(f"Dimension mismatch: gradient_dim={d}, U_dim={int(U.shape[0])}")

    neg = -g2
    proj_coef = torch.matmul(neg, U)  # [B,K]
    proj = torch.matmul(proj_coef, U.transpose(0, 1))  # [B,D]

    n = torch.linalg.norm(proj, ord=2, dim=1, keepdim=True)
    q = proj / (n + float(eps))
    if single:
        return q[0]
    return q.view_as(g)


def compare_directions(
    direction_scans: Dict[str, Sequence[Dict[str, Any]]],
    drop_key: str = "drop_q",
) -> Dict[str, float]:
    """Compare final margin drop among different directions.

    Args:
        direction_scans: mapping direction_type -> scan rows sorted by alpha.
        drop_key: key used for direction quality (more negative is better).
    """
    out: Dict[str, float] = {}
    for k, rows in direction_scans.items():
        if not rows:
            out[f"final_{drop_key}_{k}"] = 0.0
            continue
        rows_sorted = sorted(rows, key=lambda x: float(x.get("alpha", 0.0)))
        out[f"final_{drop_key}_{k}"] = float(rows_sorted[-1].get(drop_key, 0.0))
    return out


def apply_qcsd_projection_correction(
    hidden_state: torch.Tensor,
    qcsd_direction: torch.Tensor,
    lambda_value: float,
    reference_hidden_state: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Placeholder defense hook for QCSD projection correction.

    Current default uses h_ref=0 when `reference_hidden_state` is not provided.
    """
    if not isinstance(hidden_state, torch.Tensor):
        hidden_state = torch.as_tensor(hidden_state, dtype=torch.float32)
    if not isinstance(qcsd_direction, torch.Tensor):
        qcsd_direction = torch.as_tensor(qcsd_direction, dtype=torch.float32)

    h = hidden_state
    u = qcsd_direction.to(device=h.device, dtype=h.dtype)
    if reference_hidden_state is None:
        h_ref = torch.zeros_like(h)
    else:
        h_ref = reference_hidden_state.to(device=h.device, dtype=h.dtype)

    delta = h - h_ref
    if delta.dim() == 1:
        coeff = torch.dot(delta, u)
        return h - float(lambda_value) * coeff * u

    flat_delta = delta.view(int(delta.shape[0]), -1)
    flat_u = u.view(1, -1)
    coeff = torch.sum(flat_delta * flat_u, dim=1, keepdim=True)
    corrected = flat_delta - float(lambda_value) * coeff * flat_u
    return corrected.view_as(h)
