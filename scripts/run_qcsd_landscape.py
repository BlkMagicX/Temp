"""Run QCSD direction extraction and hidden-space landscape comparison.

Usage:
  python scripts/run_qcsd_landscape.py --config configs/qcsd_landscape.yaml
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover

    def tqdm(iterable=None, **_kwargs):
        return iterable


def _progress(msg: str) -> None:
    print(f"[qcsd] {msg}", flush=True)


def _ensure_repo_root_on_sys_path() -> Path:
    current_file = Path(__file__).resolve()
    repo_root = current_file.parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


def _parse_literal_value(raw: str) -> Any:
    value = raw.strip()
    lowered = value.lower()

    if lowered == "null":
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False

    if (value.startswith("[") and value.endswith("]")) or (value.startswith("{") and value.endswith("}")):
        return json.loads(value)

    if re.fullmatch(r"[+-]?\d+", value):
        return int(value)
    if re.fullmatch(r"[+-]?\d*\.\d+", value):
        return float(value)
    return value


def _set_deep(config: Dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cur: Dict[str, Any] = config
    for key in parts[:-1]:
        nxt = cur.get(key)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[key] = nxt
        cur = nxt
    cur[parts[-1]] = value


def _release_model(model: Any) -> None:
    if model is None:
        return

    if hasattr(model, "model"):
        model.model = None
    if hasattr(model, "processor"):
        model.processor = None
    if hasattr(model, "vllm_llm"):
        model.vllm_llm = None

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _set_reproducible(seed: int = 42) -> None:
    # Set deterministic behavior early to minimize nondeterministic kernels.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _build_dataset(data_cfg: Dict[str, Any]) -> Any:
    from src.data.clean_harmful_dataset import CleanHarmfulDataset
    from src.data.mm_safetybench_dataset import MMSafetyBenchDataset

    dataset_type = str(data_cfg.get("dataset_type", "clean_harmful")).lower()
    if dataset_type in {"clean_harmful", "clean-harmful"}:
        return CleanHarmfulDataset(
            annotation_path=data_cfg["annotation_path"],
            image_root=data_cfg.get("image_root"),
            check_image_exists=bool(data_cfg.get("check_image_exists", True)),
        )

    if dataset_type in {"mm_safetybench", "mm-safetybench", "mm_safety_bench"}:
        mm_cfg = dict(data_cfg.get("mm_safetybench", {}))
        return MMSafetyBenchDataset.from_config(
            mm_cfg,
            check_image_exists=bool(data_cfg.get("check_image_exists", True)),
        )

    raise ValueError(f"Unsupported data.dataset_type: {dataset_type}")


def _build_fp_model_cfg(models_cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(models_cfg.get("eval_model_template", {}))
    if not cfg:
        raise ValueError("models.eval_model_template is required")
    cfg["precision_mode"] = str(models_cfg.get("fp_precision_mode", "bf16"))
    cfg["backend_type"] = "bf16"
    cfg.pop("quant_model_path", None)
    return cfg


def _build_eval_model_cfg(models_cfg: Dict[str, Any], precision: str) -> Dict[str, Any]:
    cfg = dict(models_cfg.get("eval_model_template", {}))
    cfg["precision_mode"] = precision

    override = dict(models_cfg.get("precision_overrides", {}).get(precision, {}))
    if "quant_backend_config" in override:
        merged_qcfg = dict(cfg.get("quant_backend_config", {}))
        merged_qcfg.update(override.get("quant_backend_config") or {})
        cfg["quant_backend_config"] = merged_qcfg
        override.pop("quant_backend_config", None)
    for k, v in override.items():
        cfg[k] = v

    if "backend_type" not in cfg:
        cfg["backend_type"] = "bf16" if precision == "bf16" else "gptq"
    return cfg


def _normalize(vec: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    x = vec.detach().float()
    n = float(torch.linalg.norm(x.reshape(-1), ord=2).item())
    if n <= float(eps):
        return x
    return x / (n + float(eps))


def _sample_prompt(sample: Any) -> str:
    prompt = getattr(sample, "prompt", None)
    if prompt is None and isinstance(sample, dict):
        prompt = sample.get("prompt", sample.get("user_prompt"))
    return str(prompt)


def _sample_image(sample: Any) -> Image.Image:
    image_path = getattr(sample, "image_path", None)
    if image_path is None and isinstance(sample, dict):
        image_path = sample.get("image_path")
    if image_path is None:
        raise ValueError("sample has no image_path")
    return Image.open(Path(str(image_path)).resolve()).convert("RGB")


def _sample_id(sample: Any, idx: int) -> str:
    sid = getattr(sample, "sample_id", None)
    if sid is None and isinstance(sample, dict):
        sid = sample.get("sample_id")
    if sid is None:
        sid = f"sample_{idx}"
    return str(sid)


def _pooling_from_token_scope(token_scope: str) -> Optional[str]:
    scope = str(token_scope).lower().strip()
    if scope == "last":
        return "last_token"
    if scope == "first":
        return "cls"
    if scope == "mean":
        return "mean"
    return None


def _compute_summary_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[int, str, str, float], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        key = (
            int(r["layer_index"]),
            str(r["precision"]),
            str(r["direction_type"]),
            float(r["alpha"]),
        )
        grouped[key].append(r)

    out: List[Dict[str, Any]] = []
    for (layer_index, precision, direction_type, alpha), items in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1], x[0][2], x[0][3])):
        margin_fp = np.asarray([float(x["margin_fp"]) for x in items], dtype=np.float64)
        margin_q = np.asarray([float(x["margin_q"]) for x in items], dtype=np.float64)
        gap = np.asarray([float(x["gap_q_minus_fp"]) for x in items], dtype=np.float64)
        drop_fp = np.asarray([float(x["drop_fp"]) for x in items], dtype=np.float64)
        drop_q = np.asarray([float(x["drop_q"]) for x in items], dtype=np.float64)
        out.append(
            {
                "layer_index": int(layer_index),
                "precision": str(precision),
                "direction_type": str(direction_type),
                "alpha": float(alpha),
                "mean_margin_fp": float(np.mean(margin_fp)) if margin_fp.size else 0.0,
                "mean_margin_q": float(np.mean(margin_q)) if margin_q.size else 0.0,
                "mean_gap": float(np.mean(gap)) if gap.size else 0.0,
                "mean_drop_fp": float(np.mean(drop_fp)) if drop_fp.size else 0.0,
                "mean_drop_q": float(np.mean(drop_q)) if drop_q.size else 0.0,
                "std_drop_q": float(np.std(drop_q)) if drop_q.size else 0.0,
            }
        )
    return out


def _final_drop_by_direction(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    by_sample_dir: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_sample_dir[(str(r["sample_id"]), str(r["direction_type"]))].append(r)

    out: Dict[str, Dict[str, float]] = defaultdict(dict)
    for (sid, d_type), items in by_sample_dir.items():
        s_items = sorted(items, key=lambda x: float(x["alpha"]))
        out[sid][d_type] = float(s_items[-1]["drop_q"]) if s_items else 0.0
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run QCSD hidden-space landscape analysis")
    parser.add_argument("--config", type=str, default="configs/qcsd_landscape.yaml")
    parser.add_argument("--set", action="append", default=[], help="Override config by key=value")
    return parser.parse_args()


def main() -> None:
    _ensure_repo_root_on_sys_path()

    _set_reproducible()

    from src.algorithm.landscape import (
        make_alpha_grid_from_projection_quantiles,
        scan_hidden_direction_landscape_1d,
    )
    from src.algorithm.qcsd_direction import (
        compute_margin_gradient_surrogate,
        compute_qcsd_direction,
        load_drift_subspace,
    )
    from src.algorithm.representation_drift import extract_layer_representations
    from src.experiments.exp_boundary_drift_eval import load_experiment_config
    from src.models.model_factory import create_vlm

    args = parse_args()
    cfg = load_experiment_config(args.config)
    for item in args.set:
        if "=" not in item:
            raise ValueError(f"Invalid --set item: {item}. Expected key=value")
        k, raw_v = item.split("=", 1)
        _set_deep(cfg, k.strip(), _parse_literal_value(raw_v))

    exp_name = str(cfg.get("exp_name", "qcsd_landscape"))
    out_root = Path(str(dict(cfg.get("output", {})).get("root_dir", "outputs"))).resolve()
    exp_dir = out_root / exp_name / "qcsd_landscape"
    exp_dir.mkdir(parents=True, exist_ok=True)
    _progress(f"Experiment dir: {exp_dir}")

    data_cfg = dict(cfg.get("data", {}))
    anchors_cfg = dict(cfg.get("anchors", {}))
    models_cfg = dict(cfg.get("models", {}))
    q_cfg = dict(cfg.get("qcsd", {}))

    refusal_anchors = list(anchors_cfg.get("refusal_anchors", []))
    compliance_anchors = list(anchors_cfg.get("compliance_anchors", []))
    aggregation = str(anchors_cfg.get("aggregation", "logsumexp"))
    if not refusal_anchors or not compliance_anchors:
        raise ValueError("anchors.refusal_anchors and anchors.compliance_anchors must be non-empty")

    target_precision = str(q_cfg.get("target_precision", "w3a16"))
    layer_index = int(q_cfg.get("layer_index", -1))
    top_k = int(q_cfg.get("top_k", 3))
    token_scope = str(q_cfg.get("token_scope", "all"))
    eps = float(q_cfg.get("eps", 1e-12))
    max_samples = q_cfg.get("max_samples")
    candidate_sample_ids = [str(x) for x in list(q_cfg.get("candidate_sample_ids", []))]

    direction_file = Path(str(q_cfg.get("direction_file"))).resolve()
    if not direction_file.exists():
        raise FileNotFoundError(f"qcsd.direction_file not found: {direction_file}")

    dataset = _build_dataset(data_cfg)
    samples = list(dataset.samples)
    sample_map = {_sample_id(s, i): s for i, s in enumerate(samples)}
    if candidate_sample_ids:
        selected = [sample_map[sid] for sid in candidate_sample_ids if sid in sample_map]
    else:
        selected = list(samples)
    if max_samples is not None:
        selected = selected[: int(max_samples)]
    if not selected:
        raise ValueError("No samples selected for QCSD run")
    _progress(f"Selected samples: {len(selected)}")

    U_l = load_drift_subspace(
        layer_index=layer_index,
        direction_file=direction_file,
        top_k=top_k,
    )
    svd_first = _normalize(U_l[:, 0], eps=eps)

    fp_model = create_vlm(_build_fp_model_cfg(models_cfg))
    fp_model.load_model()

    q_model = create_vlm(_build_eval_model_cfg(models_cfg, target_precision))
    q_model.load_model()
    if str(getattr(q_model, "backend_type", "")).lower() == "vllm":
        raise RuntimeError("QCSD exact hidden-space scan does not support backend_type=vllm")

    alpha_cfg = dict(q_cfg.get("alpha", {}))
    alpha_mode = str(alpha_cfg.get("mode", "manual")).lower()
    alphas: List[float] = []

    if alpha_mode == "quantile_projection":
        pooling = _pooling_from_token_scope(token_scope)
        projections: List[float] = []
        if pooling is not None:
            _progress("Estimating alpha range from drift projection quantiles")
            for i, sample in enumerate(tqdm(selected, desc="[qcsd] projections", dynamic_ncols=True)):
                image = _sample_image(sample)
                prompt = _sample_prompt(sample)
                try:
                    fp_reps = extract_layer_representations(
                        model_wrapper=fp_model,
                        image=image,
                        prompt=prompt,
                        layer_indices=[layer_index],
                        pooling=pooling,
                    )
                    q_reps = extract_layer_representations(
                        model_wrapper=q_model,
                        image=image,
                        prompt=prompt,
                        layer_indices=[layer_index],
                        pooling=pooling,
                    )
                except RuntimeError:
                    fp_reps = {}
                    q_reps = {}
                if layer_index not in fp_reps or layer_index not in q_reps:
                    continue
                drift_vec = torch.as_tensor(q_reps[layer_index] - fp_reps[layer_index], dtype=torch.float32)
                if int(drift_vec.numel()) != int(svd_first.numel()):
                    continue
                projections.append(float(torch.dot(drift_vec.reshape(-1), svd_first.reshape(-1)).item()))
                if (i + 1) % 20 == 0:
                    _progress(f"Projection progress: {i + 1}/{len(selected)}")

        fallback = (
            float(alpha_cfg.get("alpha_min", -0.05)),
            float(alpha_cfg.get("alpha_max", 0.05)),
        )
        alphas = make_alpha_grid_from_projection_quantiles(
            projections=projections,
            n_steps=int(alpha_cfg.get("n_steps", 21)),
            q_low=float(alpha_cfg.get("quantile_low", 0.05)),
            q_high=float(alpha_cfg.get("quantile_high", 0.95)),
            fallback=fallback,
        )
    else:
        alpha_min = float(alpha_cfg.get("alpha_min", -0.05))
        alpha_max = float(alpha_cfg.get("alpha_max", 0.05))
        n_steps = max(2, int(alpha_cfg.get("n_steps", 21)))
        alphas = np.linspace(alpha_min, alpha_max, n_steps, dtype=np.float32).tolist()

    # Centered drop metrics require alpha=0 baseline.
    if not any(abs(float(a)) <= 1e-12 for a in alphas):
        alphas.append(0.0)
    alphas = sorted(set(float(a) for a in alphas))

    _progress(f"Alpha grid ready: n={len(alphas)}, min={float(min(alphas)):.6f}, max={float(max(alphas)):.6f}")

    by_sample_rows: List[Dict[str, Any]] = []
    qcsd_dirs: List[torch.Tensor] = []
    qcsd_sample_ids: List[str] = []

    direction_order = ["random", "svd_first", "pure_neg_grad", "qcsd"]

    try:
        for i, sample in enumerate(tqdm(selected, desc="[qcsd] samples", dynamic_ncols=True), start=1):
            sid = _sample_id(sample, i - 1)
            prompt = _sample_prompt(sample)
            image = _sample_image(sample)

            grad = compute_margin_gradient_surrogate(
                model_fp=fp_model,
                batch={
                    "image": image,
                    "prompt": prompt,
                    "refusal_anchors": refusal_anchors,
                    "compliance_anchors": compliance_anchors,
                    "aggregation": aggregation,
                    "token_scope": token_scope,
                },
                layer_index=layer_index,
            )
            grad_vec = grad[0] if grad.dim() > 1 else grad
            grad_vec = grad_vec.reshape(-1)
            if int(grad_vec.numel()) != int(U_l.shape[0]):
                raise ValueError(
                    "Gradient and subspace dimension mismatch: "
                    f"grad_dim={int(grad_vec.numel())}, U_dim={int(U_l.shape[0])}. "
                    "Please align token_scope/pooling with direction extraction."
                )

            pure_neg_grad = _normalize(-grad_vec, eps=eps)
            qcsd_dir = compute_qcsd_direction(gradient=grad_vec, U_l=U_l, eps=eps)
            qcsd_dir = _normalize(qcsd_dir.reshape(-1), eps=eps)

            random_dir = torch.randn_like(qcsd_dir)
            random_dir = _normalize(random_dir, eps=eps)

            directions = {
                "random": random_dir,
                "svd_first": svd_first.to(dtype=qcsd_dir.dtype),
                "pure_neg_grad": pure_neg_grad,
                "qcsd": qcsd_dir,
            }

            qcsd_dirs.append(qcsd_dir.detach().cpu())
            qcsd_sample_ids.append(sid)

            x0 = torch.zeros((1, 3, 16, 16), dtype=torch.float32, device=q_model.get_device())
            try:
                # We keep the image tensor path consistent with existing exact hidden scan.
                image_tensor = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).permute(2, 0, 1).contiguous().unsqueeze(0)
                x0 = image_tensor.to(q_model.get_device())
            except Exception:
                pass

            for d_type in direction_order:
                rows = scan_hidden_direction_landscape_1d(
                    fp_model=fp_model,
                    q_model=q_model,
                    base_image_tensor=x0,
                    prompt=prompt,
                    hidden_layer_index=layer_index,
                    direction_u=directions[d_type],
                    refusal_anchors=refusal_anchors,
                    compliance_anchors=compliance_anchors,
                    alphas=alphas,
                    aggregation=aggregation,
                    hidden_token_scope=token_scope,
                )
                for r in rows:
                    by_sample_rows.append(
                        {
                            "sample_id": sid,
                            "layer_index": int(layer_index),
                            "precision": target_precision,
                            "direction_type": d_type,
                            "alpha": float(r["alpha"]),
                            "margin_fp": float(r["margin_fp"]),
                            "margin_q": float(r["margin_q"]),
                            "gap_q_minus_fp": float(r["gap_q_minus_fp"]),
                            "drop_fp": float(r["drop_fp"]),
                            "drop_q": float(r["drop_q"]),
                        }
                    )

            if i % 5 == 0 or i == len(selected):
                _progress(f"Processed samples: {i}/{len(selected)}")

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        _release_model(q_model)
        _release_model(fp_model)

    summary_rows = _compute_summary_rows(by_sample_rows)

    final_by_sample = _final_drop_by_direction(by_sample_rows)
    random_drops: List[float] = []
    svd_drops: List[float] = []
    grad_drops: List[float] = []
    qcsd_drops: List[float] = []
    better_than_svd = 0
    better_than_random = 0
    comparable = 0
    for sid, dmap in final_by_sample.items():
        if all(k in dmap for k in ["random", "svd_first", "pure_neg_grad", "qcsd"]):
            comparable += 1
            random_drops.append(float(dmap["random"]))
            svd_drops.append(float(dmap["svd_first"]))
            grad_drops.append(float(dmap["pure_neg_grad"]))
            qcsd_drops.append(float(dmap["qcsd"]))
            if float(dmap["qcsd"]) < float(dmap["svd_first"]):
                better_than_svd += 1
            if float(dmap["qcsd"]) < float(dmap["random"]):
                better_than_random += 1

    cmp_payload = {
        "avg_final_drop_random": float(np.mean(random_drops)) if random_drops else 0.0,
        "avg_final_drop_svd": float(np.mean(svd_drops)) if svd_drops else 0.0,
        "avg_final_drop_grad": float(np.mean(grad_drops)) if grad_drops else 0.0,
        "avg_final_drop_qcsd": float(np.mean(qcsd_drops)) if qcsd_drops else 0.0,
        "qcsd_better_than_svd_rate": (float(better_than_svd) / float(comparable)) if comparable > 0 else 0.0,
        "qcsd_better_than_random_rate": (float(better_than_random) / float(comparable)) if comparable > 0 else 0.0,
        "n_comparable_samples": int(comparable),
    }

    qcsd_dir_path = exp_dir / f"qcsd_directions.layer_{layer_index}.pt"
    torch.save(
        {
            "layer_index": int(layer_index),
            "token_scope": str(token_scope),
            "sample_ids": list(qcsd_sample_ids),
            "directions": torch.stack(qcsd_dirs, dim=0) if qcsd_dirs else torch.empty((0, int(U_l.shape[0])), dtype=torch.float32),
        },
        qcsd_dir_path,
    )

    by_sample_csv = exp_dir / "qcsd_landscape.by_sample.csv"
    _write_csv(
        by_sample_csv,
        by_sample_rows,
        fieldnames=[
            "sample_id",
            "layer_index",
            "precision",
            "direction_type",
            "alpha",
            "margin_fp",
            "margin_q",
            "gap_q_minus_fp",
            "drop_fp",
            "drop_q",
        ],
    )

    summary_csv = exp_dir / "qcsd_landscape.summary.csv"
    _write_csv(
        summary_csv,
        summary_rows,
        fieldnames=[
            "layer_index",
            "precision",
            "direction_type",
            "alpha",
            "mean_margin_fp",
            "mean_margin_q",
            "mean_gap",
            "mean_drop_fp",
            "mean_drop_q",
            "std_drop_q",
        ],
    )

    comparison_json = exp_dir / "qcsd_direction_comparison.json"
    comparison_json.write_text(json.dumps(cmp_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    report = {
        "exp_name": exp_name,
        "output_dir": str(exp_dir),
        "precision": target_precision,
        "layer_index": int(layer_index),
        "n_samples": len(selected),
        "alpha_count": len(alphas),
        "qcsd_direction_file": str(qcsd_dir_path),
        "by_sample_csv": str(by_sample_csv),
        "summary_csv": str(summary_csv),
        "comparison_json": str(comparison_json),
        "comparison": cmp_payload,
    }
    run_report = exp_dir / "run_report.json"
    run_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    _progress("QCSD landscape analysis finished")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
