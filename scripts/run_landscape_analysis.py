"""Run margin landscape analysis on selected candidates.

Usage:
  python scripts/run_landscape_analysis.py --config configs/landscape.yaml
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image


def _progress(msg: str) -> None:
    print(f"[landscape] {msg}", flush=True)


def _format_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "--:--"
    s = max(0, int(round(float(seconds))))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{sec:02d}"
    return f"{m:02d}:{sec:02d}"


def _render_progress_bar(done: int, total: int, width: int = 24) -> str:
    total_safe = max(1, int(total))
    done_clamped = min(max(int(done), 0), total_safe)
    ratio = float(done_clamped) / float(total_safe)
    filled = int(round(ratio * width))
    return "#" * filled + "-" * (width - filled)


class _TimeProgress:
    def __init__(self, stage: str, total: int) -> None:
        self.stage = stage
        self.total = max(1, int(total))
        self._start = time.perf_counter()

    def update(self, done: int, detail: str = "") -> None:
        done_clamped = min(max(int(done), 0), self.total)
        elapsed = time.perf_counter() - self._start
        eta = None
        if done_clamped > 0:
            avg = elapsed / float(done_clamped)
            eta = avg * float(self.total - done_clamped)

        pct = 100.0 * float(done_clamped) / float(self.total)
        bar = _render_progress_bar(done_clamped, self.total)
        suffix = f" | {detail}" if detail else ""
        _progress(
            f"[{self.stage}] |{bar}| {done_clamped}/{self.total} ({pct:5.1f}%) "
            f"elapsed={_format_duration(elapsed)} eta={_format_duration(eta)}{suffix}"
        )


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


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _compute_attack_entry_score(m_q: Optional[float], delta_q: Optional[float], eps: float = 1e-8) -> Optional[float]:
    if m_q is None or delta_q is None:
        return None
    denom = abs(float(m_q)) + float(eps)
    if denom <= 0.0:
        return None
    return float((-float(delta_q)) / denom)


def _pil_to_unit_tensor(image: Image.Image, device: torch.device) -> torch.Tensor:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous().unsqueeze(0)
    return t.to(device)


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


def _select_candidates_from_boundary_csv(
    csv_path: Path,
    top_k: int,
) -> List[str]:
    best_score_by_sample: Dict[str, float] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            sample_id = str(r.get("sample_id", "")).strip()
            if not sample_id:
                continue

            score = _to_float(r.get("attack_entry_score"))
            if score is None:
                m_q = _to_float(r.get("m_q"))
                delta_q = _to_float(r.get("delta_q"))
                score = _compute_attack_entry_score(m_q=m_q, delta_q=delta_q)
            if score is None:
                continue

            prev = best_score_by_sample.get(sample_id)
            if prev is None or float(score) > float(prev):
                best_score_by_sample[sample_id] = float(score)

    ranked = sorted(best_score_by_sample.items(), key=lambda x: float(x[1]), reverse=True)
    return [str(sid) for sid, _ in ranked[: max(1, int(top_k))]]


def _build_direction(
    direction_mode: str,
    model: Any,
    x0: torch.Tensor,
    prompt: str,
    refusal_anchors: List[str],
    compliance_anchors: List[str],
) -> torch.Tensor:
    from src.algorithm.sensitivity import compute_margin_gradient

    mode = str(direction_mode).lower()
    if mode == "random":
        return torch.randn_like(x0)

    return compute_margin_gradient(
        model=model,
        image_tensor=x0,
        prompt=prompt,
        refusal_anchors=refusal_anchors,
        compliance_anchors=compliance_anchors,
        aggregation="logsumexp",
    )


def _load_hidden_direction_vector(direction_file: Path, direction_index: int) -> Tuple[torch.Tensor, int]:
    payload = torch.load(direction_file, map_location="cpu")
    directions = payload.get("directions")
    if directions is None:
        raise ValueError(f"Invalid direction file (missing 'directions'): {direction_file}")

    if not isinstance(directions, torch.Tensor):
        directions = torch.as_tensor(directions, dtype=torch.float32)
    directions = directions.detach().float().cpu()
    if directions.ndim != 2:
        raise ValueError(f"Invalid direction tensor ndim={directions.ndim}, expected 2 in {direction_file}")

    idx = int(direction_index)
    if idx < 0:
        idx += int(directions.shape[0])
    if idx < 0 or idx >= int(directions.shape[0]):
        raise ValueError(f"direction_index out of range: {direction_index}, available=[0,{int(directions.shape[0]) - 1}]")

    layer_index = payload.get("layer_index")
    layer_idx = int(layer_index) if layer_index is not None else -1
    return directions[idx].clone(), layer_idx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run margin landscape analysis")
    parser.add_argument("--config", type=str, default="configs/landscape.yaml")
    parser.add_argument("--set", action="append", default=[], help="Override config by key=value")
    return parser.parse_args()


def main() -> None:
    _ensure_repo_root_on_sys_path()

    from src.algorithm.landscape import (
        estimate_derivative_and_curvature,
        make_linear_grid,
        scan_landscape_1d,
        scan_landscape_2d,
    )
    from src.experiments.exp_boundary_drift_eval import load_experiment_config
    from src.models.model_factory import create_vlm

    args = parse_args()
    _progress(f"Loading config: {args.config}")
    cfg = load_experiment_config(args.config)
    for item in args.set:
        if "=" not in item:
            raise ValueError(f"Invalid --set item: {item}. Expected key=value")
        k, raw_v = item.split("=", 1)
        _set_deep(cfg, k.strip(), _parse_literal_value(raw_v))

    exp_name = str(cfg.get("exp_name", "landscape_analysis"))
    out_root = Path(str(dict(cfg.get("output", {})).get("root_dir", "outputs"))).resolve()
    exp_dir = out_root / exp_name / "landscape"
    exp_dir.mkdir(parents=True, exist_ok=True)
    _progress(f"Experiment dir: {exp_dir}")

    data_cfg = dict(cfg.get("data", {}))
    anchors_cfg = dict(cfg.get("anchors", {}))
    models_cfg = dict(cfg.get("models", {}))
    l_cfg = dict(cfg.get("landscape", {}))

    refusal_anchors = list(anchors_cfg.get("refusal_anchors", []))
    compliance_anchors = list(anchors_cfg.get("compliance_anchors", []))
    aggregation = str(anchors_cfg.get("aggregation", "logsumexp"))
    if not refusal_anchors or not compliance_anchors:
        raise ValueError("anchors.refusal_anchors and anchors.compliance_anchors must be non-empty")

    target_precision = str(l_cfg.get("target_precision", "w4a16"))
    top_k = int(l_cfg.get("top_k", 20))
    candidate_ids = list(l_cfg.get("candidate_sample_ids", []))

    _progress("Building dataset")
    dataset = _build_dataset(data_cfg)
    samples = list(dataset.samples)
    sample_map = {str(s.sample_id): s for s in samples}
    _progress(f"Dataset ready: {len(samples)} samples")

    if not candidate_ids:
        boundary_csv = l_cfg.get("boundary_csv")
        if not boundary_csv:
            raise ValueError("landscape.boundary_csv is required when candidate_sample_ids is empty")
        _progress(f"Selecting candidates from boundary CSV (top_k={top_k})")
        candidate_ids = _select_candidates_from_boundary_csv(
            csv_path=Path(str(boundary_csv)).resolve(),
            top_k=top_k,
        )
        _progress(f"Boundary selection done: {len(candidate_ids)} candidates")

    selected = [sample_map[sid] for sid in candidate_ids if sid in sample_map]
    if not selected:
        raise ValueError(
            "No candidate samples found in dataset. "
            f"dataset_samples={len(samples)}, candidate_ids={len(candidate_ids)}. "
            "Please align data.mm_safetybench.scenarios with landscape.boundary_csv source, "
            "or set landscape.candidate_sample_ids explicitly."
        )
    _progress(f"Selected {len(selected)} samples for landscape scan")

    alpha_grid = make_linear_grid(max_abs=float(l_cfg.get("alpha_max", 0.05)), n=int(l_cfg.get("alpha_steps", 21)))
    beta_steps = int(l_cfg.get("beta_steps", 0))
    beta_max = float(l_cfg.get("beta_max", 0.0))
    beta_grid = make_linear_grid(max_abs=beta_max, n=beta_steps) if beta_steps > 1 and beta_max > 0.0 else []
    two_d_progress_every = max(1, int(l_cfg.get("two_d_progress_every", 50)))

    direction_mode = str(l_cfg.get("direction_mode", "margin_grad"))
    gradient_source = str(l_cfg.get("gradient_source", "fp"))
    mode = str(l_cfg.get("mode", "approximate"))

    exact_cfg = dict(l_cfg.get("exact", {}))
    exact_layer_raw = exact_cfg.get("layer_index", l_cfg.get("exact_layer_index"))
    exact_layer_index: Optional[int] = int(exact_layer_raw) if exact_layer_raw is not None else None
    exact_token_scope = str(exact_cfg.get("token_scope", l_cfg.get("exact_token_scope", "all")))
    exact_u_direction_file = exact_cfg.get("u_direction_file", l_cfg.get("exact_u_direction_file"))
    exact_u_direction_index = int(exact_cfg.get("u_direction_index", l_cfg.get("exact_u_direction_index", 0)))
    exact_v_direction_file = exact_cfg.get("v_direction_file", l_cfg.get("exact_v_direction_file"))
    exact_v_direction_index = int(exact_cfg.get("v_direction_index", l_cfg.get("exact_v_direction_index", 1)))

    direction_cache: Dict[str, torch.Tensor] = {}
    exact_u_cpu: Optional[torch.Tensor] = None
    exact_v_cpu: Optional[torch.Tensor] = None

    compute_margin_fp = bool(l_cfg.get("compute_margin_fp", False))
    need_fp_for_direction = (
        str(mode).lower() != "exact"
        and str(direction_mode).lower() != "random"
        and str(gradient_source).lower() == "fp"
    )
    need_fp_model = need_fp_for_direction or compute_margin_fp

    fp_model = None
    if need_fp_model:
        _progress("Loading FP model")
        fp_model = create_vlm(_build_fp_model_cfg(models_cfg))
        fp_model.load_model()

    if need_fp_for_direction:
        try:
            n_selected = len(selected)
            fp_prog = _TimeProgress(stage="FP direction", total=n_selected)
            for idx, sample in enumerate(selected, start=1):
                sid = str(sample.sample_id)
                prompt = str(sample.prompt)
                _progress(f"[FP direction] {idx}/{n_selected} sample_id={sid}")
                img = Image.open(Path(str(sample.image_path)).resolve()).convert("RGB")
                x_fp = _pil_to_unit_tensor(img, device=fp_model.get_device())
                u = _build_direction(
                    direction_mode=direction_mode,
                    model=fp_model,
                    x0=x_fp,
                    prompt=prompt,
                    refusal_anchors=refusal_anchors,
                    compliance_anchors=compliance_anchors,
                )
                direction_cache[sid] = u.detach().cpu()
                fp_prog.update(idx, detail=f"sample_id={sid}")
        finally:
            if not compute_margin_fp:
                _progress("Releasing FP model (not needed for margin scan)")
                _release_model(fp_model)
                fp_model = None

    _progress(f"Loading target model: precision={target_precision}")
    q_model = create_vlm(_build_eval_model_cfg(models_cfg, target_precision))
    q_model.load_model()

    if str(mode).lower() == "exact":
        if exact_u_direction_file is None:
            raise ValueError("landscape.exact.u_direction_file is required when landscape.mode=exact")

        u_file = Path(str(exact_u_direction_file)).resolve()
        exact_u_cpu, inferred_layer_u = _load_hidden_direction_vector(
            direction_file=u_file,
            direction_index=exact_u_direction_index,
        )
        if exact_layer_index is None:
            exact_layer_index = inferred_layer_u
        if inferred_layer_u >= 0 and int(exact_layer_index) != int(inferred_layer_u):
            _progress(f"[exact] Warning: configured layer_index={exact_layer_index} differs from " f"u_direction_file layer_index={inferred_layer_u}")

        if beta_grid:
            if exact_v_direction_file is not None:
                v_file = Path(str(exact_v_direction_file)).resolve()
                exact_v_cpu, inferred_layer_v = _load_hidden_direction_vector(
                    direction_file=v_file,
                    direction_index=exact_v_direction_index,
                )
                if inferred_layer_v >= 0 and int(exact_layer_index) != int(inferred_layer_v):
                    raise ValueError(
                        "landscape.exact.v_direction_file layer_index does not match landscape.exact.layer_index: "
                        f"{inferred_layer_v} vs {exact_layer_index}"
                    )
            else:
                v_rand = torch.randn_like(exact_u_cpu)
                proj = (v_rand * exact_u_cpu).sum() / (exact_u_cpu.norm(p=2) ** 2 + 1e-12)
                exact_v_cpu = (v_rand - proj * exact_u_cpu).detach().cpu()

            if exact_v_cpu is not None and float(exact_v_cpu.norm(p=2).item()) <= 1e-12:
                exact_v_cpu = torch.randn_like(exact_u_cpu)

        _progress(f"[exact] enabled layer_index={exact_layer_index}, token_scope={exact_token_scope}, " f"u_file={u_file}")

    rows_1d: List[Dict[str, Any]] = []
    rows_2d: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    try:
        n_selected = len(selected)
        scan_prog = _TimeProgress(stage="Scan", total=n_selected)
        for idx, sample in enumerate(selected, start=1):
            sid = str(sample.sample_id)
            prompt = str(sample.prompt)
            _progress(f"[Scan] {idx}/{n_selected} sample_id={sid}: preparing input")
            img = Image.open(Path(str(sample.image_path)).resolve()).convert("RGB")
            x0 = _pil_to_unit_tensor(img, device=q_model.get_device())

            if str(mode).lower() == "exact":
                if exact_u_cpu is None or exact_layer_index is None:
                    raise RuntimeError("exact mode directions are not initialized")
                u = exact_u_cpu.to(device=x0.device)
            elif sid in direction_cache:
                u = direction_cache[sid].to(device=x0.device)
            else:
                _progress(f"[Scan] {idx}/{n_selected} sample_id={sid}: building direction")
                u = _build_direction(
                    direction_mode=direction_mode,
                    model=q_model,
                    x0=x0,
                    prompt=prompt,
                    refusal_anchors=refusal_anchors,
                    compliance_anchors=compliance_anchors,
                )

            _progress(f"[Scan] {idx}/{n_selected} sample_id={sid}: running 1D scan")
            scan_1d = scan_landscape_1d(
                model=q_model,
                base_image_tensor=x0,
                prompt=prompt,
                direction_u=u,
                refusal_anchors=refusal_anchors,
                compliance_anchors=compliance_anchors,
                alphas=alpha_grid,
                aggregation=aggregation,
                model_key=target_precision,
                mode=mode,
                hidden_layer_index=(int(exact_layer_index) if str(mode).lower() == "exact" else None),
                hidden_token_scope=exact_token_scope,
                fp_model=fp_model,
            )
            diff_stats = estimate_derivative_and_curvature(scan_1d)

            margin_fp_at_zero = None
            for item in scan_1d:
                row = {
                    "sample_id": sid,
                    "precision": target_precision,
                    "alpha": float(item["alpha"]),
                    "margin": float(item["margin"]),
                    "mode": str(item.get("mode", mode)),
                }
                if "margin_fp" in item:
                    row["margin_fp"] = float(item["margin_fp"])
                    if float(item["alpha"]) == 0.0 and margin_fp_at_zero is None:
                        margin_fp_at_zero = float(item["margin_fp"])
                rows_1d.append(row)

            summary_row: Dict[str, Any] = {
                "sample_id": sid,
                "precision": target_precision,
                "directional_derivative": float(diff_stats.get("directional_derivative", 0.0)),
                "curvature": float(diff_stats.get("curvature", 0.0)),
            }
            if margin_fp_at_zero is not None:
                summary_row["margin_fp_at_zero"] = margin_fp_at_zero
            summary_rows.append(summary_row)
            _progress(f"[Scan] {idx}/{n_selected} sample_id={sid}: 1D scan done")

            if beta_grid:
                _progress(f"[Scan] {idx}/{n_selected} sample_id={sid}: running 2D scan")
                if str(mode).lower() == "exact":
                    if exact_v_cpu is None:
                        v = torch.randn_like(u)
                        proj = (v * u).sum() / (u.norm(p=2) ** 2 + 1e-12)
                        v = v - proj * u
                    else:
                        v = exact_v_cpu.to(device=x0.device)
                else:
                    v = torch.randn_like(u)
                    proj = (v * u).sum() / (u.norm(p=2) ** 2 + 1e-12)
                    v = v - proj * u

                points_total = max(1, len(alpha_grid) * len(beta_grid))
                two_d_prog = _TimeProgress(stage=f"2D {sid}", total=points_total)

                def _on_2d_progress(info: Dict[str, Any]) -> None:
                    point_idx = int(info.get("point_index", 0))
                    if point_idx <= 0:
                        return
                    n_rows = max(1, int(info.get("n_rows", 1)))
                    n_cols = max(1, int(info.get("n_cols", 1)))
                    row = int(info.get("row_index", 0)) + 1
                    col = int(info.get("col_index", 0)) + 1

                    # Print first point, each row start/end, and periodic points.
                    is_row_edge = (col == 1) or (col == n_cols)
                    should_log = (point_idx == 1) or is_row_edge or (point_idx % two_d_progress_every == 0) or (point_idx == points_total)
                    if not should_log:
                        return

                    alpha = float(info.get("alpha", 0.0))
                    beta = float(info.get("beta", 0.0))
                    detail = f"row={row}/{n_rows}, col={col}/{n_cols}, alpha={alpha:.4f}, beta={beta:.4f}"
                    two_d_prog.update(point_idx, detail=detail)

                scan_2d = scan_landscape_2d(
                    model=q_model,
                    base_image_tensor=x0,
                    prompt=prompt,
                    direction_u=u,
                    direction_v=v,
                    refusal_anchors=refusal_anchors,
                    compliance_anchors=compliance_anchors,
                    alphas=alpha_grid,
                    betas=beta_grid,
                    aggregation=aggregation,
                    model_key=target_precision,
                    mode=mode,
                    progress_callback=_on_2d_progress,
                    hidden_layer_index=(int(exact_layer_index) if str(mode).lower() == "exact" else None),
                    hidden_token_scope=exact_token_scope,
                    fp_model=fp_model,
                )
                for item in scan_2d:
                    row = {
                        "sample_id": sid,
                        "precision": target_precision,
                        "alpha": float(item["alpha"]),
                        "beta": float(item["beta"]),
                        "margin": float(item["margin"]),
                        "mode": str(item.get("mode", mode)),
                    }
                    if "margin_fp" in item:
                        row["margin_fp"] = float(item["margin_fp"])
                    rows_2d.append(row)
                _progress(f"[Scan] {idx}/{n_selected} sample_id={sid}: 2D scan done")
            scan_prog.update(idx, detail=f"sample_id={sid}")

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        _progress("Releasing target model")
        _release_model(q_model)
        if fp_model is not None:
            _progress("Releasing FP model")
            _release_model(fp_model)

    scan_1d_fieldnames = ["sample_id", "precision", "alpha", "margin", "mode"]
    if compute_margin_fp:
        scan_1d_fieldnames.append("margin_fp")
    one_d_csv = exp_dir / "landscape_scan_1d.csv"
    _progress(f"Writing 1D CSV: {one_d_csv}")
    _write_csv(one_d_csv, rows_1d, scan_1d_fieldnames)

    summary_fieldnames = ["sample_id", "precision", "directional_derivative", "curvature"]
    if compute_margin_fp:
        summary_fieldnames.append("margin_fp_at_zero")
    summary_csv = exp_dir / "landscape_summary.csv"
    _progress(f"Writing summary CSV: {summary_csv}")
    _write_csv(summary_csv, summary_rows, summary_fieldnames)

    two_d_csv = None
    if rows_2d:
        scan_2d_fieldnames = ["sample_id", "precision", "alpha", "beta", "margin", "mode"]
        if compute_margin_fp:
            scan_2d_fieldnames.append("margin_fp")
        two_d_csv = exp_dir / "landscape_scan_2d.csv"
        _progress(f"Writing 2D CSV: {two_d_csv}")
        _write_csv(two_d_csv, rows_2d, scan_2d_fieldnames)

    report = {
        "exp_name": exp_name,
        "output_dir": str(exp_dir),
        "target_precision": target_precision,
        "mode": mode,
        "compute_margin_fp": compute_margin_fp,
        "n_candidates": len(selected),
        "scan_1d_csv": str(one_d_csv),
        "summary_csv": str(summary_csv),
        "scan_2d_csv": (str(two_d_csv) if two_d_csv is not None else None),
    }
    if str(mode).lower() == "exact":
        report["exact"] = {
            "layer_index": int(exact_layer_index) if exact_layer_index is not None else None,
            "token_scope": exact_token_scope,
            "u_direction_file": (str(Path(str(exact_u_direction_file)).resolve()) if exact_u_direction_file is not None else None),
            "u_direction_index": int(exact_u_direction_index),
            "v_direction_file": (str(Path(str(exact_v_direction_file)).resolve()) if exact_v_direction_file is not None else None),
            "v_direction_index": int(exact_v_direction_index),
        }
    report_path = exp_dir / "run_report.json"
    _progress(f"Writing run report: {report_path}")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _progress("Landscape analysis finished")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
