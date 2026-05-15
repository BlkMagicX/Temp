"""Run representation drift and dangerous-direction extraction.

Usage:
  python scripts/run_representation_drift.py --config configs/representation_drift.yaml
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run representation drift and dangerous-direction extraction")
    parser.add_argument("--config", type=str, default="configs/representation_drift.yaml")
    parser.add_argument("--set", action="append", default=[], help="Override config by key=value")
    return parser.parse_args()


def main() -> None:
    _ensure_repo_root_on_sys_path()

    from src.algorithm.dangerous_direction import (
        DangerousDirectionConfig,
        fit_dangerous_directions,
        save_dangerous_directions,
    )
    from src.algorithm.representation_drift import (
        RepresentationDriftConfig,
        compute_representation_drift_dataset,
        summarize_drift_rows,
    )
    from src.experiments.exp_boundary_drift_eval import load_experiment_config
    from src.models.model_factory import create_vlm

    args = parse_args()
    cfg = load_experiment_config(args.config)
    for item in args.set:
        if "=" not in item:
            raise ValueError(f"Invalid --set item: {item}. Expected key=value")
        k, raw_v = item.split("=", 1)
        _set_deep(cfg, k.strip(), _parse_literal_value(raw_v))

    exp_name = str(cfg.get("exp_name", "representation_drift"))
    out_root = Path(str(dict(cfg.get("output", {})).get("root_dir", "outputs"))).resolve()
    exp_dir = out_root / exp_name / "representation_drift"
    exp_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = dict(cfg.get("data", {}))
    models_cfg = dict(cfg.get("models", {}))
    rep_cfg_raw = dict(cfg.get("representation_drift", {}))
    dir_cfg_raw = dict(cfg.get("dangerous_direction", {}))

    rep_cfg = RepresentationDriftConfig(
        layer_indices=list(rep_cfg_raw.get("layer_indices", [-1])),
        pooling=str(rep_cfg_raw.get("pooling", "mean")),
        max_samples=rep_cfg_raw.get("max_samples"),
    )
    dir_cfg = DangerousDirectionConfig(
        top_k=int(dir_cfg_raw.get("top_k", 3)),
        center=bool(dir_cfg_raw.get("center", True)),
        max_samples_per_layer=dir_cfg_raw.get("max_samples_per_layer"),
    )

    dataset = _build_dataset(data_cfg)
    samples = list(dataset.samples)

    fp_model = create_vlm(_build_fp_model_cfg(models_cfg))
    fp_model.load_model()

    eval_precisions = list(models_cfg.get("eval_precisions", []))
    if not eval_precisions:
        raise ValueError("models.eval_precisions must contain at least one precision")

    run_report: Dict[str, Any] = {
        "exp_name": exp_name,
        "output_dir": str(exp_dir),
        "n_samples_total": len(samples),
        "representation_drift": {},
    }

    try:
        for precision in eval_precisions:
            if str(precision).lower() == "fp":
                continue

            q_model = create_vlm(_build_eval_model_cfg(models_cfg, precision=precision))
            q_model.load_model()

            precision_dir = exp_dir / str(precision)
            precision_dir.mkdir(parents=True, exist_ok=True)

            try:
                result = compute_representation_drift_dataset(
                    samples=samples,
                    fp_model=fp_model,
                    q_model=q_model,
                    cfg=rep_cfg,
                )
                rows = list(result.get("rows", []))
                summary_rows = summarize_drift_rows(rows)

                per_sample_csv = precision_dir / "representation_drift_per_sample.csv"
                if rows:
                    fields = [
                        "sample_id",
                        "layer_index",
                        "fp_norm_l2",
                        "drift_norm_l2",
                        "normalized_drift",
                    ]
                    _write_csv(per_sample_csv, rows=rows, fieldnames=fields)
                else:
                    _write_csv(
                        per_sample_csv,
                        rows=[],
                        fieldnames=["sample_id", "layer_index", "fp_norm_l2", "drift_norm_l2", "normalized_drift"],
                    )

                layer_summary_csv = precision_dir / "representation_drift_layer_summary.csv"
                _write_csv(
                    layer_summary_csv,
                    rows=summary_rows,
                    fieldnames=[
                        "layer_index",
                        "n_samples",
                        "drift_norm_l2_mean",
                        "drift_norm_l2_std",
                        "normalized_drift_mean",
                        "normalized_drift_std",
                        "fp_norm_l2_mean",
                    ],
                )

                directions = fit_dangerous_directions(
                    drifts_by_layer=result.get("drifts_by_layer", {}),
                    cfg=dir_cfg,
                )
                direction_paths = save_dangerous_directions(
                    directions_by_layer=directions,
                    output_dir=precision_dir / "dangerous_directions",
                    prefix="dangerous_directions",
                )

                summary_json = precision_dir / "representation_drift_summary.json"
                summary_payload = {
                    "precision": str(precision),
                    "n_rows": len(rows),
                    "n_layers": len(summary_rows),
                    "layer_summary": summary_rows,
                    "dangerous_direction_layers": sorted([int(k) for k in directions.keys()]),
                    "dangerous_direction_files": [str(p) for p in direction_paths],
                }
                summary_json.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

                run_report["representation_drift"][str(precision)] = {
                    "per_sample_csv": str(per_sample_csv),
                    "layer_summary_csv": str(layer_summary_csv),
                    "summary_json": str(summary_json),
                    "direction_files": [str(p) for p in direction_paths],
                }
            finally:
                _release_model(q_model)
    finally:
        _release_model(fp_model)

    report_path = exp_dir / "run_report.json"
    report_path.write_text(json.dumps(run_report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(run_report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
