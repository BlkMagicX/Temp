"""Run lightweight defense boundary calibration evaluation.

Usage:
  python scripts/run_defense_eval.py --config configs/defense_eval.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


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


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run defense evaluation")
    parser.add_argument("--config", type=str, default="configs/defense_eval.yaml")
    parser.add_argument("--set", action="append", default=[], help="Override config by key=value")
    return parser.parse_args()


def main() -> None:
    _ensure_repo_root_on_sys_path()

    from src.algorithm.defense_boundary_calibration import (
        DirectionalBoundaryCalibration,
        DirectionalBoundaryCalibrationConfig,
    )
    from src.eval.defense_evaluator import summarize_defense_rows
    from src.experiments.exp_boundary_drift_eval import load_experiment_config

    args = parse_args()
    cfg = load_experiment_config(args.config)
    for item in args.set:
        if "=" not in item:
            raise ValueError(f"Invalid --set item: {item}. Expected key=value")
        k, raw_v = item.split("=", 1)
        _set_deep(cfg, k.strip(), _parse_literal_value(raw_v))

    exp_name = str(cfg.get("exp_name", "defense_eval"))
    out_root = Path(str(dict(cfg.get("output", {})).get("root_dir", "outputs"))).resolve()
    exp_dir = out_root / exp_name / "defense_eval"
    exp_dir.mkdir(parents=True, exist_ok=True)

    d_cfg = dict(cfg.get("defense_eval", {}))
    rows_csv = d_cfg.get("pressure_rows_csv")
    if not rows_csv:
        raise ValueError("defense_eval.pressure_rows_csv is required")

    calib_cfg = DirectionalBoundaryCalibrationConfig(
        mode=str(d_cfg.get("mode", "approximate")),
        lambda_proj=float(d_cfg.get("lambda_proj", 0.0)),
        lambda_delta=float(d_cfg.get("lambda_delta", 1.0)),
        lambda_bias=float(d_cfg.get("lambda_bias", 0.0)),
        risk_margin_threshold=float(d_cfg.get("risk_margin_threshold", 1.0)),
        only_high_risk=bool(d_cfg.get("only_high_risk", True)),
    )
    calibrator = DirectionalBoundaryCalibration(calib_cfg)

    raw_rows: List[Dict[str, Any]] = []
    with Path(str(rows_csv)).resolve().open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            raw_rows.append(dict(r))

    fp_map: Dict[Tuple[str, str, float], float] = {}
    for r in raw_rows:
        precision = str(r.get("precision", ""))
        if precision.lower() != "fp":
            continue
        key = (
            str(r.get("sample_id", "")),
            str(r.get("attack_family", "")),
            float(_to_float(r.get("strength")) or 0.0),
        )
        fp_map[key] = float(_to_float(r.get("margin")) or 0.0)

    eval_rows: List[Dict[str, Any]] = []
    for r in raw_rows:
        sid = str(r.get("sample_id", ""))
        family = str(r.get("attack_family", ""))
        strength = float(_to_float(r.get("strength")) or 0.0)
        precision = str(r.get("precision", "unknown"))
        margin = float(_to_float(r.get("margin")) or 0.0)
        is_boundary_near = _to_bool(r.get("is_boundary_near"))

        base = {
            "sample_id": sid,
            "attack_family": family,
            "precision": precision,
            "strength": strength,
            "margin": margin,
            "is_boundary_near": is_boundary_near,
        }

        if precision.lower() == "fp":
            base["margin_defended"] = margin
            base["margin_restoration"] = 0.0
            eval_rows.append(base)
            continue

        fp_margin = fp_map.get((sid, family, strength), margin)
        delta_q = float(margin - fp_margin)
        dangerous_score = _to_float(r.get("dangerous_score"))

        calibrated = calibrator.calibrate_margin(
            m_q=margin,
            delta_q=delta_q,
            dangerous_score=dangerous_score,
        )
        base["margin_defended"] = float(calibrated)
        base["margin_restoration"] = float(calibrated - margin)
        eval_rows.append(base)

    defended_rows_csv = exp_dir / "defense_eval_rows.csv"
    _write_csv(
        defended_rows_csv,
        eval_rows,
        [
            "sample_id",
            "attack_family",
            "precision",
            "strength",
            "margin",
            "margin_defended",
            "margin_restoration",
            "is_boundary_near",
        ],
    )

    summary = summarize_defense_rows(
        rows=eval_rows,
        target_asr=float(d_cfg.get("target_asr", 0.5)),
    )
    summary_path = exp_dir / "defense_eval_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    report = {
        "exp_name": exp_name,
        "output_dir": str(exp_dir),
        "input_rows_csv": str(Path(str(rows_csv)).resolve()),
        "defended_rows_csv": str(defended_rows_csv),
        "summary_json": str(summary_path),
        "n_rows": len(eval_rows),
    }
    report_path = exp_dir / "run_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
