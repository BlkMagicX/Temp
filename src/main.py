"""CLI entrypoint for running qattack experiments.

Usage:
    python src/main.py --config configs/default_experiment.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def _ensure_repo_root_on_sys_path() -> Path:
    """Ensure repository root is importable as package root.

    This is needed when running `python src/main.py`, where `sys.path[0]`
    points to `src/` rather than repository root.
    """
    current_file = Path(__file__).resolve()
    repo_root = current_file.parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run qattack experiment")
    parser.add_argument(
        "--mode",
        type=str,
        default="transfer",
        choices=["transfer", "boundary_drift"],
        help="Experiment mode: transfer (default) or boundary_drift",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        default="configs/exp_mpcattack-eps-16_awq_qwen2-vl-2b.yaml",
        help="Path to YAML/JSON experiment config",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help='Override config using key=value, e.g. models.eval_precisions=["w4a16"]',
    )
    parser.add_argument("--eps", type=int, default=None, help="Shortcut for eps dataset selection")
    parser.add_argument("--pairs-file", type=str, default=None, help="Override runtime.direct_image_test.pairs_file")
    parser.add_argument("--annotation-path", type=str, default=None, help="Override data.annotation_path")
    parser.add_argument("--model-path", type=str, default=None, help="Override models.eval_model_template.model_path")
    parser.add_argument("--quant-model-path", type=str, default=None, help="Override quant_model_path for selected precision")
    parser.add_argument("--model-name", type=str, default=None, help="Override models.eval_model_template.name")
    parser.add_argument("--backend", type=str, default=None, help="Override backend_type for selected precision")
    parser.add_argument("--precision", type=str, default=None, help="Select one precision and write to eval_precisions")
    parser.add_argument("--output-root", type=str, default=None, help="Override output.root_dir")
    parser.add_argument("--exp-name", type=str, default=None, help="Override exp_name directly")
    parser.add_argument(
        "--auto-exp-name",
        action="store_true",
        help="Auto-generate concise exp_name from params (eps/model/backend/precision)",
    )
    return parser.parse_args()


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
        try:
            return json.loads(value)
        except Exception:
            return value

    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
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


def _sanitize_name(s: str) -> str:
    allowed = []
    for ch in s:
        if ch.isalnum() or ch in {"-", "_"}:
            allowed.append(ch)
        else:
            allowed.append("-")
    return "".join(allowed).strip("-")


def _resolve_eps_paths(repo_root: Path, eps: int) -> Dict[str, str]:
    pairs_candidates = [
        repo_root / "examples" / "mpcattack" / f"eps-{eps}" / "pairs.jsonl",
        repo_root / "examples" / f"mpcattack-eps-{eps}_pairs.jsonl",
    ]
    ann_candidates = [
        repo_root / "examples" / "mpcattack" / f"eps-{eps}" / "samples.json",
        repo_root / "examples" / f"mpcattack-eps-{eps}_samples.json",
    ]

    resolved: Dict[str, str] = {}
    for p in pairs_candidates:
        if p.exists():
            resolved["pairs_file"] = str(p.relative_to(repo_root).as_posix())
            break
    for p in ann_candidates:
        if p.exists():
            resolved["annotation_path"] = str(p.relative_to(repo_root).as_posix())
            break
    return resolved


def _apply_param_overrides(config: Dict[str, Any], args: argparse.Namespace, repo_root: Path) -> Dict[str, Any]:
    cfg = dict(config)

    for item in args.set:
        if "=" not in item:
            raise ValueError(f"Invalid --set item: {item}. Expected key=value")
        k, raw_v = item.split("=", 1)
        _set_deep(cfg, k.strip(), _parse_literal_value(raw_v))

    precision = args.precision
    if precision:
        _set_deep(cfg, "models.eval_precisions", [precision])

    if args.eps is not None:
        eps_paths = _resolve_eps_paths(repo_root, args.eps)
        if "pairs_file" in eps_paths:
            _set_deep(cfg, "runtime.direct_image_test.pairs_file", eps_paths["pairs_file"])
        if "annotation_path" in eps_paths:
            _set_deep(cfg, "data.annotation_path", eps_paths["annotation_path"])

    if args.pairs_file:
        _set_deep(cfg, "runtime.direct_image_test.pairs_file", args.pairs_file)
    if args.annotation_path:
        _set_deep(cfg, "data.annotation_path", args.annotation_path)
    if args.model_path:
        _set_deep(cfg, "models.eval_model_template.model_path", args.model_path)
    if args.model_name:
        _set_deep(cfg, "models.eval_model_template.name", args.model_name)
    if args.output_root:
        _set_deep(cfg, "output.root_dir", args.output_root)

    if precision and args.backend:
        _set_deep(cfg, f"models.precision_overrides.{precision}.backend_type", args.backend)
    if precision and args.quant_model_path:
        _set_deep(cfg, f"models.precision_overrides.{precision}.quant_model_path", args.quant_model_path)

    if args.exp_name:
        cfg["exp_name"] = args.exp_name
    elif args.auto_exp_name or args.eps is not None:
        model_name = str(args.model_name or cfg.get("models", {}).get("eval_model_template", {}).get("name", "model"))
        backend = str(args.backend or "auto")
        use_precision = precision or "multi"
        eps_tag = f"eps{args.eps}" if args.eps is not None else "epsNA"
        cfg["exp_name"] = _sanitize_name(f"{eps_tag}_{model_name}_{backend}_{use_precision}")

    return cfg


def main() -> None:
    """Run experiment from config and print output paths."""
    args = parse_args()
    repo_root = _ensure_repo_root_on_sys_path()

    if args.mode == "boundary_drift":
        from src.experiments.exp_boundary_drift_eval import load_experiment_config, run_experiment_from_config
    else:
        from src.experiments.exp_transfer_across_precision import load_experiment_config, run_experiment_from_config

    config = load_experiment_config(args.config)
    config = _apply_param_overrides(config, args, repo_root=repo_root)
    result = run_experiment_from_config(config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
