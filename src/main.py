"""CLI entrypoint for boundary drift evaluation."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict


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
    parser = argparse.ArgumentParser(description="Run boundary drift evaluation")
    parser.add_argument("--config", type=str, default="configs/boundary_drift_eval.yaml", help="Path to YAML/JSON experiment config")
    parser.add_argument("--set", action="append", default=[], help="Override config using key=value")
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


def _apply_param_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    cfg = dict(config)

    for item in args.set:
        if "=" not in item:
            raise ValueError(f"Invalid --set item: {item}. Expected key=value")
        k, raw_v = item.split("=", 1)
        _set_deep(cfg, k.strip(), _parse_literal_value(raw_v))

    return cfg


def main() -> None:
    """Run experiment from config and print output paths."""
    args = parse_args()
    _ensure_repo_root_on_sys_path()

    from src.experiments.exp_boundary_drift_eval import load_experiment_config, run_experiment_from_config

    config = load_experiment_config(args.config)
    config = _apply_param_overrides(config, args)
    result = run_experiment_from_config(config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
