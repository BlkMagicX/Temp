"""Run boundary drift evaluation from config.

Usage:
  python scripts/run_boundary_drift_eval.py --config configs/boundary_drift_eval.yaml
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run boundary drift evaluation")
    parser.add_argument("--config", type=str, default="configs/boundary_drift_eval.yaml")
    parser.add_argument("--set", action="append", default=[], help="Override config by key=value")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _ensure_repo_root_on_sys_path()

    from src.experiments.exp_boundary_drift_eval import load_experiment_config, run_experiment_from_config

    cfg = load_experiment_config(args.config)
    for item in args.set:
        if "=" not in item:
            raise ValueError(f"Invalid --set item: {item}. Expected key=value")
        k, raw_v = item.split("=", 1)
        _set_deep(cfg, k.strip(), _parse_literal_value(raw_v))

    result = run_experiment_from_config(cfg)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
