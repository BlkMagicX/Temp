"""CLI entrypoint for running qattack experiments.

Usage:
    python src/main.py --config configs/default_experiment.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


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
        "--config",
        type=str,
        required=True,
        help="Path to YAML/JSON experiment config",
    )
    return parser.parse_args()


def main() -> None:
    """Run experiment from config and print output paths."""
    args = parse_args()
    _ensure_repo_root_on_sys_path()

    from src.experiments.exp_transfer_across_precision import run_experiment

    result = run_experiment(args.config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
