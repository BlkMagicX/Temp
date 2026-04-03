"""Build summary.csv directly from per_sample/results.jsonl.

Usage:
  python scripts/build_summary_from_results.py --results outputs/<exp_name>/per_sample/results.jsonl
  python scripts/build_summary_from_results.py --exp-dir outputs/<exp_name>
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def average_metric(values: Iterable[Any]) -> float:
    cleaned: List[float] = []
    for v in values:
        if v is None:
            continue
        if isinstance(v, bool):
            cleaned.append(1.0 if v else 0.0)
            continue
        try:
            cleaned.append(float(v))
        except (TypeError, ValueError):
            continue
    if not cleaned:
        return 0.0
    return sum(cleaned) / len(cleaned)


def get_metric(record: Dict[str, Any], key: str) -> Any:
    """Read metric from top-level first, then fallback to nested judge dict."""
    value = record.get(key)
    if value is None:
        judge = record.get("judge")
        if isinstance(judge, dict):
            value = judge.get(key)

    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "false"}:
            return lowered == "true"
    return value


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"results.jsonl not found: {path}")

    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def build_summary(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        precision = str(r.get("model_precision", "unknown"))
        grouped.setdefault(precision, []).append(r)

    rows: List[Dict[str, Any]] = []
    for precision, items in grouped.items():
        rows.append(
            {
                "model_precision": precision,
                "n_samples": len(items),
                "attack_success_rate": average_metric(get_metric(r, "attack_success") for r in items),
                "caption_attack_success_rate": average_metric(get_metric(r, "caption_attack_success") for r in items),
                "refusal_rate": average_metric(get_metric(r, "refusal_flag") for r in items),
                "avg_delta_linf": average_metric(r.get("delta_linf") for r in items),
                "avg_delta_l2": average_metric(r.get("delta_l2") for r in items),
                "avg_runtime": average_metric(r.get("runtime") for r in items),
            }
        )

    return sorted(rows, key=lambda x: x["model_precision"])


def write_summary_csv(rows: List[Dict[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model_precision",
        "n_samples",
        "attack_success_rate",
        "caption_attack_success_rate",
        "refusal_rate",
        "avg_delta_linf",
        "avg_delta_l2",
        "avg_runtime",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build summary.csv from results.jsonl")
    parser.add_argument("--results", type=str, default=None, help="Path to per_sample/results.jsonl")
    parser.add_argument("--exp-dir", type=str, default=None, help="Path to outputs/<exp_name>")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if bool(args.results) == bool(args.exp_dir):
        raise ValueError("Provide exactly one of --results or --exp-dir")

    if args.results:
        results_path = Path(args.results).resolve()
        exp_dir = results_path.parent.parent
    else:
        exp_dir = Path(args.exp_dir).resolve()
        results_path = exp_dir / "per_sample" / "results.jsonl"

    summary_path = exp_dir / "summary" / "summary.csv"

    records = load_jsonl(results_path)
    rows = build_summary(records)
    out = write_summary_csv(rows, summary_path)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
