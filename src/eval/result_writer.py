"""Utilities for per-sample result persistence and summary export."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

from .metrics import average_metric


def _get_metric(record: Dict[str, Any], key: str) -> Any:
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


class ResultWriter:
    """Write per-sample records and aggregate summaries.

    Output structure:
        outputs/<exp_name>/
          - per_sample/
          - summary/
    """

    def __init__(self, base_output_dir: str | Path, exp_name: str) -> None:
        self.base_output_dir = Path(base_output_dir).resolve()
        self.exp_name = exp_name
        self.exp_root = self.base_output_dir / exp_name
        self.per_sample_dir = self.exp_root / "per_sample"
        self.summary_dir = self.exp_root / "summary"

        self.per_sample_dir.mkdir(parents=True, exist_ok=True)
        self.summary_dir.mkdir(parents=True, exist_ok=True)

        self.per_sample_jsonl_path = self.per_sample_dir / "results.jsonl"
        self.summary_csv_path = self.summary_dir / "summary.csv"

    def append_per_sample_record(self, record: Dict[str, Any]) -> None:
        """Append one sample-level evaluation record as JSONL."""
        with self.per_sample_jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def load_per_sample_records(self) -> List[Dict[str, Any]]:
        """Load all existing per-sample records from JSONL."""
        if not self.per_sample_jsonl_path.exists():
            return []

        records: List[Dict[str, Any]] = []
        with self.per_sample_jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records

    def load_existing_keys(self, key_fields: Sequence[str]) -> Set[Tuple[Any, ...]]:
        """Load existing unique keys for resume.

        Args:
            key_fields: Field names to compose one record key.
        """
        keys: Set[Tuple[Any, ...]] = set()
        for record in self.load_per_sample_records():
            keys.add(tuple(record.get(k) for k in key_fields))
        return keys

    def write_summary_csv(self, records: Iterable[Dict[str, Any]] | None = None) -> Path:
        """Aggregate per-sample records by precision and write summary CSV."""
        if records is None:
            records = self.load_per_sample_records()

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
                    "attack_success_rate": average_metric(_get_metric(r, "attack_success") for r in items),
                    "caption_attack_success_rate": average_metric(_get_metric(r, "caption_attack_success") for r in items),
                    "avg_caption_similarity": average_metric(_get_metric(r, "caption_similarity") for r in items),
                    "refusal_rate": average_metric(_get_metric(r, "refusal_flag") for r in items),
                    "avg_delta_linf": average_metric(r.get("delta_linf") for r in items),
                    "avg_delta_l2": average_metric(r.get("delta_l2") for r in items),
                    "avg_runtime": average_metric(r.get("runtime") for r in items),
                }
            )

        rows = sorted(rows, key=lambda x: x["model_precision"])

        fieldnames = [
            "model_precision",
            "n_samples",
            "attack_success_rate",
            "caption_attack_success_rate",
            "avg_caption_similarity",
            "refusal_rate",
            "avg_delta_linf",
            "avg_delta_l2",
            "avg_runtime",
        ]

        with self.summary_csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        return self.summary_csv_path
