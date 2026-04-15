"""Writers and summary aggregators for boundary drift evaluation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _avg(values: Iterable[float]) -> float:
    vals: List[float] = []
    for v in values:
        if v is None:
            continue
        try:
            vals.append(float(v))
        except (TypeError, ValueError):
            continue
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))


def _safe_rate(items: List[Dict[str, Any]], key: str) -> float:
    if not items:
        return 0.0
    return float(sum(1 for x in items if bool(x.get(key, False))) / len(items))


class BoundaryDriftResultWriter:
    """Persist per-sample and summary artifacts for boundary drift runs."""

    def __init__(self, output_root: str | Path, exp_name: str) -> None:
        self.output_root = Path(output_root).resolve()
        self.exp_name = exp_name
        self.exp_dir = self.output_root / exp_name
        self.per_sample_dir = self.exp_dir / "per_sample"
        self.summary_dir = self.exp_dir / "summary"
        self.per_sample_dir.mkdir(parents=True, exist_ok=True)
        self.summary_dir.mkdir(parents=True, exist_ok=True)

        self.per_sample_csv = self.per_sample_dir / "boundary_drift_per_sample.csv"
        self.summary_json = self.summary_dir / "boundary_drift_summary.json"

    def write_per_sample_csv(self, rows: List[Dict[str, Any]]) -> Path:
        """Write detailed rows (one sample x one precision per row)."""
        self.per_sample_csv.parent.mkdir(parents=True, exist_ok=True)

        if not rows:
            # keep an empty file with a minimal header for reproducibility
            fieldnames = [
                "sample_id",
                "model_precision",
                "m_fp",
                "m_q",
                "delta_q",
                "flip",
                "boundary_near",
            ]
            with self.per_sample_csv.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
            return self.per_sample_csv

        fieldnames = sorted({k for row in rows for k in row.keys()})
        with self.per_sample_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return self.per_sample_csv

    def build_summary(self, rows: List[Dict[str, Any]], tau: float) -> Dict[str, Any]:
        """Aggregate precision-level summary stats."""
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for row in rows:
            precision = str(row.get("model_precision", "unknown"))
            grouped.setdefault(precision, []).append(row)

        global_boundary_near = [r for r in rows if bool(r.get("boundary_near", False)) and str(r.get("model_precision")) == "fp"]
        boundary_near_count = len(global_boundary_near)

        precision_summary: Dict[str, Any] = {}
        for precision, items in grouped.items():
            if precision == "fp":
                continue

            boundary_items = [x for x in items if bool(x.get("boundary_near", False))]
            precision_summary[precision] = {
                "n_samples": len(items),
                "boundary_near_count": len(boundary_items),
                "flip_rate_all": _safe_rate(items, "flip"),
                "flip_rate_boundary_near": _safe_rate(boundary_items, "flip"),
                "delta_mean": _avg(x.get("delta_q") for x in items),
                "delta_mean_boundary_near": _avg(x.get("delta_q") for x in boundary_items),
                "margin_mean": _avg(x.get("m_q") for x in items),
            }

        return {
            "tau": float(tau),
            "n_rows": len(rows),
            "n_unique_samples": len({str(r.get("sample_id")) for r in rows}),
            "boundary_near_count": boundary_near_count,
            "precision_summary": precision_summary,
        }

    def write_summary_json(self, summary: Dict[str, Any]) -> Path:
        self.summary_json.parent.mkdir(parents=True, exist_ok=True)
        with self.summary_json.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        return self.summary_json
