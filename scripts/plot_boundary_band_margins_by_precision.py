"""Visualize m_q / w_q distributions by precision.

Comparison style:
- FP (baseline) and each quantized precision are overlaid in the same subplot.
- All precision subplots are arranged in a single row.

Usage:
  python scripts/plot_boundary_band_margins_by_precision.py \
    --input_csv outputs/<exp>/per_sample/boundary_drift_per_sample.csv
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt


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


def _read_rows(input_csv: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with input_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []

        fq_col = "w_q" if "w_q" in header else "m_q"

        for r in reader:
            item = dict(r)
            item["model_precision"] = str(item.get("model_precision") or "")
            item["fq_margin"] = _to_float(item.get(fq_col))
            rows.append(item)

    return rows


def _default_output_png(input_csv: Path) -> Path:
    if input_csv.parent.name == "per_sample":
        return input_csv.parent.parent / "summary" / "q_margin_distribution.by_precision.png"
    return input_csv.parent / "q_margin_distribution.by_precision.png"


def _prepare_precision_groups(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        p = str(r.get("model_precision", ""))
        grouped[p].append(r)
    return dict(sorted(grouped.items(), key=lambda x: x[0]))


def _plot(
    grouped: Dict[str, List[Dict[str, Any]]],
    output_png: Path,
    bins: int,
) -> None:
    all_precisions = list(grouped.keys())
    if not all_precisions:
        raise ValueError("No precision rows to plot. Check input CSV.")

    fp_key = next((p for p in all_precisions if p.lower() == "fp"), None)
    fp_vals: List[float] = []
    if fp_key is not None:
        fp_vals = [float(r["fq_margin"]) for r in grouped[fp_key] if r.get("fq_margin") is not None]

    target_precisions = [p for p in all_precisions if p.lower() != "fp"]
    if not target_precisions:
        target_precisions = [fp_key] if fp_key is not None else all_precisions

    ncols = len(target_precisions)
    fig, axes = plt.subplots(nrows=1, ncols=ncols, figsize=(5.2 * ncols, 4.6), squeeze=False)

    for cidx, p_name in enumerate(target_precisions):
        rows = grouped[p_name]
        ax_fq = axes[0, cidx]
        curr_q_vals = [float(r["fq_margin"]) for r in rows if r.get("fq_margin") is not None]
        if fp_vals and p_name.lower() != "fp":
            ax_fq.hist(fp_vals, bins=bins, color="#7F7F7F", alpha=0.45, density=True, label="FP baseline")
        if curr_q_vals:
            ax_fq.hist(curr_q_vals, bins=bins, color="#C44E52", alpha=0.8, density=True, label=p_name)
        ax_fq.set_title(f"{p_name}: m_q / w_q")
        ax_fq.set_xlabel("margin")
        ax_fq.set_ylabel("density")
        if fp_vals and p_name.lower() != "fp":
            ax_fq.legend(loc="upper right", fontsize="small")

    fig.suptitle("Q Margin Distribution by Precision (FP vs Quantized)", fontsize=14, y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot m_q(or w_q) distributions by precision")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to boundary_drift_per_sample.csv")
    parser.add_argument("--output_png", type=str, default=None, help="Output png path")
    parser.add_argument("--bins", type=int, default=200, help="Histogram bins")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_csv = Path(args.input_csv).resolve()
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    output_png = Path(args.output_png).resolve() if args.output_png else _default_output_png(input_csv)

    rows = _read_rows(input_csv)
    grouped = _prepare_precision_groups(rows)
    _plot(
        grouped,
        output_png=output_png,
        bins=int(args.bins),
    )

    print(f"[plot] input={input_csv}")
    print(f"[plot] output={output_png}")
    print("[plot] mode=q_margin_compare_one_row")


if __name__ == "__main__":
    main()
