"""Plot layer-wise representation drift statistics from summary JSON.

Usage:
  python scripts/plot_representation_drift_layers.py \
    --input outputs/representation_drift_mm_safetybench/representation_drift/w3a16/representation_drift_summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot layer-wise representation drift curves")
    parser.add_argument("--input", type=str, required=True, help="Path to representation_drift_summary.json")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output image path. Defaults to <input_dir>/representation_drift_layer_plot.png",
    )
    parser.add_argument("--dpi", type=int, default=160, help="Output figure DPI")
    return parser.parse_args()


def _load_summary(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_series(layer_summary: List[Dict[str, Any]], key: str) -> List[float]:
    values: List[float] = []
    for row in layer_summary:
        if key not in row:
            raise KeyError(f"Missing key in layer_summary row: {key}")
        values.append(float(row[key]))
    return values


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    data = _load_summary(input_path)

    layer_summary = data.get("layer_summary", [])
    if not layer_summary:
        raise ValueError("layer_summary is empty in the input JSON")

    layers = [int(row["layer_index"]) for row in layer_summary]
    drift_mean = _extract_series(layer_summary, "drift_norm_l2_mean")
    drift_std = _extract_series(layer_summary, "drift_norm_l2_std")
    norm_mean = _extract_series(layer_summary, "normalized_drift_mean")
    norm_std = _extract_series(layer_summary, "normalized_drift_std")

    output_path = Path(args.output).resolve() if args.output else input_path.parent / "representation_drift_layer_plot.png"

    fig, ax = plt.subplots(figsize=(11, 6))
    # ax.plot(layers, drift_mean, marker="o", linewidth=1.8, label="drift_norm_l2_mean")
    # ax.plot(layers, drift_std, marker="o", linewidth=1.6, label="drift_norm_l2_std")
    ax.plot(layers, norm_mean, marker="s", linewidth=1.8, label="normalized_drift_mean")
    ax.plot(layers, norm_std, marker="s", linewidth=1.6, label="normalized_drift_std")

    ax.set_xlabel("layer_index")
    ax.set_ylabel("value")
    ax.set_title("Layer-wise Representation Drift Statistics")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=int(args.dpi))
    print(f"Saved plot to: {output_path}")


if __name__ == "__main__":
    main()
