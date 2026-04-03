"""Plot caption_attack_success_rate vs eps for different precisions.

Scans outputs/*/summary/summary.csv, extracts eps from folder names
like exp_mpcattack-eps-16_transfer_across_precision_qwen2-vl-7b.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def _find_summary_csvs(outputs_dir: Path) -> List[Path]:
    return list(outputs_dir.glob("**/summary/summary.csv"))


def _extract_eps(path: Path) -> float | None:
    match = re.search(r"eps-([0-9]+(?:\.[0-9]+)?)", str(path))
    if not match:
        return None
    return float(match.group(1))


def _load_rates(csv_path: Path) -> Dict[str, float]:
    rates: Dict[str, float] = {}
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            precision = str(row.get("model_precision", "unknown"))
            value = row.get("caption_attack_success_rate")
            if value is None or value == "":
                continue
            rates[precision] = float(value)
    return rates


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    outputs_dir = repo_root / "outputs"
    if not outputs_dir.exists():
        raise FileNotFoundError(f"outputs dir not found: {outputs_dir}")

    series: Dict[str, List[Tuple[float, float]]] = {}
    for csv_path in _find_summary_csvs(outputs_dir):
        eps = _extract_eps(csv_path)
        if eps is None:
            continue
        rates = _load_rates(csv_path)
        for precision, rate in rates.items():
            series.setdefault(precision, []).append((eps, rate))

    if not series:
        raise RuntimeError("No summary.csv with eps-<n> found under outputs/")

    plt.figure(figsize=(8, 5))
    for precision, points in series.items():
        points = sorted(points, key=lambda x: x[0])
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        plt.plot(xs, ys, marker="o", label=precision)

    plt.xlabel("eps")
    plt.ylabel("caption_attack_success_rate")
    plt.title("Caption Attack Success Rate vs eps")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()

    out_path = outputs_dir / "caption_attack_success_vs_eps.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
