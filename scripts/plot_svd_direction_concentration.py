"""Analyze and visualize SVD concentration from dangerous direction files.

Usage:
  python scripts/plot_svd_direction_concentration.py \
    --direction-dir outputs/representation_drift_mm_safetybench/representation_drift/w3a16/dangerous_directions
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot SVD concentration from dangerous_directions.layer_x.pt")
    parser.add_argument("--direction-dir", type=str, required=True, help="Directory containing dangerous_directions.layer_*.pt")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. Defaults to <direction_dir>/../svd_analysis",
    )
    parser.add_argument("--precision", type=str, default=None, help="Optional precision label for plot titles")
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def _load_layer_file(path: Path) -> Dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    evr = payload["explained_variance_ratio"]
    if hasattr(evr, "detach"):
        evr = evr.detach().cpu().numpy()
    evr = np.asarray(evr, dtype=np.float32).reshape(-1)

    return {
        "layer_index": int(payload["layer_index"]),
        "n_samples": int(payload.get("n_samples", 0)),
        "evr": evr,
    }


def _rank_desc(values: np.ndarray) -> np.ndarray:
    order = np.argsort(-values)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(values) + 1)
    return ranks


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    direction_dir = Path(args.direction_dir).resolve()
    if not direction_dir.exists() or not direction_dir.is_dir():
        raise ValueError(f"Invalid --direction-dir: {direction_dir}")

    files = sorted(direction_dir.glob("dangerous_directions.layer_*.pt"), key=lambda p: int(p.stem.split("layer_")[-1]))
    if not files:
        raise ValueError(f"No dangerous_directions.layer_*.pt found under {direction_dir}")

    output_dir = Path(args.output_dir).resolve() if args.output_dir else direction_dir.parent / "svd_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    loaded = [_load_layer_file(p) for p in files]
    loaded.sort(key=lambda x: x["layer_index"])

    layers = np.asarray([x["layer_index"] for x in loaded], dtype=np.int32)
    n_samples = np.asarray([x["n_samples"] for x in loaded], dtype=np.int32)

    evr1 = np.asarray([float(x["evr"][0]) if len(x["evr"]) >= 1 else np.nan for x in loaded], dtype=np.float32)
    evr2 = np.asarray([float(x["evr"][1]) if len(x["evr"]) >= 2 else np.nan for x in loaded], dtype=np.float32)
    evr12 = np.asarray(
        [float(np.sum(x["evr"][:2])) if len(x["evr"]) >= 2 else float(x["evr"][0]) for x in loaded],
        dtype=np.float32,
    )
    evr_topk = np.asarray([float(np.sum(x["evr"])) for x in loaded], dtype=np.float32)

    rank_evr1 = _rank_desc(evr1)
    rank_evr12 = _rank_desc(evr12)

    rows: List[Dict[str, Any]] = []
    for i in range(len(layers)):
        rows.append(
            {
                "layer_index": int(layers[i]),
                "n_samples": int(n_samples[i]),
                "evr1": float(evr1[i]),
                "evr2": float(evr2[i]) if not np.isnan(evr2[i]) else None,
                "evr1_plus_evr2": float(evr12[i]),
                "evr_topk_sum": float(evr_topk[i]),
                "rank_evr1_desc": int(rank_evr1[i]),
                "rank_evr12_desc": int(rank_evr12[i]),
            }
        )

    csv_path = output_dir / "svd_concentration_by_layer.csv"
    _write_csv(
        csv_path,
        rows=rows,
        fieldnames=[
            "layer_index",
            "n_samples",
            "evr1",
            "evr2",
            "evr1_plus_evr2",
            "evr_topk_sum",
            "rank_evr1_desc",
            "rank_evr12_desc",
        ],
    )

    last_layer = int(np.max(layers))
    last_idx = int(np.where(layers == last_layer)[0][0])

    summary = {
        "n_layers": int(len(layers)),
        "last_layer": last_layer,
        "last_layer_evr1": float(evr1[last_idx]),
        "last_layer_evr1_plus_evr2": float(evr12[last_idx]),
        "last_layer_rank_evr1_desc": int(rank_evr1[last_idx]),
        "last_layer_rank_evr12_desc": int(rank_evr12[last_idx]),
        "evr1_mean": float(np.mean(evr1)),
        "evr1_min": float(np.min(evr1)),
        "evr1_max": float(np.max(evr1)),
        "evr12_mean": float(np.mean(evr12)),
        "evr12_min": float(np.min(evr12)),
        "evr12_max": float(np.max(evr12)),
        "is_last_layer_most_concentrated_evr1": bool(rank_evr1[last_idx] == 1),
        "is_last_layer_most_concentrated_evr12": bool(rank_evr12[last_idx] == 1),
        "source_direction_dir": str(direction_dir),
        "rows_csv": str(csv_path),
    }

    summary_path = output_dir / "svd_concentration_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # Plot 1: concentration curves
    fig1, ax1 = plt.subplots(figsize=(11, 6))
    ax1.plot(layers, evr1, marker="o", linewidth=2.0, label="EVR1")
    ax1.plot(layers, evr12, marker="s", linewidth=2.0, label="EVR1+EVR2")
    ax1.plot(layers, evr_topk, marker="^", linewidth=1.8, label="EVR(top-k sum)")
    ax1.axvline(x=last_layer, linestyle="--", linewidth=1.2, alpha=0.6, color="gray", label="last layer")
    ax1.set_xlabel("layer_index")
    ax1.set_ylabel("explained variance ratio")
    title_precision = f" [{args.precision}]" if args.precision else ""
    ax1.set_title(f"SVD concentration by layer{title_precision}")
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend()
    fig1.tight_layout()
    curve_path = output_dir / "svd_concentration_curves.png"
    fig1.savefig(curve_path, dpi=int(args.dpi))

    # Plot 2: per-component EVR heatmap (across available top-k components)
    k_max = max(len(x["evr"]) for x in loaded)
    heat = np.full((len(layers), k_max), np.nan, dtype=np.float32)
    for i, x in enumerate(loaded):
        k = len(x["evr"])
        heat[i, :k] = x["evr"]

    fig2, ax2 = plt.subplots(figsize=(11, 6))
    im = ax2.imshow(heat, aspect="auto", interpolation="nearest", cmap="viridis")
    ax2.set_xlabel("component index")
    ax2.set_ylabel("layer_index")
    ax2.set_title(f"SVD EVR heatmap by layer{title_precision}")
    ax2.set_xticks(range(k_max))
    ax2.set_xticklabels([str(i + 1) for i in range(k_max)])
    ax2.set_yticks(range(len(layers)))
    ax2.set_yticklabels([str(int(x)) for x in layers])
    cbar = fig2.colorbar(im, ax=ax2)
    cbar.set_label("explained variance ratio")
    fig2.tight_layout()
    heatmap_path = output_dir / "svd_concentration_heatmap.png"
    fig2.savefig(heatmap_path, dpi=int(args.dpi))

    report = {
        **summary,
        "curve_png": str(curve_path),
        "heatmap_png": str(heatmap_path),
        "summary_json": str(summary_path),
    }
    report_path = output_dir / "run_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
