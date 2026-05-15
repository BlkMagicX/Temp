"""Plot 2D comparison curves from QCSD per-sample landscape CSV."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd
import matplotlib.pyplot as plt


def _select_methods(df: pd.DataFrame, methods: Iterable[str] | None) -> list[str]:
    if methods:
        return [m for m in methods if m in set(df["direction_type"].unique())]
    return sorted(df["direction_type"].unique().tolist())


def _prepare_grouped(df: pd.DataFrame, metric: str, methods: list[str]) -> pd.DataFrame:
    if metric not in df.columns:
        raise ValueError(f"metric '{metric}' not found in CSV columns")
    if not methods:
        raise ValueError("No direction_type methods selected")
    df = df[df["direction_type"].isin(methods)].copy()
    return df.groupby(["direction_type", "alpha"], as_index=False)[metric].mean().sort_values(["direction_type", "alpha"])


def plot_curves(csv_path: Path, out_path: Path, metric: str, methods: list[str]) -> None:
    df = pd.read_csv(csv_path)
    grouped = _prepare_grouped(df, metric, methods)

    plt.figure(figsize=(7.2, 4.4))
    for method in methods:
        sub = grouped[grouped["direction_type"] == method]
        if sub.empty:
            continue
        plt.plot(sub["alpha"], sub[metric], marker="o", linewidth=0.6, label=method)

    plt.xlabel("alpha")
    plt.ylabel(metric)
    plt.title(f"QCSD landscape: {metric} vs alpha")
    plt.legend(frameon=False, fontsize=9)
    plt.grid(alpha=0.25, linestyle="--", linewidth=0.6)
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)


def plot_margin_curves(csv_path: Path, out_path: Path, methods: list[str]) -> None:
    df = pd.read_csv(csv_path)
    grouped_fp = _prepare_grouped(df, "margin_fp", methods)
    grouped_q = _prepare_grouped(df, "margin_q", methods)

    plt.figure(figsize=(7.2, 4.4))
    for method in methods:
        sub_fp = grouped_fp[grouped_fp["direction_type"] == method]
        sub_q = grouped_q[grouped_q["direction_type"] == method]
        if not sub_fp.empty:
            plt.plot(
                sub_fp["alpha"],
                sub_fp["margin_fp"],
                marker="o",
                linewidth=1.4,
                label=f"{method} (fp)",
            )
        # if not sub_q.empty:
        #     plt.plot(
        #         sub_q["alpha"],
        #         sub_q["margin_q"],
        #         marker="s",
        #         linewidth=1.4,
        #         label=f"{method} (q)",
        #     )

    plt.xlabel("alpha")
    plt.ylabel("margin")
    plt.title("QCSD landscape: margin_fp and margin_q vs alpha")
    plt.legend(frameon=False, fontsize=9)
    plt.grid(alpha=0.25, linestyle="--", linewidth=0.6)
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot QCSD landscape 2D comparison curves")
    parser.add_argument(
        "--input",
        required=True,
        type=str,
        help="Path to qcsd_landscape.by_sample.csv",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output image path (png). Defaults to <input_dir>/qcsd_landscape_2d.png",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="drop_q",
        help="Metric to plot on y-axis when plot_mode=metric",
    )
    parser.add_argument(
        "--plot-mode",
        type=str,
        default="margins",
        choices=["margins", "metric"],
        help="Plot both margin_fp/margin_q or a single metric",
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="*",
        default=None,
        help="Optional list of direction_type values to plot",
    )

    args = parser.parse_args()
    csv_path = Path(args.input).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    if args.output:
        out_path = Path(args.output).resolve()
    else:
        out_path = csv_path.parent / "qcsd_landscape_2d.png"

    df = pd.read_csv(csv_path)
    methods = _select_methods(df, args.methods)
    if args.plot_mode == "margins":
        plot_margin_curves(csv_path=csv_path, out_path=out_path, methods=methods)
    else:
        plot_curves(csv_path=csv_path, out_path=out_path, metric=args.metric, methods=methods)


if __name__ == "__main__":
    main()
