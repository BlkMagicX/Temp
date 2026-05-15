"""Plot landscape figures from landscape_scan_1d.csv or landscape_scan_2d.csv.

Auto-detects input dimension by inspecting CSV columns:
- 1D input (alpha + margin/margin_fp)        -> 2D curve plot
- 2D input (alpha + beta + margin/margin_fp) -> 3D surface plot

Usage:
  python scripts/plot_landscape_3d.py \\
    --input_csv outputs/landscape_mm_safetybench/landscape/landscape_scan_2d.csv
  python scripts/plot_landscape_3d.py \\
    --input_csv outputs/landscape_mm_safetybench/landscape/landscape_scan_1d.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


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


def _detect_csv_schema(input_csv: Path) -> Tuple[bool, str]:
    """Detect whether CSV is 2D (has 'beta') and pick value column.

    Returns:
        (has_beta, value_key)
    """
    with input_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
    has_beta = "beta" in fieldnames
    value_key = "margin_fp" if "margin_fp" in fieldnames else "margin"
    if value_key not in fieldnames:
        raise ValueError(f"CSV must contain 'margin' or 'margin_fp' column: {input_csv}")
    return has_beta, value_key


def _read_rows(input_csv: Path, has_beta: bool, value_key: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with input_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            item = dict(r)
            item["sample_id"] = str(item.get("sample_id") or "")
            item["precision"] = str(item.get("precision") or "")
            item["mode"] = str(item.get("mode") or "")
            item["alpha"] = _to_float(item.get("alpha"))
            item["beta"] = _to_float(item.get("beta")) if has_beta else None
            item["value"] = _to_float(item.get(value_key))
            if item["alpha"] is None or item["value"] is None:
                continue
            if has_beta and item["beta"] is None:
                continue
            rows.append(item)
    return rows


def _default_output_png(input_csv: Path, has_beta: bool) -> Path:
    suffix = "surface_3d" if has_beta else "curve_2d"
    return input_csv.parent / f"landscape_{suffix}.png"


def _aggregate_rows(rows: List[Dict[str, Any]], method: str, has_beta: bool) -> List[Dict[str, Any]]:
    if has_beta:
        grouped_2d: Dict[Tuple[float, float], List[float]] = defaultdict(list)
        for r in rows:
            grouped_2d[(float(r["alpha"]), float(r["beta"]))].append(float(r["value"]))
        out_2d: List[Dict[str, Any]] = []
        for (a, b), vals in grouped_2d.items():
            arr = np.asarray(vals, dtype=np.float64)
            m = float(np.median(arr)) if method == "median" else float(np.mean(arr))
            out_2d.append({"alpha": a, "beta": b, "value": m})
        return out_2d

    grouped_1d: Dict[float, List[float]] = defaultdict(list)
    for r in rows:
        grouped_1d[float(r["alpha"])].append(float(r["value"]))
    out_1d: List[Dict[str, Any]] = []
    for a, vals in grouped_1d.items():
        arr = np.asarray(vals, dtype=np.float64)
        m = float(np.median(arr)) if method == "median" else float(np.mean(arr))
        out_1d.append({"alpha": a, "value": m})
    return out_1d


def _build_dense_grid(rows: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    alphas = sorted({float(r["alpha"]) for r in rows})
    betas = sorted({float(r["beta"]) for r in rows})

    a_to_idx = {a: i for i, a in enumerate(alphas)}
    b_to_idx = {b: i for i, b in enumerate(betas)}

    z = np.full((len(betas), len(alphas)), np.nan, dtype=np.float64)
    cnt = np.zeros((len(betas), len(alphas)), dtype=np.int32)

    for r in rows:
        a = float(r["alpha"])
        b = float(r["beta"])
        m = float(r["value"])
        bi = b_to_idx[b]
        ai = a_to_idx[a]
        if np.isnan(z[bi, ai]):
            z[bi, ai] = m
            cnt[bi, ai] = 1
        else:
            z[bi, ai] += m
            cnt[bi, ai] += 1

    mask = cnt > 0
    z[mask] = z[mask] / cnt[mask]

    aa, bb = np.meshgrid(np.asarray(alphas, dtype=np.float64), np.asarray(betas, dtype=np.float64))
    complete = bool(np.all(~np.isnan(z)))
    return aa, bb, z, complete


def _gaussian_kernel_1d(sigma: float, radius: Optional[int] = None) -> np.ndarray:
    if sigma <= 0.0:
        return np.asarray([1.0], dtype=np.float64)
    r = int(radius) if radius is not None else max(1, int(math.ceil(3.0 * float(sigma))))
    xs = np.arange(-r, r + 1, dtype=np.float64)
    k = np.exp(-(xs * xs) / (2.0 * float(sigma) * float(sigma)))
    k_sum = float(np.sum(k))
    if k_sum <= 0.0:
        return np.asarray([1.0], dtype=np.float64)
    return k / k_sum


def _smooth_vector_nan(vec: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    pad = len(kernel) // 2
    if len(vec) == 0:
        return vec

    val = np.nan_to_num(vec.astype(np.float64), nan=0.0)
    w = np.isfinite(vec).astype(np.float64)

    val_pad = np.pad(val, (pad, pad), mode="edge")
    w_pad = np.pad(w, (pad, pad), mode="edge")

    val_conv = np.convolve(val_pad, kernel, mode="valid")
    w_conv = np.convolve(w_pad, kernel, mode="valid")

    out = np.full_like(val_conv, np.nan, dtype=np.float64)
    valid = w_conv > 1e-12
    out[valid] = val_conv[valid] / w_conv[valid]
    return out


def _gaussian_smooth_grid_nan(z: np.ndarray, sigma: float, radius: Optional[int] = None) -> np.ndarray:
    if sigma <= 0.0:
        return np.asarray(z, dtype=np.float64).copy()
    kernel = _gaussian_kernel_1d(sigma=sigma, radius=radius)
    tmp = np.apply_along_axis(lambda v: _smooth_vector_nan(v, kernel), axis=1, arr=z)
    out = np.apply_along_axis(lambda v: _smooth_vector_nan(v, kernel), axis=0, arr=tmp)
    return out


def _plot_surface_or_trisurf(
    ax: Any,
    aa: np.ndarray,
    bb: np.ndarray,
    zz: np.ndarray,
    complete: bool,
    vmin: float,
    vmax: float,
) -> Any:
    if complete:
        return ax.plot_surface(
            aa,
            bb,
            zz,
            cmap="viridis",
            linewidth=0.2,
            edgecolor="k",
            antialiased=True,
            alpha=0.95,
            vmin=vmin,
            vmax=vmax,
        )

    mask = np.isfinite(zz)
    x = aa[mask]
    y = bb[mask]
    z = zz[mask]
    if x.size < 3:
        raise ValueError("Not enough finite grid points to draw 3D surface")
    return ax.plot_trisurf(
        x,
        y,
        z,
        cmap="viridis",
        linewidth=0,
        edgecolor="k",
        antialiased=True,
        alpha=0.95,
        vmin=vmin,
        vmax=vmax,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot landscape: 2D curve from 1D CSV, 3D surface from 2D CSV (auto-detected)"
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Path to landscape_scan_1d.csv or landscape_scan_2d.csv",
    )
    parser.add_argument("--output_png", type=str, default=None, help="Output PNG path")
    parser.add_argument("--precision", type=str, default=None, help="Filter by precision")
    parser.add_argument("--sample_id", type=str, default=None, help="Filter by sample_id")
    parser.add_argument("--mode", type=str, default=None, help="Filter by mode")
    parser.add_argument(
        "--aggregate",
        type=str,
        default="none",
        choices=["none", "mean", "median"],
        help="Aggregate across sample_ids at each alpha (and beta if 2D)",
    )
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--elev", type=float, default=28.0, help="3D view elevation (only for 2D input)")
    parser.add_argument("--azim", type=float, default=-135.0, help="3D view azimuth (only for 2D input)")
    parser.add_argument(
        "--smooth_sigma",
        type=float,
        default=1.2,
        help="Gaussian smoothing sigma (<=0 disables smoothing panel)",
    )
    parser.add_argument(
        "--smooth_radius",
        type=int,
        default=None,
        help="Optional Gaussian kernel radius; default is ceil(3*sigma)",
    )
    parser.add_argument(
        "--no_compare_smoothed",
        action="store_true",
        help="Only plot raw without smoothed comparison",
    )
    return parser.parse_args()


def _plot_3d_surface(
    plot_rows: List[Dict[str, Any]],
    value_key: str,
    title_suffix: List[str],
    args: argparse.Namespace,
    output_png: Path,
) -> bool:
    if len(plot_rows) < 3:
        raise ValueError("Need at least 3 points to draw 3D surface")

    aa, bb, zz, complete = _build_dense_grid(plot_rows)

    use_compare_smoothed = (not args.no_compare_smoothed) and float(args.smooth_sigma) > 0.0
    zz_smooth = None
    if use_compare_smoothed:
        zz_smooth = _gaussian_smooth_grid_nan(zz, sigma=float(args.smooth_sigma), radius=args.smooth_radius)

    if zz_smooth is None:
        z_pool = zz[np.isfinite(zz)]
    else:
        z_pool = np.concatenate([zz[np.isfinite(zz)], zz_smooth[np.isfinite(zz_smooth)]], axis=0)
    if z_pool.size == 0:
        raise ValueError(f"No finite {value_key} values to plot")
    vmin = float(np.min(z_pool))
    vmax = float(np.max(z_pool))

    if use_compare_smoothed:
        fig = plt.figure(figsize=(16.2, 7.0))
        ax_raw = fig.add_subplot(1, 2, 1, projection="3d")
        ax_smooth = fig.add_subplot(1, 2, 2, projection="3d")

        surf_raw = _plot_surface_or_trisurf(ax_raw, aa=aa, bb=bb, zz=zz, complete=complete, vmin=vmin, vmax=vmax)
        surf_smooth = _plot_surface_or_trisurf(
            ax_smooth,
            aa=aa,
            bb=bb,
            zz=zz_smooth,
            complete=bool(np.all(np.isfinite(zz_smooth))),
            vmin=vmin,
            vmax=vmax,
        )

        base_title = f"3D Landscape (alpha, beta, {value_key})"
        if title_suffix:
            base_title += " | " + ", ".join(title_suffix)
        ax_raw.set_title(base_title + " | raw")
        ax_smooth.set_title(base_title + f" | gaussian smooth sigma={float(args.smooth_sigma):.2f}")

        for ax in (ax_raw, ax_smooth):
            ax.set_xlabel("alpha")
            ax.set_ylabel("beta")
            ax.set_zlabel(value_key)
            ax.view_init(elev=float(args.elev), azim=float(args.azim))

        fig.colorbar(surf_raw, ax=ax_raw, shrink=0.70, aspect=18, pad=0.08, label=value_key)
        fig.colorbar(surf_smooth, ax=ax_smooth, shrink=0.70, aspect=18, pad=0.08, label=value_key)
        fig.tight_layout()
    else:
        fig = plt.figure(figsize=(10.5, 7.2))
        ax = fig.add_subplot(111, projection="3d")
        surf = _plot_surface_or_trisurf(ax, aa=aa, bb=bb, zz=zz, complete=complete, vmin=vmin, vmax=vmax)

        ax.set_xlabel("alpha")
        ax.set_ylabel("beta")
        ax.set_zlabel(value_key)

        title = f"3D Landscape (alpha, beta, {value_key})"
        if title_suffix:
            title += " | " + ", ".join(title_suffix)
        ax.set_title(title)

        ax.view_init(elev=float(args.elev), azim=float(args.azim))
        fig.colorbar(surf, shrink=0.68, aspect=18, pad=0.08, label=value_key)
        fig.tight_layout()

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=int(args.dpi))
    plt.close(fig)
    return use_compare_smoothed


def _plot_2d_curve(
    plot_rows: List[Dict[str, Any]],
    value_key: str,
    title_suffix: List[str],
    args: argparse.Namespace,
    output_png: Path,
) -> bool:
    if len(plot_rows) < 2:
        raise ValueError("Need at least 2 points to draw 2D curve")

    sorted_rows = sorted(plot_rows, key=lambda r: float(r["alpha"]))
    alphas = np.asarray([float(r["alpha"]) for r in sorted_rows], dtype=np.float64)
    values = np.asarray([float(r["value"]) for r in sorted_rows], dtype=np.float64)

    use_compare_smoothed = (not args.no_compare_smoothed) and float(args.smooth_sigma) > 0.0
    values_smooth: Optional[np.ndarray] = None
    if use_compare_smoothed:
        kernel = _gaussian_kernel_1d(sigma=float(args.smooth_sigma), radius=args.smooth_radius)
        values_smooth = _smooth_vector_nan(values, kernel)

    base_title = f"2D Landscape (alpha, {value_key})"
    if title_suffix:
        base_title += " | " + ", ".join(title_suffix)

    if use_compare_smoothed and values_smooth is not None:
        fig, (ax_raw, ax_smooth) = plt.subplots(1, 2, figsize=(13.0, 5.0), sharey=True)
        ax_raw.plot(alphas, values, marker="o", linewidth=1.5, color="#1f77b4")
        ax_smooth.plot(alphas, values_smooth, marker="o", linewidth=1.5, color="#ff7f0e")

        ax_raw.set_title(base_title + " | raw")
        ax_smooth.set_title(base_title + f" | gaussian smooth sigma={float(args.smooth_sigma):.2f}")
        for ax in (ax_raw, ax_smooth):
            ax.set_xlabel("alpha")
            ax.set_ylabel(value_key)
            ax.grid(True, alpha=0.3)
            ax.axvline(0.0, color="gray", linestyle=":", linewidth=0.8)
        fig.tight_layout()
    else:
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        ax.plot(alphas, values, marker="o", linewidth=1.5, color="#1f77b4")
        ax.set_xlabel("alpha")
        ax.set_ylabel(value_key)
        ax.set_title(base_title)
        ax.grid(True, alpha=0.3)
        ax.axvline(0.0, color="gray", linestyle=":", linewidth=0.8)
        fig.tight_layout()

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=int(args.dpi))
    plt.close(fig)
    return use_compare_smoothed


def main() -> None:
    args = parse_args()

    input_csv = Path(args.input_csv).resolve()
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    has_beta, value_key = _detect_csv_schema(input_csv)

    output_png = Path(args.output_png).resolve() if args.output_png else _default_output_png(input_csv, has_beta)

    rows = _read_rows(input_csv, has_beta=has_beta, value_key=value_key)
    if not rows:
        raise ValueError("No valid rows in input CSV")

    if args.precision is not None:
        rows = [r for r in rows if str(r["precision"]) == str(args.precision)]
    if args.mode is not None:
        rows = [r for r in rows if str(r["mode"]) == str(args.mode)]
    if args.sample_id is not None:
        rows = [r for r in rows if str(r["sample_id"]) == str(args.sample_id)]

    if not rows:
        raise ValueError("No rows left after filters")

    sample_ids = sorted({str(r["sample_id"]) for r in rows})
    precision_values = sorted({str(r["precision"]) for r in rows})

    title_suffix: List[str] = []
    if len(precision_values) == 1:
        title_suffix.append(f"precision={precision_values[0]}")

    plot_rows: List[Dict[str, Any]]
    if args.aggregate != "none":
        plot_rows = _aggregate_rows(rows, method=args.aggregate, has_beta=has_beta)
        title_suffix.append(f"aggregate={args.aggregate}")
        title_suffix.append(f"n_sample_ids={len(sample_ids)}")
    else:
        if args.sample_id is None and len(sample_ids) > 1:
            selected = sample_ids[0]
            rows = [r for r in rows if str(r["sample_id"]) == selected]
            sample_ids = [selected]
            print(f"[plot] multiple sample_id found; auto-selected sample_id={selected}")

        plot_rows = []
        for r in rows:
            row: Dict[str, Any] = {
                "alpha": float(r["alpha"]),
                "value": float(r["value"]),
            }
            if has_beta:
                row["beta"] = float(r["beta"])
            plot_rows.append(row)
        if len(sample_ids) == 1:
            title_suffix.append(f"sample_id={sample_ids[0]}")

    if has_beta:
        use_compare_smoothed = _plot_3d_surface(plot_rows, value_key, title_suffix, args, output_png)
    else:
        use_compare_smoothed = _plot_2d_curve(plot_rows, value_key, title_suffix, args, output_png)

    z_vals = np.asarray([float(r["value"]) for r in plot_rows], dtype=np.float64)
    print(f"[plot] input={input_csv}")
    print(f"[plot] dimension={'2D (alpha,beta)' if has_beta else '1D (alpha)'}")
    print(f"[plot] value_column={value_key}")
    print(f"[plot] output={output_png}")
    print(f"[plot] points={len(plot_rows)}")
    print(f"[plot] compare_smoothed={use_compare_smoothed}")
    if use_compare_smoothed:
        print(f"[plot] smooth_sigma={float(args.smooth_sigma):.4f}")
    print(f"[plot] z_min={float(np.min(z_vals)):.6f}, z_max={float(np.max(z_vals)):.6f}, z_mean={float(np.mean(z_vals)):.6f}")


if __name__ == "__main__":
    main()
