"""Detailed analysis for boundary-drift per-sample results.

This script focuses on:
1) Delta sign distribution (delta_q > 0 / < 0 / == 0) by precision.
2) Category-level differences under each precision.

Usage:
  python scripts/analyze_boundary_drift_results.py \
    --input_csv outputs/<exp>/per_sample/boundary_drift_per_sample.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.algorithm.boundary_metrics import classify_boundary_band, classify_boundary_band_tau_scaled, resolve_boundary_tau


FOCUSED_BANDS = ("ultra_near", "sub_near", "non_near")


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


def _to_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _delta_sign(delta: float, eps: float = 1e-12) -> str:
    if delta > eps:
        return "positive"
    if delta < -eps:
        return "negative"
    return "zero"


def _safe_rate(num: int, den: int) -> float:
    if den <= 0:
        return 0.0
    return float(num / den)


def _parse_topk_values(raw: str) -> List[int]:
    out: List[int] = []
    for x in str(raw).split(","):
        s = x.strip()
        if not s:
            continue
        k = int(s)
        if k > 0:
            out.append(k)
    if not out:
        return [20, 50]
    return sorted(set(out))


def _compute_attack_entry_score(m_q: Optional[float], delta_q: Optional[float], eps: float) -> Optional[float]:
    """Candidate attack-entry score.

    We follow the intended behavior described by user:
    - larger unsafe push -> larger (-delta_q)
    - closer to boundary -> smaller |m_q|

    Score: A_q(z) = (-delta_q) / (|m_q| + eps)
    """
    if m_q is None or delta_q is None:
        return None
    denom = abs(float(m_q)) + float(eps)
    if denom <= 0.0:
        return None
    return float((-float(delta_q)) / denom)


def _build_attack_score_stat_entry(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    vals = [float(r["attack_entry_score"]) for r in rows if r.get("attack_entry_score") is not None]
    if not vals:
        return {
            "n_rows": len(rows),
            "n_with_score": 0,
            "score_mean": 0.0,
            "score_median": 0.0,
            "score_min": 0.0,
            "score_max": 0.0,
            "score_p90": 0.0,
        }

    arr = sorted(vals)
    p90_idx = int(round(0.9 * (len(arr) - 1)))
    return {
        "n_rows": len(rows),
        "n_with_score": len(arr),
        "score_mean": float(sum(arr) / len(arr)),
        "score_median": float(median(arr)),
        "score_min": float(arr[0]),
        "score_max": float(arr[-1]),
        "score_p90": float(arr[p90_idx]),
    }


def _build_stat_entry(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    deltas = [x["delta_q"] for x in rows if x.get("delta_q") is not None]
    flips = [x["flip"] for x in rows if x.get("flip") is not None]

    pos = sum(1 for d in deltas if _delta_sign(d) == "positive")
    neg = sum(1 for d in deltas if _delta_sign(d) == "negative")
    zero = sum(1 for d in deltas if _delta_sign(d) == "zero")

    return {
        "n_rows": len(rows),
        "n_with_delta": len(deltas),
        "delta_positive_count": pos,
        "delta_negative_count": neg,
        "delta_zero_count": zero,
        "delta_positive_rate": _safe_rate(pos, len(deltas)),
        "delta_negative_rate": _safe_rate(neg, len(deltas)),
        "delta_zero_rate": _safe_rate(zero, len(deltas)),
        "delta_mean": float(sum(deltas) / len(deltas)) if deltas else 0.0,
        "delta_median": float(median(deltas)) if deltas else 0.0,
        "delta_min": float(min(deltas)) if deltas else 0.0,
        "delta_max": float(max(deltas)) if deltas else 0.0,
        "flip_rate": float(sum(flips) / len(flips)) if flips else 0.0,
    }


def _build_label_cfg(ultra_near_upper: float, sub_near_upper: float, non_near_lower: float) -> Dict[str, str]:
    return {
        "ultra_near": f"(0,{ultra_near_upper}]",
        "sub_near": f"({ultra_near_upper},{sub_near_upper}]",
        "mid_range": f"({sub_near_upper},{non_near_lower}]",
        "non_near": f"({non_near_lower},+inf)",
        "non_positive": "(-inf,0]",
    }


def _build_label_cfg_tau_scaled(ultra_near_upper: float, sub_near_upper: float, non_near_lower: float, tau: float) -> Dict[str, str]:
    return {
        "ultra_near": f"(0,{ultra_near_upper}*tau]={ultra_near_upper * tau}",
        "sub_near": f"({ultra_near_upper}*tau,{sub_near_upper}*tau]=({ultra_near_upper * tau},{sub_near_upper * tau}]",
        "mid_range": f"({sub_near_upper}*tau,{non_near_lower}*tau]=({sub_near_upper * tau},{non_near_lower * tau}]",
        "non_near": f"({non_near_lower}*tau,+inf)=({non_near_lower * tau},+inf)",
        "non_positive": "(-inf,0]",
    }


def _read_rows(input_csv: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with input_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            item = dict(r)
            item["m_fp"] = _to_float(item.get("m_fp"))
            item["m_q"] = _to_float(item.get("m_q"))
            item["delta_q"] = _to_float(item.get("delta_q"))
            item["flip"] = _to_int(item.get("flip"))
            item["model_precision"] = str(item.get("model_precision") or "")
            item["category"] = str(item.get("category") or "unknown")
            rows.append(item)
    return rows


def _default_output_dir(input_csv: Path) -> Path:
    # Typical input: .../<exp>/per_sample/boundary_drift_per_sample.csv
    if input_csv.parent.name == "per_sample":
        return input_csv.parent.parent / "summary"
    return input_csv.parent


def _analyze(
    rows: List[Dict[str, Any]],
    ultra_near_upper: float,
    sub_near_upper: float,
    non_near_lower: float,
    band_mode: str,
    tau_mode: str,
    tau_fixed: float,
    tau_quantile: float,
    attack_score_eps: float,
    topk_values: List[int],
) -> Dict[str, Any]:
    quant_rows = [r for r in rows if str(r.get("model_precision", "")).lower() != "fp"]
    fp_rows = [r for r in rows if str(r.get("model_precision", "")).lower() == "fp"]
    m_fp_values_for_tau = [float(r["m_fp"]) for r in fp_rows if r.get("m_fp") is not None]

    tau = resolve_boundary_tau(
        m_fp_values=m_fp_values_for_tau,
        mode=("quantile" if str(tau_mode).lower() == "quantile" else "fixed"),
        fixed_tau=float(tau_fixed),
        quantile=float(tau_quantile),
    )

    by_precision: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_category_precision: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)

    for r in quant_rows:
        precision = r["model_precision"]
        category = r["category"]
        r["attack_entry_score"] = _compute_attack_entry_score(
            m_q=r.get("m_q"),
            delta_q=r.get("delta_q"),
            eps=float(attack_score_eps),
        )
        m_fp = r.get("m_fp")
        if m_fp is None:
            band = "unknown"
        else:
            if str(band_mode).lower() == "tau_scaled":
                band = classify_boundary_band_tau_scaled(
                    m_fp=float(m_fp),
                    tau=float(tau),
                    ultra_near_upper=ultra_near_upper,
                    sub_near_upper=sub_near_upper,
                    non_near_lower=non_near_lower,
                )
            else:
                band = classify_boundary_band(
                    m_fp=float(m_fp),
                    ultra_near_upper=ultra_near_upper,
                    sub_near_upper=sub_near_upper,
                    non_near_lower=non_near_lower,
                )
        r["boundary_band"] = band
        by_precision[precision].append(r)
        by_category_precision[(category, precision)].append(r)

    by_band_precision: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    by_band_category_precision: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in quant_rows:
        band = str(r.get("boundary_band", "unknown"))
        precision = str(r.get("model_precision", "unknown"))
        category = str(r.get("category", "unknown"))
        by_band_precision[(band, precision)].append(r)
        by_band_category_precision[(band, category, precision)].append(r)

    precision_stats = {p: _build_stat_entry(items) for p, items in sorted(by_precision.items())}

    category_precision_stats: Dict[str, Dict[str, Any]] = defaultdict(dict)
    for (category, precision), items in sorted(by_category_precision.items()):
        category_precision_stats[category][precision] = _build_stat_entry(items)

    category_contrast: Dict[str, Dict[str, Any]] = {}
    for category, pstats in category_precision_stats.items():
        means = {p: s["delta_mean"] for p, s in pstats.items()}
        pos_rates = {p: s["delta_positive_rate"] for p, s in pstats.items()}
        neg_rates = {p: s["delta_negative_rate"] for p, s in pstats.items()}

        if means:
            mean_values = list(means.values())
            pos_values = list(pos_rates.values())
            neg_values = list(neg_rates.values())
            category_contrast[category] = {
                "delta_mean_by_precision": means,
                "delta_positive_rate_by_precision": pos_rates,
                "delta_negative_rate_by_precision": neg_rates,
                "delta_mean_gap_max_minus_min": float(max(mean_values) - min(mean_values)),
                "delta_positive_rate_gap_max_minus_min": float(max(pos_values) - min(pos_values)),
                "delta_negative_rate_gap_max_minus_min": float(max(neg_values) - min(neg_values)),
            }

    top_categories_by_gap = sorted(
        (
            {
                "category": c,
                "delta_mean_gap_max_minus_min": v["delta_mean_gap_max_minus_min"],
                "delta_positive_rate_gap_max_minus_min": v["delta_positive_rate_gap_max_minus_min"],
                "delta_negative_rate_gap_max_minus_min": v["delta_negative_rate_gap_max_minus_min"],
            }
            for c, v in category_contrast.items()
        ),
        key=lambda x: x["delta_mean_gap_max_minus_min"],
        reverse=True,
    )

    band_precision_stats: Dict[str, Dict[str, Any]] = defaultdict(dict)
    for (band, precision), items in sorted(by_band_precision.items()):
        band_precision_stats[band][precision] = _build_stat_entry(items)

    focused_band_precision_stats: Dict[str, Dict[str, Any]] = {}
    for band in FOCUSED_BANDS:
        if band in band_precision_stats:
            focused_band_precision_stats[band] = band_precision_stats[band]

    band_category_precision_stats: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(lambda: defaultdict(dict))
    for (band, category, precision), items in sorted(by_band_category_precision.items()):
        band_category_precision_stats[band][category][precision] = _build_stat_entry(items)

    attack_score_by_precision: Dict[str, Dict[str, Any]] = {}
    for p, items in sorted(by_precision.items()):
        attack_score_by_precision[p] = _build_attack_score_stat_entry(items)

    attack_score_by_category_precision: Dict[str, Dict[str, Any]] = defaultdict(dict)
    for (category, precision), items in sorted(by_category_precision.items()):
        attack_score_by_category_precision[category][precision] = _build_attack_score_stat_entry(items)

    top_attack_candidates: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(dict)
    for p, items in sorted(by_precision.items()):
        scored = [x for x in items if x.get("attack_entry_score") is not None]
        scored.sort(key=lambda x: float(x["attack_entry_score"]), reverse=True)
        for k in topk_values:
            sliced = scored[:k]
            top_attack_candidates[p][str(k)] = [
                {
                    "rank": i + 1,
                    "sample_id": str(x.get("sample_id", "")),
                    "category": str(x.get("category", "unknown")),
                    "source": str(x.get("source", "")),
                    "model_precision": str(x.get("model_precision", "")),
                    "m_fp": x.get("m_fp"),
                    "m_q": x.get("m_q"),
                    "delta_q": x.get("delta_q"),
                    "attack_entry_score": x.get("attack_entry_score"),
                }
                for i, x in enumerate(sliced)
            ]

    return {
        "band_thresholds": {
            "band_mode": str(band_mode).lower(),
            "ultra_near_upper": ultra_near_upper,
            "sub_near_upper": sub_near_upper,
            "non_near_lower": non_near_lower,
            "tau": float(tau),
            "tau_mode": str(tau_mode).lower(),
            "tau_fixed": float(tau_fixed),
            "tau_quantile": float(tau_quantile),
            "labels": (
                _build_label_cfg_tau_scaled(ultra_near_upper, sub_near_upper, non_near_lower, tau)
                if str(band_mode).lower() == "tau_scaled"
                else _build_label_cfg(ultra_near_upper, sub_near_upper, non_near_lower)
            ),
        },
        "overall": {
            "n_rows": len(rows),
            "n_quant_rows": len(quant_rows),
            "precisions": sorted(list(by_precision.keys())),
            "categories": sorted(list({str(r.get("category", "unknown")) for r in quant_rows})),
        },
        "delta_sign_by_precision": precision_stats,
        "delta_sign_by_category_and_precision": category_precision_stats,
        "delta_sign_by_boundary_band_and_precision": band_precision_stats,
        "delta_sign_by_boundary_band_and_precision_focused": focused_band_precision_stats,
        "delta_sign_by_boundary_band_category_and_precision": band_category_precision_stats,
        "attack_entry_score": {
            "formula": "A_q(z)=(-delta_q)/(abs(m_q)+eps)",
            "eps": float(attack_score_eps),
            "topk_values": topk_values,
            "by_precision": attack_score_by_precision,
            "by_category_and_precision": attack_score_by_category_precision,
            "top_candidates": top_attack_candidates,
        },
        "category_contrast": category_contrast,
        "top_categories_by_delta_mean_gap": top_categories_by_gap,
    }


def _write_json(path: Path, content: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(content, f, ensure_ascii=False, indent=2)


def _write_precision_csv(path: Path, stats: Dict[str, Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "precision",
        "n_rows",
        "n_with_delta",
        "delta_positive_count",
        "delta_negative_count",
        "delta_zero_count",
        "delta_positive_rate",
        "delta_negative_rate",
        "delta_zero_rate",
        "delta_mean",
        "delta_median",
        "delta_min",
        "delta_max",
        "flip_rate",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for precision, s in sorted(stats.items()):
            row = {"precision": precision}
            row.update(s)
            writer.writerow(row)


def _write_category_precision_csv(path: Path, nested: Dict[str, Dict[str, Dict[str, Any]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "precision",
        "category",
        "n_rows",
        "n_with_delta",
        "delta_positive_count",
        "delta_negative_count",
        "delta_zero_count",
        "delta_positive_rate",
        "delta_negative_rate",
        "delta_zero_rate",
        "delta_mean",
        "delta_median",
        "delta_min",
        "delta_max",
        "flip_rate",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for category, pstats in sorted(nested.items()):
            for precision, s in sorted(pstats.items()):
                row = {"category": category, "precision": precision}
                row.update(s)
                writer.writerow(row)


def _write_band_precision_csv(path: Path, nested: Dict[str, Dict[str, Dict[str, Any]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "precision",
        "boundary_band",
        "n_rows",
        "n_with_delta",
        "delta_positive_count",
        "delta_negative_count",
        "delta_zero_count",
        "delta_positive_rate",
        "delta_negative_rate",
        "delta_zero_rate",
        "delta_mean",
        "delta_median",
        "delta_min",
        "delta_max",
        "flip_rate",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for band, pstats in sorted(nested.items()):
            for precision, s in sorted(pstats.items()):
                row = {"boundary_band": band, "precision": precision}
                row.update(s)
                writer.writerow(row)


def _write_band_category_precision_csv(path: Path, nested: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "precision",
        "boundary_band",
        "category",
        "n_rows",
        "n_with_delta",
        "delta_positive_count",
        "delta_negative_count",
        "delta_zero_count",
        "delta_positive_rate",
        "delta_negative_rate",
        "delta_zero_rate",
        "delta_mean",
        "delta_median",
        "delta_min",
        "delta_max",
        "flip_rate",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for band, cstats in sorted(nested.items()):
            for category, pstats in sorted(cstats.items()):
                for precision, s in sorted(pstats.items()):
                    row = {
                        "boundary_band": band,
                        "category": category,
                        "precision": precision,
                    }
                    row.update(s)
                    writer.writerow(row)


def _write_attack_topk_csv(path: Path, top_candidates: Dict[str, Dict[str, List[Dict[str, Any]]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "precision",
        "topk",
        "rank",
        "sample_id",
        "category",
        "source",
        "m_fp",
        "m_q",
        "delta_q",
        "attack_entry_score",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for precision, per_k in sorted(top_candidates.items()):
            for k, items in sorted(per_k.items(), key=lambda x: int(x[0])):
                for row in items:
                    out = {
                        "precision": precision,
                        "topk": int(k),
                        "rank": row.get("rank"),
                        "sample_id": row.get("sample_id"),
                        "category": row.get("category"),
                        "source": row.get("source"),
                        "m_fp": row.get("m_fp"),
                        "m_q": row.get("m_q"),
                        "delta_q": row.get("delta_q"),
                        "attack_entry_score": row.get("attack_entry_score"),
                    }
                    writer.writerow(out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detailed analysis for boundary drift per-sample CSV")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to boundary_drift_per_sample.csv")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save analysis outputs")
    parser.add_argument("--prefix", type=str, default="boundary_drift_detailed", help="Output file name prefix")
    parser.add_argument("--ultra_near_upper", type=float, default=0.05, help="Upper bound of ultra-near band")
    parser.add_argument("--sub_near_upper", type=float, default=0.10, help="Upper bound of sub-near band")
    parser.add_argument("--non_near_lower", type=float, default=10.0, help="Lower bound of non-near band")
    parser.add_argument(
        "--band_mode",
        type=str,
        default="tau_scaled",
        choices=["tau_scaled", "absolute"],
        help="Banding mode: tau_scaled uses coefficient*tau; absolute uses direct m_fp thresholds",
    )
    parser.add_argument("--tau_mode", type=str, default="quantile", choices=["fixed", "quantile"], help="Tau resolver mode")
    parser.add_argument("--tau_fixed", type=float, default=1.0, help="Fallback/fixed tau value")
    parser.add_argument("--tau_quantile", type=float, default=0.2, help="Quantile used when tau_mode=quantile")
    parser.add_argument("--attack_score_eps", type=float, default=1e-6, help="Epsilon in A_q score denominator")
    parser.add_argument("--attack_topk", type=str, default="20,50", help="Top-K list for attack entry ranking, e.g. 20,50")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv).resolve()
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    output_dir = Path(args.output_dir).resolve() if args.output_dir else _default_output_dir(input_csv)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_rows(input_csv)
    analysis = _analyze(
        rows,
        ultra_near_upper=float(args.ultra_near_upper),
        sub_near_upper=float(args.sub_near_upper),
        non_near_lower=float(args.non_near_lower),
        band_mode=str(args.band_mode),
        tau_mode=str(args.tau_mode),
        tau_fixed=float(args.tau_fixed),
        tau_quantile=float(args.tau_quantile),
        attack_score_eps=float(args.attack_score_eps),
        topk_values=_parse_topk_values(args.attack_topk),
    )

    json_path = output_dir / f"{args.prefix}.json"
    precision_csv_path = output_dir / f"{args.prefix}.by_precision.csv"
    category_precision_csv_path = output_dir / f"{args.prefix}.by_category_precision.csv"
    band_precision_csv_path = output_dir / f"{args.prefix}.by_boundary_band_precision.csv"
    focused_band_precision_csv_path = output_dir / f"{args.prefix}.by_boundary_band_precision.focused.csv"
    band_category_precision_csv_path = output_dir / f"{args.prefix}.by_boundary_band_category_precision.csv"
    attack_topk_csv_path = output_dir / f"{args.prefix}.attack_topk.csv"

    _write_json(json_path, analysis)
    _write_precision_csv(precision_csv_path, analysis["delta_sign_by_precision"])
    _write_category_precision_csv(category_precision_csv_path, analysis["delta_sign_by_category_and_precision"])
    _write_band_precision_csv(band_precision_csv_path, analysis["delta_sign_by_boundary_band_and_precision"])
    _write_band_precision_csv(
        focused_band_precision_csv_path,
        analysis["delta_sign_by_boundary_band_and_precision_focused"],
    )
    _write_band_category_precision_csv(
        band_category_precision_csv_path,
        analysis["delta_sign_by_boundary_band_category_and_precision"],
    )
    _write_attack_topk_csv(
        attack_topk_csv_path,
        analysis["attack_entry_score"]["top_candidates"],
    )

    print(f"[analysis] input={input_csv}")
    print(f"[analysis] rows={analysis['overall']['n_rows']}, quant_rows={analysis['overall']['n_quant_rows']}")
    print(f"[analysis] precisions={analysis['overall']['precisions']}")
    print(f"[analysis] boundary_band_thresholds={analysis['band_thresholds']}")
    print(f"[analysis] boundary_band_rule={analysis['band_thresholds']['band_mode']}")
    print(f"[analysis] attack_entry_score={analysis['attack_entry_score']['formula']}, eps={analysis['attack_entry_score']['eps']}")
    print(f"[analysis] outputs:")
    print(f"  - {json_path}")
    print(f"  - {precision_csv_path}")
    print(f"  - {category_precision_csv_path}")
    print(f"  - {band_precision_csv_path}")
    print(f"  - {focused_band_precision_csv_path}")
    print(f"  - {band_category_precision_csv_path}")
    print(f"  - {attack_topk_csv_path}")


if __name__ == "__main__":
    main()
