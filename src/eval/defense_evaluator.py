"""Defense evaluator for quantization-aware boundary calibration."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple


def _safe_rate(num: int, den: int) -> float:
    if den <= 0:
        return 0.0
    return float(num / den)


def _first_strength_by_condition(strength_values: List[float], metric_values: List[float], cond) -> Optional[float]:
    pairs = sorted(zip(strength_values, metric_values), key=lambda x: x[0])
    for s, v in pairs:
        if cond(float(v)):
            return float(s)
    return None


def summarize_defense_rows(
    rows: Iterable[Dict[str, Any]],
    target_asr: float = 0.5,
) -> Dict[str, Any]:
    """Summarize fp / quant / quant+defense trajectories.

    Required per-row fields:
    - attack_family
    - precision
    - strength
    - margin
    - margin_defended (optional for quant rows)
    - is_boundary_near (optional bool)
    """
    rows_list = list(rows)

    grouped: Dict[Tuple[str, str, float], List[Dict[str, Any]]] = {}
    for r in rows_list:
        key = (
            str(r.get("attack_family", "unknown")),
            str(r.get("precision", "unknown")),
            float(r.get("strength", 0.0)),
        )
        grouped.setdefault(key, []).append(r)

    table: List[Dict[str, Any]] = []
    index: Dict[str, Dict[str, Dict[float, Dict[str, float]]]] = {}

    for (family, precision, strength), items in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        margin_vals = [float(x.get("margin", 0.0)) for x in items if x.get("margin") is not None]
        asr = _safe_rate(sum(1 for m in margin_vals if m < 0.0), len(margin_vals))

        defended_vals = [float(x.get("margin_defended")) for x in items if x.get("margin_defended") is not None]
        asr_def = _safe_rate(sum(1 for m in defended_vals if m < 0.0), len(defended_vals)) if defended_vals else None

        restoration_vals = [float(x.get("margin_restoration", 0.0)) for x in items if x.get("margin_restoration") is not None]

        row = {
            "attack_family": family,
            "precision": precision,
            "strength": float(strength),
            "n_samples": len(items),
            "avg_margin": float(sum(margin_vals) / len(margin_vals)) if margin_vals else 0.0,
            "asr": asr,
            "avg_margin_defended": (float(sum(defended_vals) / len(defended_vals)) if defended_vals else None),
            "asr_defended": asr_def,
            "margin_restoration": (float(sum(restoration_vals) / len(restoration_vals)) if restoration_vals else 0.0),
        }
        table.append(row)

        index.setdefault(family, {}).setdefault(precision, {})[float(strength)] = {
            "avg_margin": row["avg_margin"],
            "asr": row["asr"],
            "avg_margin_defended": (row["avg_margin_defended"] if row["avg_margin_defended"] is not None else row["avg_margin"]),
            "asr_defended": (row["asr_defended"] if row["asr_defended"] is not None else row["asr"]),
        }

    critical: Dict[str, Dict[str, Any]] = {}
    shifts: List[Dict[str, Any]] = []

    for family, p_map in index.items():
        critical[family] = {}
        fp_map = p_map.get("fp", {})
        fp_strengths = sorted(fp_map.keys())
        fp_asrs = [fp_map[s]["asr"] for s in fp_strengths]
        fp_margins = [fp_map[s]["avg_margin"] for s in fp_strengths]

        s_fp_asr = _first_strength_by_condition(fp_strengths, fp_asrs, lambda x: x >= float(target_asr))
        s_fp_m0 = _first_strength_by_condition(fp_strengths, fp_margins, lambda x: x <= 0.0)

        for precision, s_map in p_map.items():
            strengths = sorted(s_map.keys())
            asrs = [s_map[s]["asr"] for s in strengths]
            margins = [s_map[s]["avg_margin"] for s in strengths]
            asrs_def = [s_map[s]["asr_defended"] for s in strengths]
            margins_def = [s_map[s]["avg_margin_defended"] for s in strengths]

            s_asr = _first_strength_by_condition(strengths, asrs, lambda x: x >= float(target_asr))
            s_m0 = _first_strength_by_condition(strengths, margins, lambda x: x <= 0.0)
            s_asr_def = _first_strength_by_condition(strengths, asrs_def, lambda x: x >= float(target_asr))
            s_m0_def = _first_strength_by_condition(strengths, margins_def, lambda x: x <= 0.0)

            critical[family][precision] = {
                "critical_strength_asr": s_asr,
                "critical_strength_margin_zero": s_m0,
                "critical_strength_asr_defended": s_asr_def,
                "critical_strength_margin_zero_defended": s_m0_def,
            }

            if precision != "fp":
                shifts.append(
                    {
                        "attack_family": family,
                        "precision": precision,
                        "delta_s_asr_quant_vs_fp": (None if (s_fp_asr is None or s_asr is None) else float(s_fp_asr - s_asr)),
                        "delta_s_asr_def_vs_fp": (None if (s_fp_asr is None or s_asr_def is None) else float(s_fp_asr - s_asr_def)),
                        "delta_s_margin_quant_vs_fp": (None if (s_fp_m0 is None or s_m0 is None) else float(s_fp_m0 - s_m0)),
                        "delta_s_margin_def_vs_fp": (None if (s_fp_m0 is None or s_m0_def is None) else float(s_fp_m0 - s_m0_def)),
                    }
                )

    return {
        "metrics_table": table,
        "critical_strengths": critical,
        "critical_strength_shifts": shifts,
        "target_asr": float(target_asr),
    }
