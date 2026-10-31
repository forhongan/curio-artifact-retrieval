"""Descriptive retrieval diagnostics, never a probability of relevance.

Only pass original cosine similarities here. Reciprocal-rank, min/max-scaled
and catalogue-fused scores do not measure embedding separation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any


def _field(parsed: Any, name: str, default: Any = None) -> Any:
    return parsed.get(name, default) if isinstance(parsed, Mapping) else getattr(parsed, name, default)


def retrieval_evidence(
    *,
    parsed: Any,
    dense_scores: Sequence[float] | None,
    dense_available: bool,
    results_available: bool,
    degraded: bool = False,
    probe_coverage: Mapping[str, Sequence[Mapping]] | None = None,
    probe_available: bool = False,
    probe_weight_positive: bool = False,
    probe_top_support: Sequence[Mapping] | None = None,
) -> dict:
    """Summarize evidence without calibrated correctness claims.

    ``dense_scores`` contains actual cosine values in descending dense order,
    over the eligible candidate set *before* fusion and top-k truncation.
    ``probe_top_support`` optionally contains one entry per query constraint:
    slot, value, negated, dense_top_support, max_support. The last two values
    compare the selected dense winner's constraint support with that probe's
    maximum over the same candidate set. For negations use 1-P for both.
    Equality (including numerical ties) means the dense winner is also a probe
    winner; it does not establish relevance. Missing diagnostics stay unknown.
    Probe weight zero may still permit a diagnostic, but no probe is applied.

    Parsed slots and probe coverage describe recognized constraints only.
    Neither their completeness nor a parser's own numeric confidence measures
    how faithfully the complete natural-language intent was understood.
    """
    values = [] if dense_scores is None else [float(value) for value in dense_scores]
    if any(not math.isfinite(value) or abs(value) > 1.00001 for value in values):
        raise ValueError("dense_scores must contain finite cosine similarities")
    if any(left < right for left, right in zip(values, values[1:])):
        raise ValueError("dense_scores must be in descending dense order")
    if values and not dense_available:
        raise ValueError("dense_scores cannot be supplied when dense is unavailable")

    known_coverage = {}
    for polarity in ("positive", "negations"):
        for entry in (probe_coverage or {}).get(polarity, []):
            key = (entry.get("slot"), entry.get("value"), polarity == "negations")
            known_coverage[key] = entry
    constraints = []
    seen = set()

    def add(slot, value, *, negated=False, reason=None):
        key = (slot, str(value), negated)
        if key in seen:
            return
        seen.add(key)
        coverage = known_coverage.get((slot, value, negated), {})
        covered = not reason and probe_available and coverage.get("status") == "covered"
        constraints.append({
            "slot": slot, "value": value, "negated": negated,
            "status": "covered" if covered else "unknown",
            "reason": (reason or coverage.get("reason") or
                       (None if covered else "class_not_covered" if probe_available else "probe_not_available")),
            "basis": "image_probe_class" if covered else "unverified",
        })

    for slot, slot_values in (_field(parsed, "slots", {}) or {}).items():
        for value in slot_values:
            add(slot, value)
    for entry in _field(parsed, "negations", []) or []:
        add(entry["slot"], entry["value"], negated=True)
    era = _field(parsed, "era")
    if era is not None:
        add("era", _field(parsed, "era_label") or str(era), reason="historical_attribution_not_visual_verification")
    for unsupported in _field(parsed, "unsupported", []) or []:
        add("unsupported", unsupported, reason="unsupported_constraint")

    covered = sum(entry["status"] == "covered" for entry in constraints)
    total = len(constraints)
    coverage = {
        "covered": covered, "total": total,
        "ratio": covered / total if total else None,
        "status": "covered" if total and covered == total else "partial" if covered else "unknown",
        "scope": "recognized_query_constraints_only",
        "full_query_verified": False,
        "constraints": constraints,
    }

    comparisons = []
    compared = set()
    eligible = {(entry["slot"], entry["value"], entry["negated"])
                for entry in constraints if entry["status"] == "covered"}
    if dense_available and values and results_available and probe_available:
        for entry in probe_top_support or []:
            key = (entry.get("slot"), entry.get("value"), bool(entry.get("negated", False)))
            if key not in eligible or key in compared:
                continue
            support, maximum = float(entry["dense_top_support"]), float(entry["max_support"])
            if (not math.isfinite(support) or not math.isfinite(maximum)
                    or not 0 <= support <= 1 or not 0 <= maximum <= 1
                    or support > maximum + 1e-7):
                raise ValueError("probe supports must be finite [0, 1] values with support <= maximum")
            compared.add(key)
            comparisons.append({
                "slot": key[0], "value": key[1], "negated": key[2],
                "dense_top_support": support, "max_support": maximum,
                "agrees": math.isclose(support, maximum, rel_tol=0, abs_tol=1e-7),
            })
    agreeing = sum(entry["agrees"] for entry in comparisons)
    measured = len(comparisons)
    agreement = {
        "status": ("not_measured" if not measured else "agree" if agreeing == measured
                   else "disagree" if not agreeing else "mixed"),
        "compared": measured, "agreeing": agreeing,
        "fraction": agreeing / measured if measured else None,
        "definition": "selected_dense_winner_is_among_each_probe_constraint_top_candidates",
        "comparisons": comparisons,
        "independent_validation": False,
    }
    has_dense_evidence = bool(dense_available and results_available and values)
    if not has_dense_evidence:
        level, label = "unknown", "无法判断"
    elif degraded or not total or covered != total:
        level, label = "limited", "证据有限"
    else:
        level, label = "evidence_available", "有模型证据（未校准）"
    return {
        "level": level, "label": label,
        "basis": "这是检索诊断，不是人工确认、视觉核验或结果正确率。",
        "calibration": {"status": "uncalibrated", "correctness_probability": None},
        "parser": {"confidence_calibrated": False, "full_query_understanding_verified": False},
        "visual_coverage": coverage,
        "ranking_separation": {
            "score_space": "cosine_similarity", "candidate_count": len(values),
            "top1": values[0] if values else None,
            "top2": values[1] if len(values) >= 2 else None,
            "top1_top2_margin": values[0] - values[1] if len(values) >= 2 else None,
            "interpretation": "descriptive_only_not_correctness_probability",
        },
        "signal_availability": {
            "dense": bool(dense_available), "probes": bool(probe_available),
            "probes_applied": bool(probe_available and probe_weight_positive and covered and has_dense_evidence),
        },
        "signal_agreement": agreement,
        "degraded": bool(degraded),
    }
