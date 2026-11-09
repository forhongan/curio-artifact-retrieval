"""Explicit catalogue-query sets from independently calibrated image probes.

The catalogue supplies supervision and evaluation truth.  At inference, only
the image embedding predictions and dev-frozen per-class thresholds are used.
Missing/ambiguous catalogue facets are excluded from truth rather than treated
as negatives.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np

from curio.evaluate.catalogue_boundary import aggregate_query_metrics, explicit_metrics


DOMAIN_TO_PROBE = {
    "form": "object_type",
    "fine_material": "material",
    "broad_material": "broad_material",
    "era": "era",
    "motif": "motifs",
    "color": "colors",
}


def supported_atoms(atoms: Iterable[tuple[str, str]], probe_report: Mapping) -> tuple[bool, list[str]]:
    """Report unsupported atoms without silently replacing them by a guess."""
    missing = []
    for domain, value in atoms:
        task = DOMAIN_TO_PROBE.get(domain)
        task_report = probe_report.get("tasks", {}).get(task or "", {})
        if not task or value not in task_report.get("classes", []):
            missing.append(f"{domain}/{value}")
    return not missing, missing


def threshold_rule(probability: np.ndarray, *, threshold: float, mode: str) -> np.ndarray:
    """Build a candidate predicate from a frozen per-class rule."""
    if mode == "at_least":
        return np.asarray(probability) >= threshold
    if mode == "at_most":
        return (1 - np.asarray(probability)) >= threshold
    raise ValueError(f"Unknown threshold mode: {mode}")


def explicit_probe_evaluation(
    prepared: list[dict],
    probe_report: Mapping,
    *,
    gate_threshold: float,
    gate_on_max: bool = True,
    require_class_winner: bool = False,
) -> list[dict]:
    """Evaluate per-class-threshold sets with a separate query existence gate.

    ``prepared`` is produced by the runner and contains a query specification,
    candidate IDs, their truth set, and the precomputed probe probability
    matrices.  Gate and candidate decisions are deliberately separate: a query
    can either return no matches or a set, while each accepted image must pass
    every available atom threshold.
    """
    output = []
    for entry in prepared:
        spec, data = entry["spec"], entry["data"]
        supported, missing = supported_atoms(spec.atoms, probe_report)
        if not supported:
            selected = set()
            gate_score = 0.0
            gate_open = False
        else:
            probabilities = []
            for domain, value in spec.atoms:
                task = DOMAIN_TO_PROBE[domain]
                task_report = probe_report["tasks"][task]
                class_index = task_report["classes"].index(value)
                threshold = float(task_report["models"][value]["selected_threshold"])
                probabilities.append(threshold_rule(
                    data["probabilities"][task][:, class_index],
                    threshold=threshold,
                    mode="at_least",
                ))
            matrix = np.asarray(probabilities, dtype=bool)
            if require_class_winner:
                for domain in dict.fromkeys(domain for domain, _ in spec.atoms):
                    task = DOMAIN_TO_PROBE[domain]
                    task_report = probe_report["tasks"][task]
                    requested = [value for atom_domain, value in spec.atoms
                                 if atom_domain == domain]
                    # A conjunction normally has one value per domain.  For
                    # duplicates, passing any requested class is semantically
                    # consistent with an OR inside that domain.
                    domain_scores = data["probabilities"][task]
                    matrix = matrix & (domain_scores.argmax(axis=1)
                                       == task_report["classes"].index(requested[0]))
            candidate_score = matrix.mean(axis=0) if matrix.ndim > 1 else matrix
            gate_score = float(candidate_score.max()) if len(candidate_score) else 0.0
            gate_open = gate_score >= gate_threshold
            keep = matrix.all(axis=0) & gate_open if matrix.ndim > 1 else matrix & gate_open
            selected = {item_id for item_id, accepted in zip(data["ids"], keep) if accepted}
        row = explicit_metrics(selected, data["relevant"])
        row.update({
            "qid": spec.qid,
            "family": spec.family,
            "supported": supported,
            "missing_atoms": missing,
            "gate_threshold": float(gate_threshold),
            "gate_score": gate_score,
            "gate_open": bool(gate_open),
            "gate_basis": "maximum candidate pass fraction" if gate_on_max else "candidate minimum",
            "require_class_winner": require_class_winner,
            "unknown_excluded": int(data["unknown_excluded"]),
        })
        output.append(row)
    return output


def two_stage_summary(rows: list[dict]) -> dict:
    """Expose gate performance separately from candidate-set performance."""
    supported = [row for row in rows if row["supported"]]
    positive = [row for row in supported if not row["no_match"]]
    no_match = [row for row in supported if row["no_match"]]
    gate_tp = sum(row["gate_open"] for row in positive)
    gate_fn = len(positive) - gate_tp
    gate_fp = sum(row["gate_open"] for row in no_match)
    gate_tn = len(no_match) - gate_fp
    gate_precision = gate_tp / (gate_tp + gate_fp) if gate_tp + gate_fp else 0.0
    gate_recall = gate_tp / len(positive) if positive else 0.0
    gate_f1 = (2 * gate_precision * gate_recall / (gate_precision + gate_recall)
               if gate_precision + gate_recall else 0.0)
    return {
        **aggregate_query_metrics(supported),
        "unsupported_queries": len(rows) - len(supported),
        "gate_tp": gate_tp,
        "gate_fp": gate_fp,
        "gate_tn": gate_tn,
        "gate_fn": gate_fn,
        "gate_precision": gate_precision,
        "gate_recall": gate_recall,
        "gate_f1": gate_f1,
        "gate_balanced_accuracy": (gate_recall + (gate_tn / len(no_match) if no_match else 0.0)) / 2,
        "gate_accuracy": (gate_tp + gate_tn) / len(supported) if supported else 0.0,
        "positive_nonempty_rate": sum(not row["empty_prediction"] for row in positive) / len(positive)
        if positive else 0.0,
    }


def choose_gate(dev_rows_by_threshold: Mapping[float, list[dict]]) -> tuple[float, dict]:
    """Select the development gate by gate F1, then set F1 and recall."""
    choices = []
    for threshold, rows in dev_rows_by_threshold.items():
        summary = two_stage_summary(rows)
        choices.append((
            summary["gate_f1"], summary["macro_set_score"], summary["macro_f1"],
            -float(threshold), float(threshold), summary,
        ))
    if not choices:
        raise ValueError("No development gate thresholds were supplied")
    *_, threshold, summary = max(choices)
    return threshold, summary
