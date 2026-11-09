"""Explicit relevance-set prediction and boundary metrics for CV v4.

Candidate scores become a yes/no set.  Thresholds and fusion weights are fit
only on dev; the final test call consumes a frozen configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class BoundaryConfig:
    threshold: float
    dense_weight: float = 1.0
    attribute_weight: float = 0.0
    min_dense_percentile: float = 0.0
    require_attributes: bool = False

    def as_dict(self) -> dict:
        return {
            "threshold": self.threshold,
            "dense_weight": self.dense_weight,
            "attribute_weight": self.attribute_weight,
            "min_dense_percentile": self.min_dense_percentile,
            "require_attributes": self.require_attributes,
        }


def set_metrics(selected: set[str], relevant: set[str], gallery_size: int) -> dict:
    true_positive = len(selected & relevant)
    false_positive = len(selected - relevant)
    false_negative = len(relevant - selected)
    precision = true_positive / len(selected) if selected else 0.0
    recall = true_positive / len(relevant) if relevant else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "gallery_size": gallery_size,
        "relevant_size": len(relevant),
        "selected_size": len(selected),
        "true_positives": true_positive,
        "false_positives": false_positive,
        "false_negatives": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": 1 - false_positive / (gallery_size - len(relevant)) if gallery_size > len(relevant) else 0.0,
    }


def combine_scores(dense, attributes, config: BoundaryConfig, kept_mask=None):
    dense = np.asarray(dense, dtype=np.float64).ravel()
    if attributes is None:
        attributes = np.zeros_like(dense)
    attributes = np.asarray(attributes, dtype=np.float64).ravel()
    if dense.shape != attributes.shape:
        raise ValueError("Dense and attribute score vectors must align")
    if kept_mask is not None:
        kept_mask = np.asarray(kept_mask, dtype=bool).ravel()
        if kept_mask.shape != dense.shape:
            raise ValueError("kept_mask must align with candidate scores")
    else:
        kept_mask = np.ones(dense.shape, dtype=bool)
    score = config.dense_weight * dense + config.attribute_weight * attributes
    cutoff = np.quantile(dense[kept_mask], config.min_dense_percentile) if kept_mask.any() else np.inf
    yes = kept_mask & (dense >= cutoff) & (score >= config.threshold)
    if config.require_attributes:
        yes &= attributes > 0
    return yes


def calibrate_scores(scores, *, floor: float = 0.0, top_scale: float = 1.0) -> np.ndarray:
    """Map each query's cosine vector to a shared, interpretable zero-to-one scale.

    The calibration uses only the candidate distribution for the same query.
    ``top_scale`` is frozen on dev before test.  It does not treat the remaining
    gallery as labelled negatives.
    """
    values = np.asarray(scores, dtype=np.float64).ravel()
    transformed = np.maximum(0.0, values - float(floor))
    denominator = max(float(top_scale) - float(floor), 1e-12)
    return np.clip(transformed / denominator, 0.0, 1.0)


def query_confidence(dense, selected_count: int, config: dict) -> dict:
    """Return auditable yes-set confidence components, never a truth claim."""
    values = np.sort(np.asarray(dense, dtype=np.float64).ravel())[::-1]
    top = float(values[0]) if len(values) else 0.0
    runner = float(values[1]) if len(values) > 1 else 0.0
    accepted = values[:selected_count]
    boundary = float(accepted[-1]) if selected_count else 0.0
    agreement = float(np.mean(accepted > 0)) if selected_count else 0.0
    return {
        "top_score": top,
        "top_gap": top - runner,
        "boundary_score": boundary,
        "component_agreement": agreement,
        "selected_count": selected_count,
        "status": "confident" if selected_count and boundary >= float(config.get("threshold", 0)) else "uncertain",
    }


def empty_query_decision(dense, *, threshold: float, max_top_score: float) -> dict:
    """Allow an independent empty prediction when the strongest score is low.

    The gate is trained on dev no-match queries and is applied alongside the
    nonempty-set boundary.  It avoids forcing at least one candidate into every
    result while keeping the rule inspectable.
    """
    top = float(np.max(np.asarray(dense, dtype=np.float64))) if len(dense) else 0.0
    yes = top < threshold or top < max_top_score
    return {"empty": yes, "top_score": top, "threshold": float(threshold),
            "max_top_score": float(max_top_score)}


def boundary_metrics(selected: set[str], relevant: set[str], gallery_size: int) -> dict:
    """Set metrics with an explicit no-match accuracy component."""
    values = set_metrics(selected, relevant, gallery_size)
    values["no_match"] = not relevant
    values["no_match_correct"] = int(not relevant and not selected)
    values["no_match_errors"] = int((not relevant and bool(selected)) or (bool(relevant) and not selected))
    return values


def choose_config(queries, ids, gallery_sizes, candidates):
    """Select one configuration by macro query F1; ties use precision then recall."""
    rows = []
    best = None
    for config in candidates:
        metrics = []
        for query in queries:
            kept = query.get("attribute_kept")
            yes = combine_scores(query["dense_scores"], query.get("attribute_scores"), config, kept)
            selected = {item_id for item_id, value in zip(ids, yes) if value}
            metrics.append(set_metrics(selected, set(query["relevant"]), gallery_sizes[query["qid"]]))
        result = {
            "config": config.as_dict(),
            "macro_f1": float(np.mean([row["f1"] for row in metrics])),
            "macro_precision": float(np.mean([row["precision"] for row in metrics])),
            "macro_recall": float(np.mean([row["recall"] for row in metrics])),
            "mean_selected_size": float(np.mean([row["selected_size"] for row in metrics])),
        }
        rows.append(result)
        if best is None or (result["macro_f1"], result["macro_precision"], result["macro_recall"]) > (
            best["macro_f1"], best["macro_precision"], best["macro_recall"]
        ):
            best = result
    return best, rows


def write_evaluation(path: str | Path, evaluation: dict) -> None:
    Path(path).write_text(json.dumps(evaluation, ensure_ascii=False, indent=2), encoding="utf-8")
