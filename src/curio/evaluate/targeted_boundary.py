"""Evaluation helpers for targeted, AI-reviewed binary boundary experiments."""

import numpy as np


def validate_partitions(rows):
    """Reject leakage even when different queries refer to the same image/object."""
    keys, groups, hashes = set(), {}, {}
    for row in rows:
        if row["key"] in keys:
            raise ValueError("Duplicate query-image label")
        keys.add(row["key"])
        split = row["split"]
        if split not in {"train", "dev", "test"}:
            raise ValueError("Unknown partition")
        for mapping, field in ((groups, "object_group"), (hashes, "image_sha256")):
            value = row[field]
            if mapping.setdefault(value, split) != split:
                raise ValueError("Object/image leaks across partitions")
        if row["grade"] not in (-1, 0, 3) or row["decision"] is not (row["grade"] == 3):
            raise ValueError("Inconsistent binary label")


def policy_metrics(rows, accepted):
    """Unknown facts are strict policy No; report their acceptance separately."""
    yes = np.asarray(accepted, dtype=bool)
    if len(rows) != len(yes):
        raise ValueError("Prediction count mismatch")
    expected = np.array([r["decision"] for r in rows], dtype=bool)
    tp = int((yes & expected).sum())
    fp = int((yes & ~expected).sum())
    fn = int((~yes & expected).sum())
    return {
        "n": len(rows),
        "positive": int(expected.sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "unknown_accepted": sum(bool(a) and r["grade"] == -1 for r, a in zip(rows, yes)),
        "grayscale_accepted": sum(
            bool(a) and r["reason"] == "color_not_observable" for r, a in zip(rows, yes)
        ),
    }


def calibrate(rows, scores):
    """Dev-only empirical operating point; never imply statistical certification."""
    if any(r["split"] != "dev" for r in rows):
        raise ValueError("Threshold selection only accepts dev labels")
    scores = np.asarray(scores)
    choices = []
    for threshold in np.unique(scores[scores > 0]):
        m = policy_metrics(rows, scores >= threshold)
        if m["tp"] >= 2 and m["precision"] >= 0.9 and m["recall"] >= 0.5:
            choices.append((m["recall"], m["precision"], float(threshold)))
    threshold = max(choices)[2] if choices else 0.5
    return {
        "threshold": threshold,
        "goal_met": bool(choices),
        "metrics": policy_metrics(rows, scores >= threshold),
        "explicit_negative_count": sum(r["grade"] == 0 for r in rows),
        "release_approved": False,
    }


def positive_support_threshold(positive_vectors, margin=0.05):
    """Training-only neighborhood radius; self-similarity cannot set the radius."""
    x = np.asarray(positive_vectors)
    if len(x) < 2:
        raise ValueError("At least two distinct positive objects are required")
    similarity = x @ x.T
    np.fill_diagonal(similarity, -np.inf)
    return float(np.clip(similarity.max(axis=1).min() - margin, -1, 1))
