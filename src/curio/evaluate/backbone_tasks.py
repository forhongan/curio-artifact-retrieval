"""Catalogue-supervised image probes for backbone comparison.

The catalogue supplies labels; ranking never receives per-item catalogue text.
All splits are object-group isolated by the frozen catalogue.  A missing or
multi-valued catalogue facet is unknown and is excluded rather than turned into
an automatic negative.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression

from curio.evaluate.catalogue_tasks import single_label


@dataclass(frozen=True)
class DomainSplit:
    domain: str
    classes: tuple[str, ...]
    labels: tuple[str | None, ...]
    indices: dict[str, np.ndarray]
    coverage: dict[str, dict]


def domain_split(rows: list[dict], domain: str, *, min_train: int = 20) -> DomainSplit:
    """Build one mutually exclusive catalogue task without reading test outcomes."""
    labels = tuple(single_label(row, domain) for row in rows)
    train_counts = Counter(
        label for row, label in zip(rows, labels)
        if row["split"] == "train" and label is not None
    )
    classes = tuple(sorted(label for label, count in train_counts.items()
                           if count >= min_train))
    class_set = set(classes)
    indices = {
        split: np.asarray([
            index for index, (row, label) in enumerate(zip(rows, labels))
            if row["split"] == split and label in class_set
        ], dtype=np.int64)
        for split in ("train", "dev", "test")
    }
    coverage = {}
    for split in ("train", "dev", "test"):
        split_labels = [label for row, label in zip(rows, labels)
                        if row["split"] == split]
        known = sum(label is not None for label in split_labels)
        supported = sum(label in class_set for label in split_labels)
        total = len(split_labels)
        coverage[split] = {
            "total": total,
            "catalogue_known": known,
            "catalogue_unknown": total - known,
            "supported": supported,
            "known_but_unsupported": known - supported,
            "known_coverage": known / total if total else 0.0,
            "evaluated_coverage": supported / total if total else 0.0,
            "supported_of_known": supported / known if known else 0.0,
        }
    return DomainSplit(domain, classes, labels, indices, coverage)


def encode_labels(labels, classes: tuple[str, ...]) -> np.ndarray:
    lookup = {value: index for index, value in enumerate(classes)}
    return np.asarray([lookup[value] for value in labels], dtype=np.int64)


def _encoded_split(rows: list[dict], split: DomainSplit, split_name: str):
    indices = split.indices[split_name]
    truth = encode_labels([split.labels[index] for index in indices], split.classes)
    return indices, truth


def metric_values(truth: np.ndarray, prediction: np.ndarray, classes: tuple[str, ...]) -> dict:
    """Exact multiclass metrics from a confusion matrix.

    Primary macro F1 averages the union of true and predicted classes, matching
    sklearn's default. Observed-only and all-trained-class means are controls.
    """
    truth = np.asarray(truth, dtype=np.int64)
    prediction = np.asarray(prediction, dtype=np.int64)
    count = len(classes)
    confusion = np.zeros((count, count), dtype=np.int64)
    if len(truth):
        np.add.at(confusion, (truth, prediction), 1)
    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    true_positive = np.diag(confusion)
    precision = np.divide(true_positive, predicted, out=np.zeros(count), where=predicted > 0)
    recall = np.divide(true_positive, support, out=np.zeros(count), where=support > 0)
    f1 = np.divide(2 * true_positive, support + predicted,
                   out=np.zeros(count), where=(support + predicted) > 0)
    observed = support > 0
    union = (support + predicted) > 0
    return {
        "n": int(len(truth)),
        "classes": list(classes),
        "support": {label: int(support[i]) for i, label in enumerate(classes)},
        "predicted": {label: int(predicted[i]) for i, label in enumerate(classes)},
        "per_class": {
            label: {
                "support": int(support[i]),
                "predicted": int(predicted[i]),
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
            }
            for i, label in enumerate(classes)
        },
        "accuracy": float(true_positive.sum() / len(truth)) if len(truth) else 0.0,
        "balanced_accuracy": float(recall[observed].mean()) if observed.any() else 0.0,
        "macro_f1": float(f1[union].mean()) if union.any() else 0.0,
        "macro_f1_observed": float(f1[observed].mean()) if observed.any() else 0.0,
        "macro_f1_all_classes": float(f1.mean()) if count else 0.0,
        "observed_classes": [label for label, keep in zip(classes, observed) if keep],
        "confusion_matrix": confusion.tolist(),
    }


def fit_probe(rows: list[dict], split: DomainSplit, vectors: np.ndarray,
              *, c_values=(0.1, 1.0, 10.0), seed: int = 0) -> dict:
    """Fit on train, select C on dev, then score test exactly once."""
    train_i, train_y = _encoded_split(rows, split, "train")
    dev_i, dev_y = _encoded_split(rows, split, "dev")
    test_i, test_y = _encoded_split(rows, split, "test")
    if not len(train_i) or not len(dev_i) or not len(test_i):
        return {"status": "insufficient_split_support",
                "coverage": split.coverage}
    candidates = []
    for c in c_values:
        model = LogisticRegression(C=c, class_weight="balanced", max_iter=2000,
                                   random_state=seed)
        model.fit(vectors[train_i], train_y)
        dev_prediction = model.predict(vectors[dev_i])
        dev_metrics = metric_values(dev_y, dev_prediction, split.classes)
        candidates.append({
            "C": float(c),
            "model": model,
            "dev_prediction": dev_prediction,
            "dev": dev_metrics,
        })
    chosen = max(candidates, key=lambda item: (
        item["dev"]["macro_f1"], item["dev"]["balanced_accuracy"], -item["C"]
    ))
    test_prediction = chosen["model"].predict(vectors[test_i])
    return {
        "status": "evaluable",
        "classes": list(split.classes),
        "selected_C": chosen["C"],
        "dev": {"coverage": split.coverage["dev"], **chosen["dev"]},
        "test": {"coverage": split.coverage["test"],
                 **metric_values(test_y, test_prediction, split.classes)},
        "dev_candidates": [
            {"C": item["C"], "macro_f1": item["dev"]["macro_f1"],
             "balanced_accuracy": item["dev"]["balanced_accuracy"],
             "accuracy": item["dev"]["accuracy"]}
            for item in candidates
        ],
        "test_prediction": test_prediction,
        "dev_prediction": chosen["dev_prediction"],
    }


def paired_bootstrap(truth: np.ndarray, predictions: dict[str, np.ndarray],
                     classes: tuple[str, ...], *, repetitions: int = 1000,
                     seed: int = 5523) -> dict:
    """Bootstrap item-level paired differences in macro F1 and accuracy."""
    names = list(predictions)
    rng = np.random.default_rng(seed)
    values = {name: [] for name in names}
    for _ in range(repetitions):
        sample = rng.integers(len(truth), size=len(truth))
        for name in names:
            values[name].append(metric_values(truth[sample], predictions[name][sample], classes))
    result = {}
    for name in names:
        macro = np.asarray([row["macro_f1"] for row in values[name]])
        accuracy = np.asarray([row["accuracy"] for row in values[name]])
        result[name] = {
            "macro_f1_ci": _interval(macro),
            "accuracy_ci": _interval(accuracy),
        }
    reference = "siglip2" if "siglip2" in predictions else names[0]
    for name in names:
        macro = np.asarray([row["macro_f1"] for row in values[name]])
        reference_macro = np.asarray([row["macro_f1"] for row in values[reference]])
        accuracy = np.asarray([row["accuracy"] for row in values[name]])
        reference_accuracy = np.asarray([row["accuracy"] for row in values[reference]])
        result[name]["vs_siglip2"] = {
            "macro_f1_delta": _interval(macro - reference_macro),
            "accuracy_delta": _interval(accuracy - reference_accuracy),
        }
    return result


def _interval(values: np.ndarray) -> dict:
    return {
        "mean": float(values.mean()),
        "lower": float(np.quantile(values, 0.025)),
        "upper": float(np.quantile(values, 0.975)),
    }


def domain_metric_means(domain_results: dict) -> dict:
    """Summarise only evaluable domains; no domain receives item-level weight."""
    available = [result["test"] for result in domain_results.values()
                 if result.get("status") == "evaluable"]
    if not available:
        return {"evaluable_domains": 0, "macro_f1": 0.0,
                "balanced_accuracy": 0.0, "accuracy": 0.0}
    return {
        "evaluable_domains": len(available),
        "macro_f1": float(np.mean([row["macro_f1"] for row in available])),
        "balanced_accuracy": float(np.mean([row["balanced_accuracy"] for row in available])),
        "accuracy": float(np.mean([row["accuracy"] for row in available])),
        "evaluated_item_domain_predictions": int(sum(row["n"] for row in available)),
    }
