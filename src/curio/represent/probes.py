"""Weakly supervised linear probes on frozen image features; never ground-truth claims."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score

from curio.query.ontology import VOCAB
from curio.represent.validation import validate_object_splits

TASKS = ("material", "object_type")
C_VALUES = (0.1, 1.0, 10.0)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(data_path: Path, index: Path):
    items = [
        json.loads(line)
        for line in data_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not items or any(x.get("split") not in {"train", "dev", "test"} for x in items):
        raise ValueError("Every catalog item must have an explicit train/dev/test split")
    validate_object_splits(items)
    if set(x["split"] for x in items) != {"train", "dev", "test"}:
        raise ValueError("All three splits must be present")
    ids = json.loads((index / "item_ids.json").read_text(encoding="utf-8"))
    manifest = json.loads((index / "manifest.json").read_text(encoding="utf-8"))
    vectors = np.load(index / "image_embeddings.npy", allow_pickle=False)
    if (
        not isinstance(ids, list)
        or not all(isinstance(x, str) for x in ids)
        or len(ids) != len(set(ids))
    ):
        raise ValueError("Index IDs must be unique strings")
    if (
        vectors.ndim != 2
        or vectors.shape[0] != len(ids)
        or not np.issubdtype(vectors.dtype, np.number)
        or not np.isfinite(vectors).all()
    ):
        raise ValueError("Invalid index embedding matrix")
    if set(ids) != {str(x["item_id"]) for x in items}:
        raise ValueError("Index must cover exactly the catalog IDs")
    index_sha = hashlib.sha256(vectors.tobytes() + json.dumps(ids).encode()).hexdigest()
    if manifest.get("data_sha256") != _sha(data_path) or manifest.get("index_sha256") != index_sha:
        raise ValueError("Data or index hash differs from index manifest")
    if manifest.get("items") != len(ids) or manifest.get("dimension") != vectors.shape[1]:
        raise ValueError("Index manifest dimensions do not match embeddings")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("Index contains zero image features")
    # Match SearchService's normalization even for manually produced old indexes.
    vectors = vectors / norms
    by_id = {str(item["item_id"]): item for item in items}
    return [by_id[id_] for id_ in ids], vectors, manifest, index_sha


def _label(item, task):
    value = item.get(task)
    if not isinstance(value, str):
        return None
    value = value.strip().lower()
    return value if value in VOCAB[task] else None


def _metrics(y, predictions, classes):
    return {
        "support": len(y),
        "support_by_class": dict(sorted(Counter(y).items())),
        "accuracy": float(accuracy_score(y, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
        "macro_f1": float(
            f1_score(y, predictions, labels=classes, average="macro", zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(y, predictions, labels=classes).tolist(),
    }


def train_probes(data_path: str | Path, index: str | Path, output: str | Path, min_train: int = 10,
                 defer_test: bool = False):
    if min_train < 10:
        raise ValueError("min_train must be at least 10")
    data_path, index, output = Path(data_path), Path(index), Path(output)
    items, vectors, manifest, index_sha = _load(data_path, index)
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_sha256": _sha(data_path),
        "index_sha256": index_sha,
        "index_manifest_sha256": _sha(index / "manifest.json"),
        "index_version": manifest.get("index_version"),
        "model_version": manifest.get("model_version"),
        "feature_dimension": vectors.shape[1],
        "feature_preprocessing": "L2 normalize image features, matching SearchService",
        "min_train_per_class": min_train,
        "provenance": "Frozen image embeddings; metadata-derived labels are weak supervision, not visual ground truth.",
        "selection": "C chosen by dev macro F1; test deferred to frozen-head evaluation" if defer_test else "C chosen by dev macro F1; test evaluated once after selection",
        "test_deferred": defer_test,
        "tasks": {},
    }
    ontology = {task: sorted(VOCAB[task]) for task in TASKS}
    for task in TASKS:
        labels = [_label(x, task) for x in items]
        counts = Counter(
            labels[i] for i, x in enumerate(items) if x["split"] == "train" and labels[i]
        )
        classes = sorted(label for label, count in counts.items() if count >= min_train)
        indices = {
            split: [i for i, x in enumerate(items) if x["split"] == split and labels[i] in classes]
            for split in ("train", "dev", "test")
        }
        exclusions = {
            split: sum(x["split"] == split for x in items) - len(indices[split])
            for split in indices
        }
        split_totals = {split: sum(x["split"] == split for x in items) for split in indices}
        coverage = {
            split: (len(indices[split]) / split_totals[split] if split_totals[split] else 0.0)
            for split in indices
        }
        if len(classes) < 2 or not indices["dev"] or (not defer_test and not indices["test"]):
            report["tasks"][task] = {
                "status": "skipped",
                "reason": "Need at least two eligible train classes and labeled dev/test examples",
                "classes": classes,
                "train_class_counts": dict(counts),
                "excluded": exclusions,
                "coverage": coverage,
            }
            continue
        train, dev, test = (indices[k] for k in ("train", "dev", "test"))
        y_train = [labels[i] for i in train]
        y_dev = [labels[i] for i in dev]
        candidates = []
        for c in C_VALUES:
            model = LogisticRegression(C=c, class_weight="balanced", max_iter=1000, random_state=0)
            model.fit(vectors[train], y_train)
            candidates.append(
                (
                    float(
                        f1_score(
                            y_dev,
                            model.predict(vectors[dev]),
                            labels=classes,
                            average="macro",
                            zero_division=0,
                        )
                    ),
                    c,
                    model,
                )
            )
        best_score, best_c, model = max(candidates, key=lambda x: (x[0], -x[1]))
        weights, bias = model.coef_, model.intercept_
        if len(model.classes_) == 2:  # sklearn's binary logit is sigmoid(z)
            weights = np.vstack([np.zeros_like(weights[0]), weights[0]])
            bias = np.array([0.0, bias[0]])
        np.savez_compressed(
            output / f"{task}.npz",
            weights=weights,
            bias=bias,
            classes=np.asarray(model.classes_, dtype=str),
        )
        report["tasks"][task] = {
            "status": "trained",
            "classes": classes,
            "train_class_counts": dict(counts),
            "train_support": len(train),
            "dev_support": len(dev),
            "excluded": exclusions,
            "coverage": coverage,
            "test_eligible_support": len(test),
            "test_excluded_support": exclusions["test"],
            "selected_C": best_c,
            "dev_macro_f1": best_score,
            "dev_candidates": [{"C": c, "macro_f1": score} for score, c, _ in candidates],
            "test": None if defer_test else _metrics(
                [labels[i] for i in test], model.predict(vectors[test]), classes
            ),
        }
    (output / "ontology.json").write_text(
        json.dumps(ontology, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    artifact_paths = [output / "ontology.json"] + [
        output / f"{task}.npz"
        for task in TASKS
        if report["tasks"].get(task, {}).get("status") == "trained"
    ]
    report["artifacts_sha256"] = {
        path.name: _sha(path) for path in artifact_paths if path.is_file()
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


class LinearProbes:
    """Apply trained image-feature probes. Outputs are weak-supervision predictions."""

    def __init__(self, output_path: str | Path):
        self.models = {}
        path = Path(output_path)
        report = json.loads((path / "report.json").read_text(encoding="utf-8"))
        self.dimension = report["feature_dimension"]
        for name, expected in report.get("artifacts_sha256", {}).items():
            artifact = path / name
            if not artifact.is_file() or _sha(artifact) != expected:
                raise ValueError(f"Probe artifact hash mismatch: {name}")
        for task, task_report in report["tasks"].items():
            if task not in VOCAB:
                raise ValueError(f"Unknown probe task: {task}")
            if report["tasks"].get(task, {}).get("status") != "trained":
                continue
            with np.load(path / f"{task}.npz", allow_pickle=False) as model:
                weights, bias, classes = model["weights"], model["bias"], model["classes"]
                if weights.shape != (len(classes), self.dimension) or bias.shape != (len(classes),):
                    raise ValueError(f"Invalid {task} model dimensions")
                activation = task_report.get("activation", "softmax")
                if activation not in {"softmax", "sigmoid"}:
                    raise ValueError(f"Unknown probe activation: {activation}")
                self.models[task] = (weights.copy(), bias.copy(), classes.tolist(), activation)

    def predict(self, embeddings):
        x = np.asarray(embeddings)
        if x.ndim != 2 or x.shape[1] != self.dimension or not np.isfinite(x).all():
            raise ValueError("Expected finite 2D image embeddings with the trained dimension")
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        if np.any(norms <= 0):
            raise ValueError("Zero image features are invalid")
        x = x / norms
        result = {}
        for task, (weights, bias, classes, activation) in self.models.items():
            logits = x @ weights.T + bias
            if activation == "sigmoid":
                probabilities = 1 / (1 + np.exp(-np.clip(logits, -50, 50)))
            else:
                logits -= logits.max(axis=1, keepdims=True)
                exp = np.exp(logits)
                probabilities = exp / exp.sum(axis=1, keepdims=True)
            result[task] = {
                "classes": classes,
                "probabilities": probabilities,
            }
        return result


def evaluate_test(data_path, index, output, annotations_path=None):
    """Predict held-out labels with immutable saved heads; never fit or select C."""
    data_path, index, output = map(Path, (data_path, index, output))
    items, vectors, manifest, index_sha = _load(data_path, index)
    data_sha, manifest_sha = _sha(data_path), _sha(index / "manifest.json")
    report_path = output / "report.json"
    report_sha = _sha(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (report.get("data_sha256") != data_sha
            or report.get("index_sha256") != index_sha
            or report.get("index_manifest_sha256") != manifest_sha):
        raise ValueError("Frozen probes do not match test data/index")
    annotations = None
    annotation_sha = None
    if report.get("detail_annotations_sha256"):
        from .detail_probes import load_annotations

        if not annotations_path:
            raise ValueError("Detail test evaluation requires --annotations")
        annotation_sha = _sha(Path(annotations_path))
        if annotation_sha != report["detail_annotations_sha256"]:
            raise ValueError("Frozen detail annotation hash mismatch")
        annotations = load_annotations(annotations_path, items)
    artifact_hashes = {path.name: _sha(path) for path in output.glob("*.npz")}
    for task, task_report in report["tasks"].items():
        if task_report.get("status") == "trained" and report.get("artifacts_sha256", {}).get(
            f"{task}.npz"
        ) != artifact_hashes.get(f"{task}.npz"):
            raise ValueError(f"Frozen probe artifact hash mismatch: {task}.npz")
    heads = LinearProbes(output)
    test_indices = [i for i, item in enumerate(items) if item["split"] == "test"]
    predictions = heads.predict(vectors[test_indices]) if test_indices else {}
    results = {}
    for task, task_report in report["tasks"].items():
        if task_report.get("status") != "trained":
            continue
        classes = task_report["classes"]
        probabilities = predictions.get(task, {}).get("probabilities")
        if task_report.get("activation", "softmax") == "sigmoid":
            if annotations is None:
                raise ValueError("Sigmoid detail heads require frozen visual annotations")
            metrics = {}
            for column, label in enumerate(classes):
                known = [j for j, i in enumerate(test_indices)
                         if annotations.get(items[i]["item_id"], {}).get(task, {}).get(label) is not None]
                y = np.array([annotations[items[test_indices[j]]["item_id"]][task][label] for j in known])
                p = probabilities[known, column] if known else np.array([])
                metrics[label] = {
                    "support": len(y), "positives": int(y.sum()),
                    "f1": float(f1_score(y, p >= .5, zero_division=0)) if len(y) else None,
                    "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
                }
        else:
            labels = [_label(items[i], task) for i in test_indices]
            known = [j for j, label in enumerate(labels) if label in classes]
            metrics = _metrics(
                [labels[j] for j in known],
                [classes[column] for column in probabilities[known].argmax(axis=1)], classes
            ) if known else {"support": 0, "accuracy": None, "balanced_accuracy": None,
                             "macro_f1": None, "support_by_class": {}, "confusion_matrix": []}
        results[task] = {"test": metrics}
    if _sha(report_path) != report_sha or artifact_hashes != {
        path.name: _sha(path) for path in output.glob("*.npz")
    }:
        raise ValueError("Frozen probe files changed during test evaluation")
    if (_sha(data_path) != data_sha or _sha(index / "manifest.json") != manifest_sha
            or (annotations_path and annotation_sha and _sha(Path(annotations_path)) != annotation_sha)):
        raise ValueError("Frozen test inputs changed during evaluation")
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evaluation": "Frozen NPZ prediction only; no fit or hyperparameter selection",
        "data_sha256": data_sha, "index_sha256": index_sha,
        "index_manifest_sha256": manifest_sha,
        "index_version": manifest.get("index_version"),
        "training_report_sha256": report_sha, "artifacts_sha256": artifact_hashes,
        "detail_annotations_sha256": annotation_sha, "tasks": results,
    }
    (output / "test_report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/snapshots/collection_v2.jsonl")
    parser.add_argument("--index", default="artifacts/siglip")
    parser.add_argument("--output", default="artifacts/probes/siglip")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--defer-test", action="store_true")
    mode.add_argument("--evaluate-test", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(evaluate_test(args.data, args.index, args.output) if args.evaluate_test else
                   train_probes(args.data, args.index, args.output, defer_test=args.defer_test),
                   ensure_ascii=False, indent=2)
    )


if __name__ == "__main__":
    main()
