"""Image-only multi-domain probes using official catalogue labels.

The probe consumes pixels through frozen image embeddings.  Catalogue fields
provide supervision and evaluation labels but are never read at prediction or
ranking time.  Multi-label output probabilities are independently calibrated
from a dev-selected decision threshold; they are not probabilities that an
object is authentic.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from curio.represent.validation import validate_object_splits

VERSION = "curio-cv-v4-official-probes-v1"
C_VALUES = (0.1, 1.0, 10.0)
DOMAINS = (
    "object_type",
    "broad_material",
    "material",
    "motifs",
    "colors",
    "era",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load(data_path: Path, labels_path: Path, index: Path):
    labels = _read_jsonl(labels_path)
    catalog = _read_jsonl(data_path)
    catalog_by_id = {row["item_id"]: row for row in catalog}
    if not labels or len(catalog) != len(labels):
        raise ValueError("The v4 label table and catalogue must cover the same items")
    rows = []
    for label in labels:
        source = catalog_by_id.get(label["item_id"])
        if source is None:
            raise ValueError(f"Unknown item in v4 labels: {label['item_id']}")
        if source["split"] != label["split"] or source["object_group"] != label["object_group"]:
            raise ValueError(f"Split provenance mismatch for {label['item_id']}")
        rows.append({**source, "v4": label})
    validate_object_splits(rows)
    ids = json.loads((index / "item_ids.json").read_text(encoding="utf-8"))
    manifest = json.loads((index / "manifest.json").read_text(encoding="utf-8"))
    vectors = np.load(index / "image_embeddings.npy", allow_pickle=False)
    index_sha = hashlib.sha256(vectors.tobytes() + json.dumps(ids).encode()).hexdigest()
    if manifest.get("data_sha256") != _sha(data_path) or manifest.get("index_sha256") != index_sha:
        raise ValueError("Image index does not match the frozen catalogue")
    if set(ids) != set(catalog_by_id):
        raise ValueError("Image index and catalogue item IDs differ")
    by_id = {row["item_id"]: row for row in rows}
    ordered = [by_id[item_id] for item_id in ids]
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    return ordered, vectors, manifest, index_sha


def _values(label: dict, domain: str) -> list[str]:
    value = label["v4"]["derived_ontology"].get(domain)
    if isinstance(value, list):
        return sorted(set(str(x) for x in value if x))
    return [str(value)] if value else []


def _multi_probability(model, x):
    raw = model.predict_proba(x)
    return raw[:, list(model.classes_).index(1)]


def _binary_metrics(y, probability, threshold):
    pred = probability >= threshold
    return {
        "support": int(len(y)),
        "positives": int(y.sum()),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "auc": float(roc_auc_score(y, probability)) if len(np.unique(y)) == 2 else None,
        "threshold": float(threshold),
    }


def _choose_threshold(y, probability):
    candidates = (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75)
    scored = []
    for threshold in candidates:
        pred = probability >= threshold
        scored.append({
            "threshold": threshold,
            "f1": float(f1_score(y, pred, zero_division=0)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        })
    selected = max(scored, key=lambda row: (row["f1"], row["balanced_accuracy"], row["threshold"]))
    return selected, scored


def train_official_probes(data_path: str | Path, labels_path: str | Path, index: str | Path, output: str | Path):
    data_path, labels_path, index, output = map(Path, (data_path, labels_path, index, output))
    rows, vectors, manifest, index_sha = _load(data_path, labels_path, index)
    output.mkdir(parents=True, exist_ok=True)
    split_indices = {split: [i for i, row in enumerate(rows) if row["split"] == split] for split in ("train", "dev", "test")}
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "version": VERSION,
        "catalogue_sha256": _sha(data_path),
        "labels_sha256": _sha(labels_path),
        "index_sha256": index_sha,
        "index_manifest_sha256": _sha(index / "manifest.json"),
        "model_version": manifest.get("model_version"),
        "label_source": "official_catalogue_derived",
        "selection": "C and per-label threshold selected on dev only; test not read during fit or selection",
        "prediction_input": "frozen image embeddings only",
        "tasks": {},
    }
    artifacts = {}
    for domain in DOMAINS:
        train_labels = {i: set(_values(rows[i], domain)) for i in split_indices["train"]}
        counts = Counter(value for values in train_labels.values() for value in values)
        classes = sorted(value for value, count in counts.items() if count >= 20)
        task = {"classes": classes, "train_class_counts": {value: counts[value] for value in classes}, "models": {}}
        for value in classes:
            def target(i):
                return int(value in set(_values(rows[i], domain)))
            train = [i for i in split_indices["train"] if value in train_labels[i] or len(train_labels[i])]
            dev = [i for i in split_indices["dev"] if target(i) or len(_values(rows[i], domain))]
            test = [i for i in split_indices["test"] if target(i) or len(_values(rows[i], domain))]
            y_train = np.asarray([target(i) for i in train])
            y_dev = np.asarray([target(i) for i in dev])
            y_test = np.asarray([target(i) for i in test])
            if y_train.sum() < 20 or y_dev.sum() < 3 or len(y_dev) < 10:
                continue
            candidates = []
            for c in C_VALUES:
                model = LogisticRegression(C=c, class_weight="balanced", max_iter=1000, random_state=0)
                model.fit(vectors[train], y_train)
                dev_probability = _multi_probability(model, vectors[dev])
                threshold, threshold_candidates = _choose_threshold(y_dev, dev_probability)
                candidates.append({
                    "C": c,
                    "threshold": threshold["threshold"],
                    "dev_f1": threshold["f1"],
                    "model": model,
                    "dev_probability": dev_probability,
                    "threshold_candidates": threshold_candidates,
                })
            chosen = max(candidates, key=lambda row: (row["dev_f1"], -row["C"]))
            model = chosen["model"]
            # The coefficient vector is saved in a deterministic, compact artifact.
            slug = f"{domain}__{value}".replace(" ", "_").replace("/", "_")
            np.savez_compressed(
                output / f"{slug}.npz",
                weights=model.coef_.astype(np.float32),
                bias=model.intercept_.astype(np.float32),
                classes=np.asarray([0, 1], dtype=np.int8),
                threshold=np.asarray([chosen["threshold"]], dtype=np.float32),
            )
            artifacts[f"{slug}.npz"] = _sha(output / f"{slug}.npz")
            dev_metrics = _binary_metrics(y_dev, chosen["dev_probability"], chosen["threshold"])
            test_probability = _multi_probability(model, vectors[test]) if test else np.array([])
            test_metrics = _binary_metrics(y_test, test_probability, chosen["threshold"]) if test else None
            task["models"][value] = {
                "artifact": f"{slug}.npz",
                "selected_C": chosen["C"],
                "selected_threshold": chosen["threshold"],
                "dev": dev_metrics,
                "dev_candidates": [
                    {
                        "C": c["C"],
                        "dev_f1": c["dev_f1"],
                        "threshold": c["threshold"],
                    }
                    for c in candidates
                ],
                "test": test_metrics,
                "eligible": {"train": len(train), "dev": len(dev), "test": len(test)},
            }
        report["tasks"][domain] = task
    report["artifacts_sha256"] = artifacts
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/snapshots/collection_v2.jsonl")
    parser.add_argument("--labels", default="artifacts/cv_v4_20261108/official/labels.jsonl")
    parser.add_argument("--index", default="artifacts/siglip")
    parser.add_argument("--output", default="artifacts/cv_v4_20261108/probes_siglip")
    args = parser.parse_args()
    report = train_official_probes(args.data, args.labels, args.index, args.output)
    print(json.dumps({domain: {"classes": len(task["classes"])} for domain, task in report["tasks"].items()}, indent=2))
