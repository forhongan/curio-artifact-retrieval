"""Train official-label probes and evaluate explicit catalogue constraint sets.

This complements open-text rankings. It reports separate form/material/era
accuracy and a tri-state conjunction boundary whose absent catalogue facts stay
unknown rather than becoming automatic negatives.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.neighbors import KNeighborsClassifier

from curio.evaluate.catalogue_tasks import TASKS, query_text, single_label, task_manifest
from curio.evaluate.set_boundary import set_metrics
from curio.represent.encoders import Encoder

ROOT = Path(__file__).resolve().parents[1]
CS = (0.1, 1.0, 10.0)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(y, pred):
    # Macro averages must include a predicted-only class.  Omitting it can hide
    # a severe false-positive class when the modal test class happens to match.
    labels = sorted(set(y) | set(pred))
    # Support of zero truth examples is intentional: this makes a predicted-only
    # class visible and assigns it zero precision rather than silently skipping it.
    supports = {label: y.count(label) for label in labels}
    return {
        "n": len(y), "known_coverage": len(y) / max(1, len(pred)),
        "truth_labels": sorted(set(y)), "predicted_labels": sorted(set(pred)),
        "labels_scored": labels,
        "support": supports,
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", labels=labels, zero_division=0)),
        "per_class_f1": {
            label: float(f1_score(y, pred, labels=[label], average="macro", zero_division=0))
            for label in labels
        },
        "labels": labels,
        "confusion_matrix": confusion_matrix(y, pred, labels=labels).tolist(),
    }


def load_rows(labels_path):
    rows = read_jsonl(labels_path)
    index = ROOT / "artifacts/siglip"
    ids = json.loads((index / "item_ids.json").read_text(encoding="utf-8"))
    vectors = np.load(index / "image_embeddings.npy", allow_pickle=False)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    by_id = {row["item_id"]: row for row in rows}
    return [by_id[item_id] for item_id in ids], vectors


def subset(rows, vectors, split, classes, labels):
    idx = [i for i, row in enumerate(rows) if row["split"] == split and labels[i] in classes]
    truth = [labels[i] for i in idx]
    return idx, vectors[idx], truth


def train_probe(train_x, train_y, dev_x, dev_y, test_x, test_y):
    candidates = []
    for c in CS:
        model = LogisticRegression(C=c, class_weight="balanced", max_iter=1500, random_state=0)
        model.fit(train_x, train_y)
        dev_pred = model.predict(dev_x)
        candidates.append({
            "C": c, "model": model, "dev": metrics(dev_y, dev_pred),
            "dev_pred": dev_pred,
        })
    chosen = max(candidates, key=lambda x: (x["dev"]["macro_f1"], x["dev"]["balanced_accuracy"], -x["C"]))
    return chosen, candidates, chosen["model"].predict(test_x)


def predictor_probability(model, x, value):
    idx = list(model.classes_).index(value)
    return model.predict_proba(x)[:, idx]


def constraint_facets(row):
    """Return only mutually exclusive labels that a constraint can adjudicate."""
    return {domain: single_label(row, domain) for domain in TASKS}


def constraint_truth(row, atoms):
    """Return true, false, or unknown for one catalogue-known conjunction.

    A generic material such as ``metal`` cannot decide whether an item is
    bronze.  Such incomplete evidence stays unknown instead of being counted as
    a positive or negative for a bronze constraint.
    """
    facets = constraint_facets(row)
    unknown = False
    for domain, value in atoms:
        label = facets.get(domain)
        if domain == "fine_material" and label is None:
            broad = facets.get("broad_material")
            if broad is not None and broad != "metal":
                return False
            unknown = True
            continue
        if label is None:
            unknown = True
            continue
        if label != value:
            return False
    return None if unknown else True


def constraint_rows(rows, atoms, split):
    result = []
    for index, row in enumerate(rows):
        if row["split"] != split:
            continue
        truth = constraint_truth(row, atoms)
        if truth is not None:
            result.append((index, row, truth))
    return result


def constraint_scores(rows, vectors, models, atoms, truth_rows):
    scored = []
    for index, row, truth in truth_rows:
        probabilities = []
        for domain, value in atoms:
            if domain not in models or value not in models[domain]["classes"]:
                continue
            probabilities.append(float(predictor_probability(
                models[domain]["lr"], vectors[index][None], value
            )[0]))
        if len(probabilities) != len(atoms):
            continue
        arrays = np.asarray(probabilities, dtype=np.float64)
        scores = {
            "product": float(np.prod(arrays)),
            "mean_min": float(np.mean([arrays.mean(), arrays.min()])),
        }
        scored.append({"item_id": row["item_id"], "truth": truth, "scores": scores})
    return scored


def freeze_constraint_threshold(dev_rows, thresholds):
    trials = []
    relevant = {row["item_id"] for row in dev_rows if row["truth"]}
    for method in ("product", "mean_min"):
        for threshold in thresholds:
            selected = {row["item_id"] for row in dev_rows if row["scores"][method] >= threshold}
            trials.append({"method": method, "threshold": float(threshold),
                           **set_metrics(selected, relevant, len(dev_rows))})
    eligible = [row for row in trials if row["true_positives"] or not row["relevant_size"]]
    if not eligible:
        return None, trials
    return max(eligible, key=lambda row: (row["f1"], row["precision"], row["recall"], -row["threshold"])), trials


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default="artifacts/cv_v4_20261108/catalogue_v2/labels_catalogue_v2.jsonl")
    parser.add_argument("--output", default="artifacts/cv_v4_20261108/catalogue_tasks")
    args = parser.parse_args()
    labels_path = ROOT / args.labels
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    rows, vectors = load_rows(labels_path)
    Encoder("siglip", device="auto")
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "version": "catalogue-tasks-v4",
        "labels_sha256": digest(labels_path),
        "index_sha256": digest(ROOT / "artifacts/siglip/image_embeddings.npy"),
        "label_policy": "official records only; missing and multi-value labels are unknown; train/dev/test are object-group separated; no per-item catalogue text is used at inference",
        "tasks": {},
        "constraints": {},
    }
    models = {}
    for domain in TASKS:
        manifest = task_manifest(rows, domain)
        labels = manifest.pop("labels")
        if manifest["status"] != "evaluable":
            report["tasks"][domain] = {**manifest, "methods": {}}
            print(domain, manifest["status"], flush=True)
            continue
        train_i, train_x, train_y = subset(rows, vectors, "train", manifest["classes"], labels)
        dev_i, dev_x, dev_y = subset(rows, vectors, "dev", manifest["classes"], labels)
        test_i, test_x, test_y = subset(rows, vectors, "test", manifest["classes"], labels)
        lr, candidates, test_pred = train_probe(train_x, train_y, dev_x, dev_y, test_x, test_y)
        knn_k = min(15, int(np.sqrt(len(train_y))))
        knn = KNeighborsClassifier(n_neighbors=knn_k, metric="cosine", weights="distance")
        knn.fit(train_x, train_y)
        models[domain] = {"lr": lr["model"], "knn": knn, "classes": manifest["classes"]}
        report["tasks"][domain] = {
            **manifest,
            "methods": {
                "siglip_logreg": {
                    "selected_C": lr["C"], "dev": lr["dev"],
                    "test": {**metrics(test_y, test_pred),
                             "item_ids": [rows[i]["item_id"] for i in test_i],
                             "truth": test_y,
                             "predicted": test_pred.tolist()},
                    "dev_candidates": [{k: v for k, v in c.items() if k not in {"model", "dev_pred"}} for c in candidates],
                },
                f"siglip_knn{knn_k}": {
                    "dev": metrics(dev_y, knn.predict(dev_x)),
                    "test": {**metrics(test_y, knn.predict(test_x)),
                             "item_ids": [rows[i]["item_id"] for i in test_i],
                             "truth": test_y,
                             "predicted": knn.predict(test_x).tolist()},
                },
            },
            "indices": {"train": train_i, "dev": dev_i, "test": test_i},
        }
        print(domain, {name: source["test"]["macro_f1"] for name, source in report["tasks"][domain]["methods"].items()}, flush=True)

    # Fixed constraints are evaluated on catalogue-supported test items. The
    # tri-state label is not a graded-visual qrel; unknown stays outside scoring.
    constraints = [
        ("porcelain_vase", [("form", "vase"), ("fine_material", "porcelain")]),
        ("porcelain_bowl", [("form", "bowl"), ("fine_material", "porcelain")]),
        ("jade_ornament", [("form", "ornament"), ("fine_material", "jade")]),
        ("bronze_mirror", [("form", "mirror"), ("fine_material", "bronze")]),
        ("dragon_design", [("motif", "dragon")]),
        ("porcelain_blue", [("fine_material", "porcelain"), ("color", "blue")]),
    ]
    for qid, atoms in constraints:
        dev_truth = constraint_rows(rows, atoms, "dev")
        test_truth = constraint_rows(rows, atoms, "test")
        dev_scores = constraint_scores(rows, vectors, models, atoms, dev_truth)
        test_scores = constraint_scores(rows, vectors, models, atoms, test_truth)
        if not any(row["truth"] for row in dev_scores) or not any(row["truth"] for row in test_scores):
            report["constraints"][qid] = {
                "atoms": atoms, "query": query_text(atoms),
                "dev_known_catalogue_items": len(dev_scores),
                "test_known_catalogue_items": len(test_scores),
                "dev_catalogue_positives": sum(row["truth"] for row in dev_scores),
                "test_catalogue_positives": sum(row["truth"] for row in test_scores),
                "status": "not_evaluable_requires_dev_and_test_positives",
            }
            continue
        thresholds = (0.01, 0.02, 0.04, 0.08, 0.12, 0.16, 0.24, 0.32, 0.40, 0.50, 0.65, 0.80)
        frozen, dev_trials = freeze_constraint_threshold(dev_scores, thresholds)
        if frozen is None:
            report["constraints"][qid] = {
                "atoms": atoms, "query": query_text(atoms),
                "dev_known_catalogue_items": len(dev_scores),
                "test_known_catalogue_items": len(test_scores),
                "status": "not_evaluable_no_dev_true_positive_at_candidate_thresholds",
            }
            continue
        method, threshold = frozen["method"], frozen["threshold"]
        selected = {row["item_id"] for row in test_scores if row["scores"][method] >= threshold}
        relevant = {row["item_id"] for row in test_scores if row["truth"]}
        report["constraints"][qid] = {
            "atoms": atoms, "query": query_text(atoms),
            "dev_known_catalogue_items": len(dev_scores),
            "test_known_catalogue_items": len(test_scores),
            "dev_catalogue_positives": sum(row["truth"] for row in dev_scores),
            "test_catalogue_positives": len(relevant),
            "selection": "one method and threshold frozen on catalogue-known dev; test evaluated once",
            "dev_selected": frozen,
            "dev_trials": dev_trials,
            "test": set_metrics(selected, relevant, len(test_scores)),
            "test_selected_ids": sorted(selected),
            "test_false_positive_ids": sorted(selected - relevant),
            "test_false_negative_ids": sorted(relevant - selected),
            "coverage_note": "Catalogue-unknown items are outside set metrics; missing catalogue evidence is never treated as a negative.",
        }
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", out / "report.json")


if __name__ == "__main__":
    main()
