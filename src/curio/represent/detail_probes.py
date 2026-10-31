"""Train image-visible multi-label heads from disclosed AI blind-image annotations."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

from curio.query.ontology import VOCAB
from .probes import C_VALUES, _load, _sha, evaluate_test

DETAIL_TASKS = ("colors", "shape", "motifs")


def load_annotations(path, items):
    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    catalog = {item["item_id"]: item for item in items}
    output = {}
    for row in rows:
        key = row["item_id"]
        if key not in catalog or key in output:
            raise ValueError(f"Unknown or duplicate annotated item: {key}")
        if row["split"] != catalog[key]["split"]:
            raise ValueError(f"Annotation/catalog split mismatch: {key}")
        if row.get("label_source") != "ai_image_only":
            raise ValueError("Detail labels must disclose ai_image_only provenance")
        if not isinstance(row.get("caption_en"), str) or len(row["caption_en"].strip()) < 8:
            raise ValueError(f"Missing image caption: {key}")
        for task in DETAIL_TASKS:
            labels = row.get(task, {})
            if not isinstance(labels, dict) or set(labels) - set(VOCAB[task]):
                raise ValueError(f"Invalid {task} labels: {key}")
            if any(
                value is not None and (type(value) is not int or value not in (0, 1))
                for value in labels.values()
            ):
                raise ValueError(f"Labels must be 0, 1, or null: {key}")
        if row.get("image_sha256") != catalog[key].get("image_sha256"):
            raise ValueError(f"Annotation/image hash mismatch: {key}")
        output[key] = row
    return output


def train(data, index, annotations_path, base_probes, output, min_positive=5, defer_test=False):
    data, index, output, base_probes = map(Path, (data, index, output, base_probes))
    items, vectors, manifest, index_sha = _load(data, index)
    annotations = load_annotations(annotations_path, items)
    base_report = json.loads((base_probes / "report.json").read_text(encoding="utf-8"))
    if base_report["data_sha256"] != _sha(data) or base_report["index_sha256"] != index_sha:
        raise ValueError("Base probes do not match image index")
    output.mkdir(parents=True, exist_ok=True)
    report = {
        **base_report,
        "detail_annotations_sha256": _sha(Path(annotations_path)),
        "detail_label_source": "ai_image_only",
        "detail_min_positive_negative": min_positive,
        "test_deferred": defer_test,
        "detail_annotation_counts": dict(Counter(row["split"] for row in annotations.values())),
        "detail_protocol": "AI image-only labels; unknown excluded per attribute. C selected by dev macro F1; test reported after selection. Sigmoid per attribute, fixed 0.5 threshold.",
    }
    report["tasks"] = dict(base_report["tasks"])
    if defer_test:
        report["tasks"] = {task: {**result, "test": None}
                           for task, result in report["tasks"].items()}
        report["detail_protocol"] = "AI image-only labels; unknown excluded per attribute. C selected by dev macro F1; test deferred to frozen-head evaluation. Sigmoid per attribute, fixed 0.5 threshold."
    for task in ("material", "object_type"):
        if report["tasks"].get(task, {}).get("status") == "trained":
            filename = f"{task}.npz"
            if base_report.get("artifacts_sha256", {}).get(filename) != _sha(base_probes / filename):
                raise ValueError(f"Base probe artifact hash mismatch: {filename}")
            shutil.copy2(base_probes / filename, output / filename)
    for task in DETAIL_TASKS:
        supported = []
        for label in sorted(VOCAB[task]):
            indices = {
                split: [
                    i
                    for i, item in enumerate(items)
                    if item["split"] == split
                    and item["item_id"] in annotations
                    and annotations[item["item_id"]].get(task, {}).get(label) is not None
                ]
                for split in ("train", "dev", "test")
            }
            values = {
                split: np.array([annotations[items[i]["item_id"]][task][label] for i in idx])
                for split, idx in indices.items()
            }
            counts = Counter(values["train"])
            if (
                min(counts.get(0, 0), counts.get(1, 0)) < min_positive
                or len(np.unique(values["dev"])) < 2
            ):
                continue
            supported.append((label, indices, values))
        if not supported:
            report["tasks"][task] = {
                "status": "skipped",
                "reason": "Insufficient positive/negative train and dev image labels",
                "classes": [],
            }
            continue
        candidates = []
        for c in C_VALUES:
            models, dev_scores = [], []
            for label, indices, values in supported:
                model = LogisticRegression(
                    C=c, class_weight="balanced", max_iter=1000, random_state=5523
                )
                model.fit(vectors[indices["train"]], values["train"])
                models.append(model)
                dev_scores.append(
                    f1_score(values["dev"], model.predict(vectors[indices["dev"]]), zero_division=0)
                )
            candidates.append((float(np.mean(dev_scores)), c, models))
        dev_f1, best_c, models = max(candidates, key=lambda row: (row[0], -row[1]))
        weights = np.concatenate([model.coef_ for model in models], axis=0)
        bias = np.concatenate([model.intercept_ for model in models])
        classes = [row[0] for row in supported]
        np.savez_compressed(
            output / f"{task}.npz", weights=weights, bias=bias, classes=np.array(classes, dtype=str)
        )
        test_metrics = {}
        for (label, indices, values), model in ([] if defer_test else zip(supported, models)):
            test_y = values["test"]
            if not len(test_y):
                test_metrics[label] = {"support": 0, "f1": None, "roc_auc": None}
                continue
            p = model.predict_proba(vectors[indices["test"]])[:, 1]
            test_metrics[label] = {
                "support": len(test_y),
                "positives": int(test_y.sum()),
                "f1": float(f1_score(test_y, p >= 0.5, zero_division=0)),
                "roc_auc": float(roc_auc_score(test_y, p)) if len(np.unique(test_y)) == 2 else None,
            }
        report["tasks"][task] = {
            "status": "trained",
            "activation": "sigmoid",
            "classes": classes,
            "label_source": "ai_image_only",
            "selected_C": best_c,
            "dev_macro_f1": dev_f1,
            "dev_candidates": [{"C": c, "macro_f1": score} for score, c, _ in candidates],
            "train_counts": {
                label: {str(k): int(v) for k, v in Counter(values["train"]).items()}
                for label, _, values in supported
            },
            "test": None if defer_test else test_metrics,
        }
    ontology = {task: sorted(VOCAB[task]) for task in report["tasks"]}
    (output / "ontology.json").write_text(
        json.dumps(ontology, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    paths = [output / "ontology.json"] + [
        output / f"{task}.npz"
        for task, result in report["tasks"].items()
        if result.get("status") == "trained"
    ]
    report["artifacts_sha256"] = {path.name: _sha(path) for path in paths}
    report["index_version"] = manifest["index_version"]
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/snapshots/collection_v2.jsonl")
    parser.add_argument("--index", default="artifacts/siglip")
    parser.add_argument("--annotations", default="eval/visual_labels/annotations.jsonl")
    parser.add_argument("--base-probes", default="artifacts/probes/siglip")
    parser.add_argument("--output", default="artifacts/probes/vision")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--defer-test", action="store_true")
    mode.add_argument("--evaluate-test", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            evaluate_test(args.data, args.index, args.output, args.annotations) if args.evaluate_test else
            train(args.data, args.index, args.annotations, args.base_probes, args.output,
                  defer_test=args.defer_test), indent=2
        )
    )


if __name__ == "__main__":
    main()
