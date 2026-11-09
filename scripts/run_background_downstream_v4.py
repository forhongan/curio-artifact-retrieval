"""Fixed segmentation comparison and catalogue-grounded downstream tasks.

Segmentation quality is selected only on the 24-image dev subset.  Classifier
hyperparameters are selected on catalogue train/dev.  Test labels are loaded
for the final fixed evaluation only.  The six image inputs are frozen: whole
image, legacy GrabCut, border connected components, v4, and two high-resolution
rectangular GrabCut variants.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from curio.evaluate.catalogue_tasks import single_label  # noqa: E402
from curio.represent.background_benchmark import (  # noqa: E402
    FEATURE_CHANNELS,
    METHODS,
    benchmark_image,
    masks_for_image,
)
from eval_background_v4 import metrics as mask_metrics  # noqa: E402
from explore_background_variants import stage_split  # noqa: E402

LABELS = ROOT / "artifacts/cv_v4_20261108/catalogue_v3_primary_plus_candidates/labels_catalogue_v2.jsonl"
TRUTH = ROOT / "eval/background_truth_20261108"
OUT = ROOT / "artifacts/background_downstream_338_v4_20261108"
CACHE = OUT / "features.npz"
MIN_TRAIN = 5
C_VALUES = (.1, 1.0, 10.0)
DOMAINS = ("form", "broad_material", "fine_material")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def binary_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict:
    tp = int(np.logical_and(truth, prediction).sum())
    fp = int(np.logical_and(~truth, prediction).sum())
    fn = int(np.logical_and(truth, ~prediction).sum())
    tn = int(np.logical_and(~truth, ~prediction).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / max(1e-12, precision + recall),
            "balanced_accuracy": .5 * (tp / max(1, tp + fn) + tn / max(1, tn + fp))}


def process_row(row: dict) -> tuple[str, dict]:
    start = time.perf_counter()
    result = benchmark_image(ROOT / row["image_path"])
    return row["item_id"], {**result, "seconds": time.perf_counter() - start}


def prepare(rows: list[dict], workers: int) -> dict:
    if CACHE.is_file():
        with np.load(CACHE, allow_pickle=False) as saved:
            metadata = json.loads(str(saved["metadata"]))
            complete = all(f"{method}__{channel}" in saved.files
                           for method in METHODS for channel in FEATURE_CHANNELS)
        if (complete and metadata["labels_sha256"] == sha256(LABELS)
                and metadata["rows"] == [row["item_id"] for row in rows]):
            return metadata
    tasks = [(row["item_id"], row["image_path"], row["image_sha256"]) for row in rows]
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        processed = list(pool.map(_process_task, tasks, chunksize=4))
    arrays = {}
    diagnostics = {}
    coverage = {name: [] for name in METHODS}
    for item_id, result in processed:
        diagnostics[item_id] = result["diagnostics"]
        for method in METHODS:
            coverage[method].append(result["coverage"][method])
            for channel in FEATURE_CHANNELS:
                arrays[(method, channel)] = arrays.get((method, channel), [])
                arrays[(method, channel)].append(result["descriptors"][method][channel])
    saved_arrays = {f"{method}__{channel}": np.stack(values)
                    for (method, channel), values in arrays.items()}
    metadata = {
        "labels_sha256": sha256(LABELS), "rows": [row["item_id"] for row in rows],
        "split": [row["split"] for row in rows], "prepared_seconds": time.perf_counter() - start,
        "methods": list(METHODS), "channels": list(FEATURE_CHANNELS),
        "coverage": {name: sum(values) / len(values) for name, values in coverage.items()},
        "diagnostics": diagnostics, "timing_note": "whole extraction of all six masks and channels",
    }
    np.savez_compressed(CACHE, **saved_arrays, metadata=np.array(json.dumps(metadata)))
    return metadata


def _process_task(task: tuple[str, str, str]) -> tuple[str, dict]:
    item_id, image_path, image_sha = task
    cv2.setNumThreads(1)
    if sha256(ROOT / image_path) != image_sha:
        raise ValueError(f"Image hash changed for {item_id}")
    return process_row({"item_id": item_id, "image_path": image_path})


def class_selection(rows: list[dict], domains: list[str],
                    min_train: int = MIN_TRAIN) -> dict[str, list[str]]:
    selected = {}
    for domain in domains:
        counts = {}
        for row in rows:
            value = single_label(row, domain)
            if row["split"] == "train" and value is not None:
                counts[value] = counts.get(value, 0) + 1
        selected[domain] = sorted(label for label, count in counts.items() if count >= min_train)
    return selected


def feature_matrix(features: dict, split: str, method: str, channel: str,
                   classes: list[str], domain: str, rows: list[dict]) -> tuple[np.ndarray, np.ndarray, list[int]]:
    indices = []
    for i, row in enumerate(rows):
        if row["split"] != split:
            continue
        label = single_label(row, domain)
        if label in classes:
            indices.append(i)
    values = features[f"{method}__{channel}"][indices].astype(np.float64)
    truth = np.array([single_label(rows[i], domain) for i in indices], dtype=object)
    return values, truth, indices


def choose_classifier(train_x, train_y, dev_x, dev_y) -> tuple[object, dict]:
    options = []
    for c in C_VALUES:
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=c, class_weight="balanced", max_iter=2500,
                               random_state=5523, solver="lbfgs"),
        )
        model.fit(train_x, train_y)
        prediction = model.predict(dev_x)
        macro = float(f1_score(dev_y, prediction, average="macro", zero_division=0))
        from sklearn.metrics import balanced_accuracy_score
        balanced = float(balanced_accuracy_score(dev_y, prediction))
        options.append({"C": c, "dev_macro_f1": macro, "dev_balanced_accuracy": balanced,
                        "model": model})
    best = max(options, key=lambda row: (row["dev_macro_f1"], row["dev_balanced_accuracy"], -row["C"]))
    return best["model"], {"selected_C": best["C"], "dev_macro_f1": best["dev_macro_f1"],
                           "dev_balanced_accuracy": best["dev_balanced_accuracy"],
                           "candidates": [{k: v for k, v in row.items() if k != "model"}
                                          for row in options]}


def evaluate_domain(rows, features, domain, classes) -> dict:
    result = {"classes": classes, "results": {}}
    for method in METHODS:
        result["results"][method] = {}
        for channel in FEATURE_CHANNELS:
            x_train, y_train, _ = feature_matrix(features, "train", method, channel, classes, domain, rows)
            x_dev, y_dev, _ = feature_matrix(features, "dev", method, channel, classes, domain, rows)
            x_test, y_test, _ = feature_matrix(features, "test", method, channel, classes, domain, rows)
            if len(set(y_train)) < 2 or not len(y_dev) or not len(y_test):
                result["results"][method][channel] = {"status": "insufficient_support"}
                continue
            model, selection = choose_classifier(x_train, y_train, x_dev, y_dev)
            test_predict = model.predict(x_test)
            from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
            all_labels = sorted(set(y_test) | set(test_predict))
            result["results"][method][channel] = {
                "selected_C": selection["selected_C"],
                "dev": {"n": len(y_dev), "macro_f1": selection["dev_macro_f1"],
                        "balanced_accuracy": selection["dev_balanced_accuracy"]},
                "test": {
                    "n": len(y_test), "macro_f1": float(f1_score(y_test, test_predict,
                                                                   average="macro", labels=all_labels,
                                                                   zero_division=0)),
                    "accuracy": float(accuracy_score(y_test, test_predict)),
                    "balanced_accuracy": float(balanced_accuracy_score(y_test, test_predict)),
                    "labels": all_labels,
                    "confusion_matrix": confusion_matrix(y_test, test_predict, labels=all_labels).tolist(),
                },
                "selection": selection["candidates"],
            }
    return result


def binary_task(rows, features, domain, value) -> dict:
    result = {}
    for method in METHODS:
        result[method] = {}
        for channel in FEATURE_CHANNELS:
            train = [i for i, row in enumerate(rows) if row["split"] == "train"
                     and single_label(row, domain) is not None]
            dev = [i for i, row in enumerate(rows) if row["split"] == "dev"
                   and single_label(row, domain) is not None]
            test = [i for i, row in enumerate(rows) if row["split"] == "test"
                    and single_label(row, domain) is not None]
            y_train = np.array([single_label(rows[i], domain) == value for i in train])
            y_dev = np.array([single_label(rows[i], domain) == value for i in dev])
            y_test = np.array([single_label(rows[i], domain) == value for i in test])
            if min(y_train.sum(), (~y_train).sum(), y_dev.sum(), (~y_dev).sum()) < 5:
                result[method][channel] = {"status": "insufficient_support"}
                continue
            x_train = features[f"{method}__{channel}"][train]
            x_dev = features[f"{method}__{channel}"][dev]
            x_test = features[f"{method}__{channel}"][test]
            model, selection = choose_classifier(x_train, y_train, x_dev, y_dev)
            result[method][channel] = {"selected_C": selection["selected_C"],
                                       "dev": binary_metrics(y_dev, model.predict(x_dev)),
                                       "test": binary_metrics(y_test, model.predict(x_test))}
    return result


def quality_metrics() -> dict:
    records = json.loads((TRUTH / "records.json").read_text(encoding="utf-8"))
    per_image = []
    for row in records:
        _, masks = masks_for_image(ROOT / row["image_path"])
        with Image.open(TRUTH / f"{row['item_id']}.png") as opened:
            truth = np.asarray(opened, bool)
        for method in METHODS:
            mask, diagnostics = masks[method]
            if mask.shape != truth.shape:
                mask = cv2.resize(mask.astype(np.uint8), truth.shape[::-1],
                                  interpolation=cv2.INTER_NEAREST).astype(bool)
            scored = mask_metrics(mask, truth)
            denominator = scored["precision"] + scored["recall"]
            per_image.append({"item_id": row["item_id"], "method": method,
                              "stage_split": stage_split(row["item_id"]),
                              "truth_type": row["truth_type"],
                              "valid": diagnostics["valid"],
                              "seconds": diagnostics["seconds"],
                              "predicted_area": float(mask.mean()),
                              "truth_area": float(truth.mean()),
                              **scored,
                              "Dice": (2 * scored["precision"] * scored["recall"] / denominator
                                       if denominator else 0.0)})
    output = {}
    for method in METHODS:
        output[method] = {}
        for split in ("dev", "test", "all"):
            current = [row for row in per_image if row["method"] == method
                       and (split == "all" or row["stage_split"] == split)]
            output[method][split] = {
                "n": len(current), "coverage": sum(row["valid"] for row in current) / len(current),
                "mean_iou": float(np.mean([row["IoU"] for row in current])),
                "mean_dice": float(np.mean([row["Dice"] for row in current])),
                "mean_boundary_f1": float(np.mean([row["boundary_f1"] for row in current])),
                "mean_precision": float(np.mean([row["precision"] for row in current])),
                "mean_recall": float(np.mean([row["recall"] for row in current])),
                "mean_seconds": float(np.mean([row["seconds"] for row in current])),
            }
    return {"source": str(TRUTH.relative_to(ROOT)), "methods": output,
            "per_image": per_image,
            "truth_note": "24 approximate masks drafted by the primary agent for method diagnosis; not human gold"}


def contact_sheet(quality: dict) -> None:
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 14)
    truth_rows = json.loads((TRUTH / "records.json").read_text(encoding="utf-8"))
    qa = Image.new("RGB", (1600, 1140), "white")
    qdraw = ImageDraw.Draw(qa)
    qdraw.text((8, 6), "rows: truth / old GrabCut / border CC / v4 / rect384 raw / rect384 largest",
               fill="black", font=font)
    for column, row in enumerate(truth_rows[:8]):
        item_id = row["item_id"]
        with Image.open(TRUTH / f"{item_id}.png") as loaded:
            tiles = [loaded.convert("RGB")]
        _, masks = masks_for_image(ROOT / row["image_path"])
        for method in METHODS[1:]:
            tile = Image.fromarray(masks[method][0].astype(np.uint8) * 255).convert("RGB")
            tiles.append(tile)
        for row_index, tile in enumerate(tiles):
            tile = tile.copy()
            tile.thumbnail((170, 150))
            qa.paste(tile, (column * 200 + 8, 32 + row_index * 180))
        qdraw.text((column * 200 + 8, 1120), item_id, fill="black", font=font)
    qa.save(OUT / "qa_mask_comparison.jpg", quality=92)
    by_id = {row["item_id"]: row for row in truth_rows}
    worst = sorted((row for row in quality["per_image"] if row["method"] == "rect_gc384_raw"),
                   key=lambda row: row["IoU"])[:6]
    failure = Image.new("RGB", (1200, 880), "white")
    fdraw = ImageDraw.Draw(failure)
    fdraw.text((8, 6), "Low-IoU examples: original / approximate truth / rectangle / v4 / border",
               fill="black", font=font)
    for column, scored in enumerate(worst):
        row = by_id[scored["item_id"]]
        with Image.open(ROOT / row["image_path"]) as opened:
            tiles = [opened.convert("RGB")]
        with Image.open(TRUTH / f"{row['item_id']}.png") as opened:
            tiles.append(opened.convert("RGB"))
        _, masks = masks_for_image(ROOT / row["image_path"])
        for method in ("rect_gc384_raw", "v4", "border_cc"):
            tiles.append(Image.fromarray(masks[method][0].astype(np.uint8) * 255).convert("RGB"))
        for index, tile in enumerate(tiles):
            tile.thumbnail((180, 150))
            failure.paste(tile, (column * 200 + 8, 30 + index * 165))
        fdraw.text((column * 200 + 8, 850), f"{row['item_id']} IoU={scored['IoU']:.2f}",
                   fill="black", font=font)
    failure.save(OUT / "failure_examples.jpg", quality=92)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    records = json.loads((ROOT / "artifacts/bg_catalogue_eval_20261108/records.json").read_text(encoding="utf-8"))["records"]
    allowed_ids = {row["item_id"] for row in records}
    rows = [row for row in read_jsonl(LABELS) if row["item_id"] in allowed_ids]
    metadata = prepare(rows, max(1, args.workers))
    with np.load(CACHE, allow_pickle=False) as loaded:
        features = {key: loaded[key] for key in loaded.files if key != "metadata"}
    class_map = class_selection(rows, list(DOMAINS))
    results = {
        "version": "background-downstream-v4",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "methods": list(METHODS), "channels": list(FEATURE_CHANNELS),
            "classifier": "StandardScaler fitted on train + LogisticRegression lbfgs; C selected only on dev macro F1",
            "material_supervision": "single-label catalogue facets; missing or ambiguous values unknown",
            "gallery_scope": "338 fixed labelled images; train=218, dev=60, test=60; not the full 4,692-image catalogue",
            "test_disclosure": "test split was used in earlier catalogue studies; this is a fixed supplementary evaluation",
        },
        "provenance": {"labels_sha256": sha256(LABELS), "features_sha256": sha256(CACHE),
                       "coverage": metadata["coverage"], "prepared_seconds": metadata["prepared_seconds"]},
        "segmentation_quality": quality_metrics(),
        "domains": {},
        "visual_tasks": {
            "color_blue": binary_task(rows, features, "color", "blue"),
            "texture_motif_floral": binary_task(rows, features, "motif", "floral"),
            # Technique is retained as a negative/brittle result unless there is
            # enough known support.  No unknown catalogue field becomes negative.
            "texture_technique_carved": binary_task(rows, features, "technique", "carved"),
        },
    }
    for domain in DOMAINS:
        results["domains"][domain] = evaluate_domain(
            rows, features, domain, class_map[domain])
    dump(OUT / "report.json", results)
    (OUT / "feature_schema.json").write_text(json.dumps({
        name: int(features[f"{method}__{name}"].shape[1])
        for method in METHODS for name in FEATURE_CHANNELS
    }, indent=2), encoding="utf-8")
    contact_sheet(results["segmentation_quality"])
    important = {}
    for domain, domain_report in results["domains"].items():
        important[domain] = {}
        for method in METHODS:
            important[domain][method] = {
                channel: row.get("test", {}).get("macro_f1")
                for channel, row in domain_report["results"][method].items()
            }
    dump(OUT / "summary.json", {"domains": important,
                                "segmentation": results["segmentation_quality"],
                                "hash": sha256(OUT / "report.json")})
    print(json.dumps(important, indent=2))


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
