"""Train/dev/test evaluation of mask-derived catalogue attributes.

The class labels come from the catalogue; masks and features use only pixels.
All candidates use the same train/dev/test items and split-only hyperparameter
selection. Test is loaded only after dev choices are frozen.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image, ImageOps
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits

from curio.represent.background_v4 import isolate_object
from curio.represent.classical_cv import extract_classical

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/bg_catalogue_eval_20261108"
MAX_COLOR_DIM = 112
MAX_SHAPE_DIM = 80


def rows(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf8").splitlines() if x.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def color_vector(image, mask):
    rgb = np.asarray(image)
    if mask.shape != rgb.shape[:2]:
        mask = cv2.resize(mask.astype(np.uint8), rgb.shape[1::-1], interpolation=cv2.INTER_NEAREST).astype(bool)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    values = hsv[mask]
    if not len(values):
        return np.zeros(88, np.float32)
    # Circular hue histogram plus saturated/value distribution.  Features are
    # fixed without looking at train/dev/test outcome labels.
    hue = np.histogram(values[:, 0], 24, (0, 180))[0]
    sat = np.histogram(values[:, 1], 16, (0, 256))[0]
    val = np.histogram(values[:, 2], 16, (0, 256))[0]
    channels = []
    for part in (hue, sat, val):
        part = part / max(1, part.sum())
        channels.extend((part, np.sqrt(part)))
    return np.asarray(np.concatenate(channels), np.float32)


def shape_vector(mask):
    preview = Image.fromarray(mask.astype(np.uint8))
    preview.thumbnail((128, 128))
    small = np.asarray(preview, bool)
    if not small.any():
        return np.zeros(80, np.float32)
    ys, xs = np.where(small)
    crop = small[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    resized = np.asarray(Image.fromarray(crop.astype(np.uint8)).resize((32, 32), Image.Resampling.NEAREST), bool)
    h, w = crop.shape
    moments = cv2.HuMoments(cv2.moments(crop.astype(np.uint8))).ravel()
    hu = -np.sign(moments) * np.log10(np.abs(moments) + 1e-30)
    vector = np.r_[resized.mean(0), resized.mean(1), hu,
                   np.log(max(1, w) / max(1, h)), crop.mean(), small.mean()]
    return vector.astype(np.float32)


def fit_length(vector, length):
    """Keep a fixed endpoint dimension when a rejected mask has no pixels."""
    values = np.asarray(vector, np.float32).ravel()
    if len(values) > length:
        raise ValueError(f"Feature length {len(values)} exceeds fixed schema {length}")
    return np.pad(values, (0, length - len(values)))


def prepare():
    catalog = rows(ROOT / "data/snapshots/collection_v2.jsonl")
    labels = {r["item_id"]: r for r in rows(ROOT / "eval/visual_labels/annotations.jsonl")}
    items = [r for r in catalog if r["item_id"] in labels]
    records = []
    start = time.perf_counter()
    for row in items:
        image = ImageOps.exif_transpose(Image.open(ROOT / row["image_path"])).convert("RGB")
        full = np.ones((image.height, image.width), bool)
        old = extract_classical(image)
        old_mask = old["mask"].astype(bool)
        v4 = isolate_object(image)["mask_binary"]
        v4 = cv2.resize(v4.astype(np.uint8), image.size, interpolation=cv2.INTER_NEAREST).astype(bool)
        if not v4.any():
            v4 = full
        record = {
            "item_id": row["item_id"], "split": row["split"],
            "object_type": labels[row["item_id"]]["object_type"],
            "material": row["material"],
            "no_mask_color": fit_length(color_vector(image, full), MAX_COLOR_DIM).tolist(),
            "no_mask_shape": fit_length(shape_vector(full), MAX_SHAPE_DIM).tolist(),
            "old_color": fit_length(color_vector(image, old_mask), MAX_COLOR_DIM).tolist(),
            "old_shape": fit_length(shape_vector(old_mask), MAX_SHAPE_DIM).tolist(),
            "v4_color": fit_length(color_vector(image, v4), MAX_COLOR_DIM).tolist(),
            "v4_shape": fit_length(shape_vector(v4), MAX_SHAPE_DIM).tolist(),
        }
        records.append(record)
    return records, time.perf_counter() - start


def fit_binary(x_train, y_train, x_dev, y_dev):
    best = None
    options = []
    for c in (.1, 1, 10, 100):
        model = LogisticRegression(C=c, class_weight="balanced", max_iter=2000, solver="liblinear", random_state=5523)
        model.fit(x_train, y_train)
        score = float(f1_score(y_dev, model.predict(x_dev), zero_division=0))
        options.append({"C": c, "dev_binary_F1": score})
        if best is None or score > best[0] + 1e-12:
            best = score, c, model
    return best[2], best[1], best[0], options


def evaluate(records, task, minimum_train=5):
    train = [r for r in records if r["split"] == "train"]
    dev = [r for r in records if r["split"] == "dev"]
    test = [r for r in records if r["split"] == "test"]
    classes = sorted({
        (r["object_type"] if task == "object_type" else r["material"])
        for r in train
        if (r["object_type"] if task == "object_type" else r["material"])
    })
    outcome = {}
    for method in ("no_mask", "old", "v4"):
        per_class = []
        for label in classes:
            xtr = np.asarray([r[f"{method}_color"] for r in train], np.float32)
            xdv = np.asarray([r[f"{method}_color"] for r in dev], np.float32)
            xte = np.asarray([r[f"{method}_color"] for r in test], np.float32)
            ytr = np.array([r[task] == label for r in train])
            ydv = np.array([r[task] == label for r in dev])
            yte = np.array([r[task] == label for r in test])
            if ytr.sum() < minimum_train or (~ytr).sum() < minimum_train or ydv.sum() < 2 or (~ydv).sum() < 2:
                continue
            model, c, dev_f1, options = fit_binary(xtr, ytr, xdv, ydv)
            pred = model.predict(xte)
            per_class.append({"class": label, "test_binary_F1": float(f1_score(yte, pred, zero_division=0)),
                              "test_precision": float(((pred == 1) & (yte == 1)).sum() / max(1, pred.sum())),
                              "test_recall": float(((pred == 1) & (yte == 1)).sum() / max(1, yte.sum())),
                              "selected_C": c, "dev_binary_F1": dev_f1, "options": options})
        outcome[method] = {
            "macro_binary_F1": float(np.mean([r["test_binary_F1"] for r in per_class])) if per_class else 0.0,
            "classes": per_class,
        }
    return outcome


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    record_path = OUT / "records.json"
    if record_path.is_file():
        cached = json.loads(record_path.read_text(encoding="utf8"))
        records, elapsed = cached["records"], cached.get("prepared_seconds", 0.0)
    else:
        records, elapsed = prepare()
        record_path.write_text(json.dumps({"prepared_seconds": elapsed, "records": records}), encoding="utf8")
    # Normalize cached and fresh vectors to one endpoint schema. Older rejected
    # masks emitted 88/80 dimensions while accepted masks emitted 112/74.
    records = [{
        **row,
        **{field: fit_length(row[field], MAX_COLOR_DIM if field.endswith("_color") else MAX_SHAPE_DIM).tolist()
           for field in ("no_mask_color", "old_color", "v4_color", "no_mask_shape", "old_shape", "v4_shape")},
    } for row in records]
    # Persist the repaired endpoint schema so repeat runs do not depend on the
    # now-documented legacy shape bug.
    record_path.write_text(json.dumps({"prepared_seconds": elapsed, "records": records}), encoding="utf8")
    results = {task: evaluate(records, task) for task in ("object_type", "material")}
    payload = {
        "methods": "no_mask / old GrabCut / v4 approximate mask; foreground HSV only; identical train/dev/test and LogisticRegression C grid",
        "results": results,
        "split_counts": {s: sum(r["split"] == s for r in records) for s in ("train", "dev", "test")},
        "limitations": "AI pixel labels for object_type; catalogue material labels; 218/60/60 small stratified items; F1 is per-class binary macro, not full multiclass/retrieval.",
    }
    (OUT / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf8")
    (OUT / "manifest.json").write_text(json.dumps({p.name: sha(p) for p in OUT.iterdir() if p.is_file() and p.name != "manifest.json"}, indent=2), encoding="utf8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
