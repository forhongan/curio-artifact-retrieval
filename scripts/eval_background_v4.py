"""Evaluate deterministic background isolation without using catalog semantics.

The first suite is synthetic and has pixel-exact masks. The second performs
frozen, stratified real-image QA and a downstream foreground-color comparison.
"""

from __future__ import annotations

import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from curio.represent.background_v4 import isolate_object
from curio.represent.classical_cv import extract_classical
from curio.represent.visual_parts import extract_visual_parts

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/bg_v4"


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf8").splitlines()]


def sha256(path):
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(prediction, truth):
    prediction, truth = np.asarray(prediction, bool), np.asarray(truth, bool)
    inter = np.logical_and(prediction, truth).sum()
    union = np.logical_or(prediction, truth).sum()
    iou = float(inter / union) if union else 1.0
    p = float(inter / prediction.sum()) if prediction.sum() else 0.0
    r = float(inter / truth.sum()) if truth.sum() else 0.0
    return {"IoU": iou, "precision": p, "recall": r, "boundary_f1": boundary_f1(prediction, truth)}


def boundary_f1(prediction, truth, tolerance=2):
    kernel = np.ones((tolerance * 2 + 1,) * 2, np.uint8)
    pe = cv2.Canny(prediction.astype(np.uint8) * 255, 50, 100) > 0
    te = cv2.Canny(truth.astype(np.uint8) * 255, 50, 100) > 0
    dp = cv2.dilate(te.astype(np.uint8), kernel).astype(bool)
    dt = cv2.dilate(pe.astype(np.uint8), kernel).astype(bool)
    precision = float((pe & dp).sum() / pe.sum()) if pe.any() else 1.0
    recall = float((te & dt).sum() / te.sum()) if te.any() else 1.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def synthetic():
    records = []
    cases = []
    for i, bg in enumerate(("#f7f7f7", "#d0d0d0", "#203050")):
        image = Image.new("RGB", (320, 300), bg)
        draw = ImageDraw.Draw(image)
        draw.ellipse((75, 40, 245, 265), fill="#1a6a9a")
        mask = np.zeros((300, 320), np.uint8)
        cv2.ellipse(mask, (160, 152), (85, 113), 0, 0, 360, 1, -1)
        cases.append((image, mask.astype(bool), f"object_bg_{i}"))
        image = Image.new("RGB", (320, 300), bg)
        draw.rectangle((35, 45, 285, 256), fill="#8d6b40")
        mask = np.zeros((300, 320), np.uint8)
        cv2.rectangle(mask, (35, 45), (285, 256), 1, -1)
        cases.append((image, mask.astype(bool), f"planar_bg_{i}"))
    for image, truth, name in cases:
        result = isolate_object(image)
        prediction = cv2.resize(
            result["mask_binary"].astype(np.uint8), truth.shape[::-1], interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        records.append({"name": name, "status": result["status"], **metrics(prediction, truth)})
    payload = {
        "suite": "six simple synthetic object/planar images over light gray dark backgrounds",
        "results": records,
        "accepted": sum(r["status"] == "accepted" for r in records),
        "mean_IoU_all_with_abstention_as_empty": float(np.mean([r["IoU"] for r in records])),
        "mean_IoU_accepted": float(np.mean([r["IoU"] for r in records if r["status"] == "accepted"])) if any(r["status"] == "accepted" for r in records) else 0.0,
        "mean_boundary_f1_accepted": float(np.mean([r["boundary_f1"] for r in records if r["status"] == "accepted"])) if any(r["status"] == "accepted" for r in records) else 0.0,
    }
    # Exact mask augmentation: paste synthetic object over stratified real photos.
    catalog = rows(ROOT / "data/snapshots/collection_v2.jsonl")
    import hashlib

    chosen = sorted(catalog, key=lambda r: hashlib.sha256(("bg-stability" + r["item_id"]).encode()).hexdigest())[:12]
    stability = []
    for row in chosen:
        with Image.open(ROOT / row["image_path"]) as loaded:
            base = ImageOps.exif_transpose(loaded).convert("RGB")
        base.thumbnail((420, 320))
        object_layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(object_layer)
        cx, cy, rx, ry = base.width // 2, base.height // 2, max(15, base.width // 7), max(20, base.height // 4)
        draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=(220, 55, 40, 255))
        composite = Image.alpha_composite(base.convert("RGBA"), object_layer).convert("RGB")
        truth = np.zeros((base.height, base.width), bool)
        yy, xx = np.mgrid[: base.height, : base.width]
        truth[((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1] = True
        result = isolate_object(composite)
        pred = result["mask_binary"]
        if pred.shape != truth.shape:
            pred = cv2.resize(pred.astype(np.uint8), truth.shape[::-1], interpolation=cv2.INTER_NEAREST).astype(bool)
        stability.append({"item_id": row["item_id"], "status": result["status"], **metrics(pred, truth)})
    payload["synthetic_over_real_background"] = {
        "results": stability,
        "mean_IoU_all_with_abstention_as_empty": float(np.mean([r["IoU"] for r in stability])),
        "mean_IoU_accepted": float(np.mean([r["IoU"] for r in stability if r["status"] == "accepted"])),
        "accepted": sum(r["status"] == "accepted" for r in stability),
        "limits": "Exact synthetic foreground is red ellipse over real museum backgrounds; tests stability under real clutter, not real-object segmentation accuracy.",
    }
    return payload


def real_qa(rows_per_stratum=3):
    catalog = rows(ROOT / "data/snapshots/collection_v2.jsonl")
    import hashlib

    strata = {}
    for row in catalog:
        kind = row["object_type"] or "other"
        strata.setdefault(kind, []).append(row)
    chosen = []
    for kind, values in sorted(strata.items()):
        ranked = sorted(values, key=lambda r: hashlib.sha256(("bg-v4-review" + r["item_id"]).encode()).hexdigest())
        chosen.extend(ranked[:rows_per_stratum])
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 17)
    tile_width, tile_height = 300, 180
    sheet = Image.new("RGB", (1200, ((len(chosen) + 3) // 4) * tile_height), "white")
    draw = ImageDraw.Draw(sheet)
    records = []
    start = time.perf_counter()
    for i, row in enumerate(chosen):
        result = isolate_object(ROOT / row["image_path"])
        with Image.open(ROOT / row["image_path"]) as loaded:
            image = ImageOps.exif_transpose(loaded).convert("RGB")
        image.thumbnail((140, 135))
        mask_image = Image.fromarray(result["mask_binary"].astype(np.uint8) * 255).convert("RGB")
        mask_image.thumbnail((140, 135))
        x, y = (i % 4) * tile_width, (i // 4) * tile_height
        sheet.paste(image, (x, y + 35))
        sheet.paste(mask_image, (x + 150, y + 35))
        draw.text((x + 4, y + 5), row["item_id"], fill="black", font=font)
        records.append({
            "item_id": row["item_id"],
            "object_type": row["object_type"],
            "material": row["material"],
            "status": result["status"],
            "bbox": result["bbox"],
            **result["diagnostics"],
        })
    sheet.save(OUT / "real_stratified_qa.jpg", quality=92)
    # Downstream color consistency: compare own foreground HSV color with two
    # established background estimators. This is a color-isolation signal, not
    # semantic color accuracy because no pixel-level human truth exists.
    downstream = []
    for row in chosen[:36]:
        path = ROOT / row["image_path"]
        own = isolate_object(path)
        gc = extract_classical(path)
        old = extract_visual_parts(path)
        if own["status"] != "accepted":
            continue
        with Image.open(path) as loaded:
            image = np.asarray(ImageOps.exif_transpose(loaded).convert("RGB"))
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        own_mask = cv2.resize(own["mask_binary"].astype(np.uint8), image.shape[1::-1], interpolation=cv2.INTER_NEAREST).astype(bool)
        gc_mask = gc["mask"].astype(bool)
        gc_mask = cv2.resize(gc_mask.astype(np.uint8), image.shape[1::-1], interpolation=cv2.INTER_NEAREST).astype(bool)
        downstream.append({
            "item_id": row["item_id"],
            "own_foreground_fraction": float(own_mask.mean()),
            "grabcut_foreground_fraction": float(gc_mask.mean()),
            "foreground_area_overlap": metrics(own_mask, gc_mask)["IoU"] if gc_mask.any() else None,
            "hsv_median_own": [float(x) for x in np.median(hsv[own_mask], axis=0)] if own_mask.any() else None,
            "hsv_median_grabcut": [float(x) for x in np.median(hsv[gc_mask], axis=0)] if gc_mask.any() else None,
            "old_visual_parts_status": old["diagnostics"]["status"],
        })
    return {
        "count": len(records),
        "selection": "deterministic SHA256 at most three per catalog object_type; includes all available strata",
        "accepted": sum(r["status"] == "accepted" for r in records),
        "elapsed_seconds": time.perf_counter() - start,
        "records": records,
        "color_audit": downstream,
        "limits": "Visual QA has no pixel-level human ground truth; inspect real_stratified_qa.jpg. Downstream agreement with GrabCut/old estimator is not proof of correctness.",
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {"synthetic": synthetic(), "real_qa": real_qa()}
    (OUT / "evaluation.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf8")
    (OUT / "manifest.json").write_text(json.dumps({p.name: sha256(p) for p in OUT.iterdir() if p.is_file() and p.name != "manifest.json"}, indent=2), encoding="utf8")
    print(json.dumps({"synthetic": payload["synthetic"], "real": {k: v for k, v in payload["real_qa"].items() if k != "records" and k != "color_audit"}}, indent=2))


if __name__ == "__main__":
    main()
