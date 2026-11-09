"""Mask invariance to background replacement and pixel-only image retrieval."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from curio.evaluate.catalogue_tasks import single_label  # noqa: E402
from curio.represent.background_benchmark import (  # noqa: E402
    METHODS,
    _border_component,
    _old_grabcut,
    _rect_grabcut,
    _v4_mask,
    descriptors,
    masks_for_image,
    open_rgb,
)
from explore_background_variants import stage_split  # noqa: E402
from run_background_downstream_v4 import LABELS, OUT, TRUTH, read_jsonl  # noqa: E402


def normalize(values):
    array = np.asarray(values, np.float64)
    return array / np.maximum(np.linalg.norm(array, axis=-1, keepdims=True), 1e-12)


def background_composites(rgb, truth):
    h, w = rgb.shape[:2]
    yy, xx = np.mgrid[:h, :w]
    checker = ((xx // max(8, w // 8) + yy // max(8, h // 8)) % 2)[..., None]
    backgrounds = (
        np.full_like(rgb, 235),
        np.broadcast_to(np.array([32, 48, 72], np.uint8), rgb.shape),
        np.where(checker, np.array([180, 150, 120]), np.array([90, 120, 145])).astype(np.uint8),
    )
    return [np.where(truth[..., None], rgb, background).astype(np.uint8) for background in backgrounds]


def run_mask(method, rgb):
    if method == "no_mask":
        return np.ones(rgb.shape[:2], bool), {"valid": True}
    if method == "old_grabcut":
        return _old_grabcut(rgb)
    if method == "border_cc":
        return _border_component(rgb)
    if method == "v4":
        return _v4_mask(rgb)
    return _rect_grabcut(rgb, largest=method.endswith("largest"))


def invariance():
    records = json.loads((TRUTH / "records.json").read_text(encoding="utf-8"))
    outputs = []
    for row in records:
        rgb = open_rgb(ROOT / row["image_path"])
        with Image.open(TRUTH / f"{row['item_id']}.png") as opened:
            truth = np.asarray(opened, bool)
        truth = cv2.resize(truth.astype(np.uint8), rgb.shape[1::-1],
                           interpolation=cv2.INTER_NEAREST).astype(bool)
        _, original = masks_for_image(ROOT / row["image_path"])
        for variant, composite in enumerate(background_composites(rgb, truth)):
            for method in METHODS:
                initial_mask, initial_diag = original[method]
                initial = descriptors(rgb, initial_mask, initial_diag)
                new_mask, new_diag = run_mask(method, composite)
                changed = descriptors(composite, new_mask, new_diag)
                row_out = {"item_id": row["item_id"], "split": stage_split(row["item_id"]),
                           "method": method, "background": variant,
                           "original_valid": bool(initial_diag["valid"]),
                           "changed_valid": bool(new_diag["valid"])}
                for channel in ("shape", "color", "texture", "all"):
                    left, right = normalize(initial[channel]), normalize(changed[channel])
                    row_out[f"{channel}_cosine"] = float(left @ right)
                outputs.append(row_out)
    summary = {}
    for method in METHODS:
        subset = [row for row in outputs if row["method"] == method]
        summary[method] = {"pairs": len(subset),
                           "both_masks_valid": sum(r["original_valid"] and r["changed_valid"] for r in subset) / len(subset),
                           **{channel: float(np.mean([r[f"{channel}_cosine"] for r in subset]))
                              for channel in ("shape", "color", "texture", "all")}}
    return {"summary": summary, "pairs": outputs,
            "limits": "24 approximate masks, pasted over three synthetic backgrounds; original foreground edges may contain background pixels; not real viewpoint invariance"}


def retrieval():
    path = OUT / "features.npz"
    with np.load(path, allow_pickle=False) as saved:
        metadata = json.loads(str(saved["metadata"]))
        vectors = {f"{method}__{channel}": saved[f"{method}__{channel}"].copy()
                   for method in METHODS for channel in ("shape", "color", "texture", "all")}
    by_id = {row["item_id"]: row for row in read_jsonl(LABELS)}
    rows = [by_id[item_id] for item_id in metadata["rows"]]
    domain_results = {}
    for domain in ("form", "broad_material", "fine_material"):
        domain_results[domain] = {}
        for method in METHODS:
            domain_results[domain][method] = {}
            for channel in ("shape", "color", "texture", "all"):
                details = []
                features = normalize(vectors[f"{method}__{channel}"])
                for split in ("dev", "test"):
                    indices = [i for i, row in enumerate(rows) if row["split"] == split
                               and single_label(row, domain) is not None]
                    gallery_labels = [single_label(rows[i], domain) for i in indices]
                    for position, index in enumerate(indices):
                        label = gallery_labels[position]
                        peers = [j for j in range(len(indices)) if j != position and gallery_labels[j] == label
                                 and rows[indices[j]]["object_group"] != rows[index]["object_group"]]
                        if not peers:
                            continue
                        scores = features[indices] @ features[index]
                        order = sorted((j for j in range(len(indices)) if j != position
                                        and rows[indices[j]]["object_group"] != rows[index]["object_group"]),
                                       key=lambda j: (-scores[j], rows[indices[j]]["item_id"]))[:5]
                        details.append({"split": split, "item_id": rows[index]["item_id"],
                                        "label": label, "gallery": len(indices) - 1,
                                        "precision_at_5": sum(gallery_labels[j] == label for j in order) / 5,
                                        "hit_at_5": any(gallery_labels[j] == label for j in order)})
                domain_results[domain][method][channel] = {
                    split: {"queries": len(current),
                            "macro_p_at_5": float(np.mean([r["precision_at_5"] for r in current])) if current else None,
                            "hit_at_5": float(np.mean([r["hit_at_5"] for r in current])) if current else None}
                    for split in ("dev", "test")
                    if (current := [r for r in details if r["split"] == split])
                }
    return {"domains": domain_results, "method": "within-split leave-one-group-out image similarity; same catalogue class is relevant; unknown labels excluded",
            "limits": "closed known-label galleries; form/material identity is a proxy for retrieval relevance; highly imbalanced classes and exposed test split"}


if __name__ == "__main__":
    cv2.setNumThreads(1)
    report = {"background_replacement": invariance(), "image_similarity": retrieval()}
    (OUT / "invariance_retrieval.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                                   encoding="utf-8")
    print(json.dumps({"background_replacement": report["background_replacement"]["summary"],
                      "form_retrieval": report["image_similarity"]["domains"]["form"]},
                     ensure_ascii=False, indent=2))
