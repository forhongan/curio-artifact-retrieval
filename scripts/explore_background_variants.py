"""Explore compact classical segmentation designs on the frozen 24-mask set.

This script is model-selection evidence, not the final method.  It caches no
ground-truth-derived features into production and writes every query-level
variant/result so a subsequent selection is auditable.
"""

from __future__ import annotations

from itertools import product
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

from curio.represent.background_v4 import _background_probability, _border_mask, _lab, _open

ROOT = Path(__file__).resolve().parents[1]
TRUTH = ROOT / "eval/background_truth_20261108"
OUT = ROOT / "artifacts/bg_variants_20261108"


def stage_split(item_id: str) -> str:
    """Deterministic balanced-ish split independent of measured outcomes."""
    import hashlib

    return "dev" if int(hashlib.sha256(("bg-variant-v1:" + item_id).encode()).hexdigest()[:8], 16) % 2 == 0 else "test"


def resize(image: np.ndarray, side: int):
    scale = side / max(image.shape[:2])
    if scale >= 1:
        return image, None
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), scale


def largest_component(mask: np.ndarray):
    mask = ndimage.binary_closing(mask, iterations=1)
    mask = ndimage.binary_fill_holes(mask)
    labels, count = ndimage.label(mask)
    if not count:
        return mask.astype(bool)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == int(np.argmax(sizes))


def rect_grabcut(rgb: np.ndarray, *, side: int, margin: float, iterations: int,
                 mode: str, post: str):
    source, scale = resize(rgb, side)
    height, width = source.shape[:2]
    work = cv2.cvtColor(source, cv2.COLOR_RGB2LAB) if mode == "lab" else source
    x, y = round(width * margin), round(height * margin)
    rect = (x, y, max(2, width - 2 * x), max(2, height - 2 * y))
    mask = np.full((height, width), cv2.GC_BGD, np.uint8)
    cv2.setRNGSeed(0)
    cv2.grabCut(work, mask, rect, None, None, iterations, cv2.GC_INIT_WITH_RECT)
    foreground = np.isin(mask, (cv2.GC_FGD, cv2.GC_PR_FGD))
    if post == "largest":
        foreground = largest_component(foreground)
    elif post == "close":
        foreground = ndimage.binary_closing(foreground, iterations=2)
        foreground = ndimage.binary_fill_holes(foreground)
    if scale is not None:
        foreground = cv2.resize(foreground.astype(np.uint8), rgb.shape[1::-1],
                                interpolation=cv2.INTER_NEAREST).astype(bool)
    return foreground


def mask_grabcut(rgb: np.ndarray, *, side: int, margin: float, iterations: int,
                 mode: str, post: str):
    """Use a central ellipse as probable FG and the border as sure BG."""
    source, scale = resize(rgb, side)
    height, width = source.shape[:2]
    work = cv2.cvtColor(source, cv2.COLOR_RGB2LAB) if mode == "lab" else source
    mask = np.full((height, width), cv2.GC_PR_BGD, np.uint8)
    border, thick = _border_mask((height, width), margin)
    mask[border] = cv2.GC_BGD
    ellipse = np.zeros((height, width), np.uint8)
    cv2.ellipse(ellipse, (width // 2, height // 2),
                (max(2, round(width * .30)), max(2, round(height * .34))), 0, 0, 360, 1, -1)
    mask[(ellipse > 0) & ~border] = cv2.GC_PR_FGD
    cv2.setRNGSeed(0)
    cv2.grabCut(work, mask, None, None, None, iterations, cv2.GC_INIT_WITH_MASK)
    foreground = np.isin(mask, (cv2.GC_FGD, cv2.GC_PR_FGD))
    if post == "largest":
        foreground = largest_component(foreground)
    elif post == "close":
        foreground = ndimage.binary_closing(foreground, iterations=2)
        foreground = ndimage.binary_fill_holes(foreground)
    if scale is not None:
        foreground = cv2.resize(foreground.astype(np.uint8), rgb.shape[1::-1],
                                interpolation=cv2.INTER_NEAREST).astype(bool)
    return foreground


def border_probability(rgb: np.ndarray, *, side: int, cutoff: float, close: int,
                       post: str):
    source, scale = resize(rgb, side)
    lab = _lab(source)
    border, _ = _border_mask(lab.shape, .04)
    probability, _ = _background_probability(lab, border)
    foreground = probability < cutoff
    if close:
        foreground = ndimage.binary_closing(foreground, iterations=close)
    if post == "largest":
        foreground = largest_component(foreground)
    elif post == "fill":
        foreground = ndimage.binary_fill_holes(foreground)
    if scale is not None:
        foreground = cv2.resize(foreground.astype(np.uint8), rgb.shape[1::-1],
                                interpolation=cv2.INTER_NEAREST).astype(bool)
    return foreground


def metrics(prediction, truth):
    intersection = np.logical_and(prediction, truth).sum()
    union = np.logical_or(prediction, truth).sum()
    precision = intersection / prediction.sum() if prediction.sum() else 0.0
    recall = intersection / truth.sum() if truth.sum() else 0.0
    return {
        "IoU": float(intersection / union) if union else 1.0,
        "Dice": float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
        "precision": float(precision), "recall": float(recall),
        "predicted_area": float(prediction.mean()), "truth_area": float(truth.mean()),
    }


def variants():
    output = []
    # A bounded factorial design: enough to test initialization, work space,
    # margin and post-processing without spending hours on redundant settings.
    for kind, side, margin, iterations, mode, post in product(
            ("rect", "mask"), (256, 384), (.03, .06, .10), (2, 4), ("rgb", "lab"), ("raw", "largest")):
        output.append({"family": kind, "side": side, "margin": margin, "iterations": iterations,
                       "mode": mode, "post": post})
    for side, cutoff, close, post in product((128, 192, 256), (.03, .06, .10, .15), (0, 1), ("raw", "fill", "largest")):
        output.append({"family": "border", "side": side, "cutoff": cutoff,
                       "close": close, "post": post})
    return output


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records = json.loads((TRUTH / "records.json").read_text(encoding="utf-8"))
    images = {row["item_id"]: _open(ROOT / row["image_path"]) for row in records}
    truths = {row["item_id"]: np.asarray(Image.open(TRUTH / f"{row['item_id']}.png"), bool)
              for row in records}
    result_path = OUT / "results.jsonl"
    result_path.write_text("", encoding="utf-8")
    for index, variant in enumerate(variants(), 1):
        for record in records:
            source = images[record["item_id"]]
            start = time.perf_counter()
            if variant["family"] == "border":
                prediction = border_probability(source, side=variant["side"], cutoff=variant["cutoff"],
                                                close=variant["close"], post=variant["post"])
            else:
                predictor = rect_grabcut if variant["family"] == "rect" else mask_grabcut
                prediction = predictor(source, side=variant["side"], margin=variant["margin"],
                                       iterations=variant["iterations"], mode=variant["mode"],
                                       post=variant["post"])
            elapsed = time.perf_counter() - start
            truth = truths[record["item_id"]]
            if prediction.shape != truth.shape:
                prediction = cv2.resize(prediction.astype(np.uint8), truth.shape[::-1],
                                        interpolation=cv2.INTER_NEAREST).astype(bool)
            result = {"variant_id": index, **variant, "item_id": record["item_id"],
                      "truth_type": record["truth_type"], "stage_split": stage_split(record["item_id"]),
                      "seconds": elapsed, **metrics(prediction, truth)}
            with result_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
        print(index, len(variants()), variant, flush=True)


if __name__ == "__main__":
    main()
