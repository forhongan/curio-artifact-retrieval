"""Fixed pixel-only segmentation comparators and downstream descriptors.

The functions in this module deliberately avoid catalogue fields.  A mask may
change the visual descriptor passed to a classifier, but label text can never
change the mask or the pixels.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage
from skimage.feature import hog, local_binary_pattern
from skimage.transform import resize as sk_resize

from curio.represent.background_v4 import (
    _background_probability,
    _border_mask,
    _lab,
    isolate_object,
)

METHODS = (
    "no_mask",
    "old_grabcut",
    "border_cc",
    "v4",
    "rect_gc384_raw",
    "rect_gc384_largest",
)
FEATURE_CHANNELS = ("shape", "color", "texture", "all", "all_without_shape")
FEATURE_SIDE = 224


def open_rgb(path: str | Path) -> np.ndarray:
    with Image.open(path) as opened:
        return np.asarray(ImageOps.exif_transpose(opened).convert("RGB"))


def _resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == shape:
        return mask.astype(bool)
    return cv2.resize(mask.astype(np.uint8), shape[::-1],
                      interpolation=cv2.INTER_NEAREST).astype(bool)


def _largest_component(mask: np.ndarray, *, fill: bool = False) -> np.ndarray:
    current = ndimage.binary_closing(mask, iterations=1)
    if fill:
        current = ndimage.binary_fill_holes(current)
    labels, count = ndimage.label(current)
    if not count:
        return current.astype(bool)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == int(np.argmax(sizes))


def _rect_grabcut(rgb: np.ndarray, *, side: int = 384, margin: float = .03,
                  iterations: int = 2, largest: bool = False) -> tuple[np.ndarray, dict]:
    scale = side / max(rgb.shape[:2])
    source = (cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
              if scale < 1 else rgb.copy())
    height, width = source.shape[:2]
    x, y = round(width * margin), round(height * margin)
    rectangle = (x, y, max(2, width - 2 * x), max(2, height - 2 * y))
    mask = np.full((height, width), cv2.GC_BGD, np.uint8)
    cv2.setRNGSeed(0)
    cv2.grabCut(source, mask, rectangle, None, None, iterations, cv2.GC_INIT_WITH_RECT)
    foreground = np.isin(mask, (cv2.GC_FGD, cv2.GC_PR_FGD))
    if largest:
        foreground = _largest_component(foreground, fill=True)
    foreground = _resize_mask(foreground, rgb.shape[:2])
    return foreground, {
        "status": "completed", "valid": bool(_area_plausible(foreground)),
        "score_method": "grabcut_rectangle", "method_side": side,
    }


def _old_grabcut(rgb: np.ndarray) -> tuple[np.ndarray, dict]:
    side = 192
    scale = side / max(rgb.shape[:2])
    source = (cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
              if scale < 1 else rgb.copy())
    height, width = source.shape[:2]
    mx, my = max(1, int(width * .05)), max(1, int(height * .05))
    mask = np.zeros((height, width), np.uint8)
    cv2.setRNGSeed(int.from_bytes(hashlib.sha256(source.tobytes()).digest()[:4], "little")
                   & 0x7FFFFFFF)
    cv2.grabCut(source, mask, (mx, my, width - 2 * mx, height - 2 * my),
                None, None, 3, cv2.GC_INIT_WITH_RECT)
    foreground = np.isin(mask, (cv2.GC_FGD, cv2.GC_PR_FGD))
    foreground = _resize_mask(foreground, rgb.shape[:2])
    return foreground, {
        "status": "completed", "valid": bool(_area_plausible(foreground)),
        "score_method": "legacy_rect_grabcut", "method_side": side,
    }


def _border_component(rgb: np.ndarray) -> tuple[np.ndarray, dict]:
    side = 256
    scale = side / max(rgb.shape[:2])
    source = (cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
              if scale < 1 else rgb.copy())
    lab = _lab(source)
    border, _ = _border_mask(lab.shape, .04)
    probability, _ = _background_probability(lab, border)
    foreground = probability < .12
    foreground = ndimage.binary_opening(foreground, iterations=1)
    foreground = ndimage.binary_closing(foreground, iterations=1)
    foreground = _largest_component(foreground, fill=True)
    foreground = _resize_mask(foreground, rgb.shape[:2])
    return foreground, {
        "status": "completed", "valid": bool(_area_plausible(foreground)),
        "score_method": "border_color_component", "method_side": side,
    }


def _v4_mask(image: np.ndarray) -> tuple[np.ndarray, dict]:
    result = isolate_object(Image.fromarray(image))
    mask = _resize_mask(result["mask_binary"], image.shape[:2])
    return mask, {
        "status": result["status"], "valid": result["status"] == "accepted",
        "score_method": "multi_evidence_v4",
        "reasons": list(result["reasons"]),
        "archetype": result["diagnostics"]["archetype"],
    }


def masks_for_image(path: str | Path) -> tuple[np.ndarray, dict[str, tuple[np.ndarray, dict]]]:
    """Return the six frozen candidate masks using pixels only."""
    rgb = open_rgb(path)
    height, width = rgb.shape[:2]
    full = np.ones((height, width), bool)
    def timed(fn):
        start = time.perf_counter()
        mask, diagnostics = fn(rgb)
        diagnostics["seconds"] = time.perf_counter() - start
        return mask, diagnostics

    old, old_diag = timed(_old_grabcut)
    border, border_diag = timed(_border_component)
    v4, v4_diag = timed(_v4_mask)
    raw, raw_diag = timed(lambda image: _rect_grabcut(image, largest=False))
    largest, largest_diag = timed(lambda image: _rect_grabcut(image, largest=True))
    return rgb, {
        "no_mask": (full, {"status": "complete", "valid": True, "seconds": 0.0,
                               "score_method": "whole_image"}),
        "old_grabcut": (old, old_diag),
        "border_cc": (border, border_diag),
        "v4": (v4, v4_diag),
        "rect_gc384_raw": (raw, raw_diag),
        "rect_gc384_largest": (largest, largest_diag),
    }


def _area_plausible(mask: np.ndarray) -> bool:
    area = float(mask.mean())
    return .02 <= area <= .95


def _thumbnail(image: np.ndarray, side: int = FEATURE_SIDE) -> tuple[np.ndarray, float]:
    scale = side / max(image.shape[:2])
    return (cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), scale) \
        if scale < 1 else (image.copy(), 1.0)


def shape_features(mask: np.ndarray) -> np.ndarray:
    """Normalized silhouette descriptor; an empty/abstained mask returns zeros."""
    mask = np.asarray(mask, bool)
    if not mask.any():
        return np.zeros(302, np.float32)
    ys, xs = np.where(mask)
    crop = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    height, width = crop.shape
    preview = cv2.resize(crop.astype(np.uint8), (16, 16),
                         interpolation=cv2.INTER_AREA).astype(np.float32)
    moments = cv2.HuMoments(cv2.moments(crop.astype(np.uint8))).ravel()
    hu = -np.sign(moments) * np.log10(np.abs(moments) + 1e-30)
    contours, _ = cv2.findContours(crop.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    perimeter = float(sum(cv2.arcLength(item, True) for item in contours))
    area = float(crop.sum())
    compactness = 4 * np.pi * area / max(perimeter ** 2, 1e-6)
    extent = area / max(1, height * width)
    aspect = width / max(1, height)
    crop_fraction = area / mask.size
    row_profile = preview.mean(axis=1)
    column_profile = preview.mean(axis=0)
    return np.r_[preview.ravel(), row_profile, column_profile, hu, np.log(aspect), extent,
                 compactness, crop_fraction, mask.mean(), width / mask.shape[1],
                 height / mask.shape[0]].astype(np.float32)


def _histogram(values: np.ndarray, bins: int, value_range: tuple[float, float]) -> np.ndarray:
    histogram = np.histogram(values, bins=bins, range=value_range)[0].astype(np.float32)
    histogram /= max(1.0, float(histogram.sum()))
    return np.r_[histogram, np.sqrt(histogram)].astype(np.float32)


def color_features(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    # Masked color uses the full image as its fallback; a failed segment can
    # still supply color, while its invalid status remains separately reported.
    pixels = hsv[np.asarray(mask, bool)]
    if not len(pixels):
        pixels = hsv.reshape(-1, 3)
    return np.r_[_histogram(pixels[:, 0], 24, (0, 180)),
                 _histogram(pixels[:, 1], 16, (0, 256)),
                 _histogram(pixels[:, 2], 16, (0, 256))].astype(np.float32)


def texture_features(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    valid = np.asarray(mask, bool)
    # Fixed-size texture canvas makes every image contribute the same dense
    # descriptor dimension; extreme panoramas are padded to preserve aspect.
    scaled = max(image.shape[:2])
    resized = sk_resize(image, (scaled, scaled), anti_aliasing=True,
                        preserve_range=True).astype(np.uint8)
    resized_mask = sk_resize(valid.astype(np.uint8), (scaled, scaled), order=0,
                             preserve_range=True).astype(bool)
    preview, _ = _thumbnail(resized, 128)
    current = cv2.resize(resized_mask.astype(np.uint8), preview.shape[1::-1],
                         interpolation=cv2.INTER_NEAREST).astype(bool)
    if not current.any():
        current = np.ones(preview.shape[:2], bool)
    gray = cv2.cvtColor(preview, cv2.COLOR_RGB2GRAY)
    neutral = gray.copy()
    neutral[~current] = 127
    edge = hog(neutral, orientations=9, pixels_per_cell=(16, 16),
               cells_per_block=(2, 2), block_norm="L2-Hys", feature_vector=True)
    edges = cv2.Canny(neutral, 40, 120) > 0
    textures = []
    for radius, points in ((1, 8), (2, 16)):
        pattern = local_binary_pattern(gray, points, radius, method="uniform")
        textures.append(_histogram(pattern[current], points + 2, (0, points + 2)))
    scalars = np.array([edges[current].mean(), float(gray[current].std()),
                        float(np.mean(current))], np.float32)
    return np.r_[edge, *textures, scalars].astype(np.float32)


def descriptors(image: np.ndarray, mask: np.ndarray, diagnostics: dict) -> dict:
    preview, _ = _thumbnail(image)
    current = _resize_mask(mask, preview.shape[:2])
    shape = shape_features(current) if diagnostics.get("valid") else np.zeros(302, np.float32)
    color = color_features(preview, current)
    texture = texture_features(preview, current)
    return {
        "shape": shape,
        "color": color,
        "texture": texture,
        "all": np.r_[shape, color, texture].astype(np.float32),
        "all_without_shape": np.r_[color, texture].astype(np.float32),
    }


def benchmark_image(path: str | Path) -> dict:
    image, masks = masks_for_image(path)
    descriptors_by_method = {}
    diagnostics = {}
    coverage = {}
    for name in METHODS:
        mask, diagnostic = masks[name]
        descriptors_by_method[name] = descriptors(image, mask, diagnostic)
        diagnostics[name] = diagnostic
        coverage[name] = bool(diagnostic["valid"])
    return {"descriptors": descriptors_by_method, "diagnostics": diagnostics,
            "coverage": coverage}
