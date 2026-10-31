"""Classical, abstaining foreground and visual descriptors; no trained segmenter.

All descriptors are image-only. Validity flags must gate downstream similarities:
a rejected mask produces zero descriptors, never a confident background descriptor.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage


VERSION = "border-plane-specialists-v1"
FEATURE_DIMENSIONS = {"shape": 132, "color": 192, "texture": 34}
PARAMETERS = {"max_side": 256, "border_fraction": 0.035,
              "max_border_residual_p90": 0.13, "min_border_inlier_fraction": 0.80,
              "foreground_fraction_range": [0.025, 0.83], "min_dominant_component": 0.85,
              "max_foreground_border_fraction": 0.025,
              "max_enclosed_missing_fraction": 0.12,
              "threshold_floor": 0.065, "threshold_ceiling": 0.20}


def _unit(values):
    values = np.asarray(values, dtype=np.float32)
    norm = float(np.linalg.norm(values))
    return values / norm if norm > 1e-12 else np.zeros_like(values)


def _open(image):
    if isinstance(image, (str, Path)):
        with Image.open(image) as loaded:
            return ImageOps.exif_transpose(loaded).convert("RGB")
    return ImageOps.exif_transpose(image).convert("RGB")


def _foreground(rgb):
    """Robust planar RGB background estimated exclusively from border pixels."""
    height, width = rgb.shape[:2]
    yy, xx = np.mgrid[-1:1:complex(height), -1:1:complex(width)]
    design = np.stack([np.ones_like(xx), xx, yy], axis=-1)
    thickness = max(2, round(min(height, width) * PARAMETERS["border_fraction"]))
    border = np.zeros((height, width), dtype=bool)
    border[:thickness] = border[-thickness:] = True
    border[:, :thickness] = border[:, -thickness:] = True
    bx, by = design[border], rgb[border]
    # Deterministic bounded sampling and trimmed least squares prevent a small
    # object intrusion at the border from dominating a studio background fit.
    step = max(1, len(bx) // 1500)
    bx, by = bx[::step], by[::step]
    keep = np.ones(len(bx), dtype=bool)
    for _ in range(4):
        coefficients = np.linalg.lstsq(bx[keep], by[keep], rcond=None)[0]
        residual = np.linalg.norm(by - bx @ coefficients, axis=1)
        keep = residual <= max(0.035, float(np.quantile(residual, 0.80)))
    distance = np.linalg.norm(rgb - design @ coefficients, axis=-1)
    border_distance = distance[border]
    p90 = float(np.quantile(border_distance, 0.90))
    median = float(np.median(border_distance))
    mad = float(np.median(np.abs(border_distance - median)))
    threshold = float(np.clip(median + 5 * mad + 0.025,
                              PARAMETERS["threshold_floor"], PARAMETERS["threshold_ceiling"]))
    raw = distance > threshold
    raw = ndimage.binary_closing(raw, iterations=1)
    labels, count = ndimage.label(raw)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    mask = labels == int(np.argmax(sizes)) if count else np.zeros_like(raw)
    dominant = float(mask.sum() / max(1, raw.sum()))
    fraction = float(mask.mean())
    edge_fraction = float(mask[border].mean())
    border_inliers = float((border_distance <= threshold).mean())
    filled = ndimage.binary_fill_holes(mask)
    enclosed_missing = float((filled.sum() - mask.sum()) / max(1, filled.sum()))
    reaches_edge = bool(mask[1, :].any() or mask[-2, :].any()
                        or mask[:, 1].any() or mask[:, -2].any()) if min(mask.shape) > 3 else True
    reasons = []
    if p90 > PARAMETERS["max_border_residual_p90"] or border_inliers < 0.80:
        reasons.append("complex_or_inconsistent_border")
    if not 0.025 <= fraction <= 0.83:
        reasons.append("foreground_area_out_of_range")
    if dominant < PARAMETERS["min_dominant_component"]:
        reasons.append("multiple_or_fragmented_foregrounds")
    if edge_fraction > PARAMETERS["max_foreground_border_fraction"]:
        reasons.append("foreground_touches_frame")
    elif reaches_edge:
        reasons.append("foreground_reaches_image_edge")
    if enclosed_missing > PARAMETERS["max_enclosed_missing_fraction"]:
        reasons.append("large_enclosed_missing_region")
    # Masks remain available for QA even on abstention, but cannot authorize scores.
    return mask, {"status": "valid" if not reasons else "unknown", "reasons": reasons,
                  "border_residual_p90": p90, "border_inlier_fraction": border_inliers,
                  "threshold": threshold, "foreground_fraction": fraction,
                  "dominant_component_fraction": dominant, "foreground_border_fraction": edge_fraction,
                  "enclosed_missing_fraction": enclosed_missing}


def _shape(mask):
    yy, xx = np.where(mask)
    crop = mask[yy.min():yy.max() + 1, xx.min():xx.max() + 1]
    grid = np.asarray(Image.fromarray(crop).resize((32, 32), Image.Resampling.NEAREST), bool)
    row_width = grid.mean(axis=1)
    col_height = grid.mean(axis=0)
    coordinates = np.linspace(-0.5, 0.5, 32)
    row_center = (grid * coordinates[None, :]).sum(axis=1) / np.maximum(1, grid.sum(axis=1))
    col_center = (grid * coordinates[:, None]).sum(axis=0) / np.maximum(1, grid.sum(axis=0))
    h, w = crop.shape
    aspect = float(np.clip(np.log(w / h), -3, 3) / 3)
    moments = [aspect, float(crop.mean()), float(np.std(xx) / max(1, w)),
               float(np.std(yy) / max(1, h))]
    return _unit(np.r_[row_width, col_height, row_center, col_center, moments])


def _color(rgb, mask):
    hsv = np.asarray(Image.fromarray(np.uint8(np.clip(rgb * 255, 0, 255))).convert("HSV"))
    hist, _ = np.histogramdd(hsv[mask], bins=(12, 4, 4), range=((0, 256),) * 3)
    # Hellinger map: dot product measures distribution overlap without dominant
    # bins completely hiding small colored decoration.
    return _unit(np.sqrt(hist.ravel() / max(1, hist.sum())))


def _texture(rgb, mask):
    gray = rgb @ np.array([0.299, 0.587, 0.114])
    interior = ndimage.binary_erosion(mask, iterations=2)
    center = gray[1:-1, 1:-1]
    neighbors = [gray[:-2, :-2], gray[:-2, 1:-1], gray[:-2, 2:], gray[1:-1, 2:],
                 gray[2:, 2:], gray[2:, 1:-1], gray[2:, :-2], gray[1:-1, :-2]]
    bits = np.stack([neighbor >= center for neighbor in neighbors], axis=-1)
    transitions = (bits != np.roll(bits, 1, axis=-1)).sum(axis=-1)
    codes = np.where(transitions <= 2, bits.sum(axis=-1), 9)
    lbp = np.bincount(codes[interior[1:-1, 1:-1]], minlength=10).astype(float)
    gx = ndimage.sobel(gray, axis=1) / 8
    gy = ndimage.sobel(gray, axis=0) / 8
    magnitude = np.hypot(gx, gy)
    angle = np.mod(np.arctan2(gy, gx), np.pi)
    orientation = np.histogram(angle[interior], 16, (0, np.pi), weights=magnitude[interior])[0]
    contrast = np.histogram(np.minimum(magnitude[interior], 0.25), 8, (0, 0.250001))[0]
    return _unit(np.r_[_unit(lbp), _unit(orientation), _unit(contrast)])


def extract_visual_parts(image_or_path):
    """Return three float32 vectors, per-feature validity, uint8 mask and diagnostics.

    Mask uses the resized, EXIF-oriented image coordinate system. ``valid`` is
    a dictionary keyed by feature name. ``features`` contains their vectors;
    feature names are also top-level keys for simple query callers.
    """
    image = _open(image_or_path)
    image.thumbnail((PARAMETERS["max_side"],) * 2, Image.Resampling.LANCZOS)
    rgb = np.asarray(image, dtype=np.float64) / 255
    mask, diagnostics = _foreground(rgb)
    accepted = diagnostics["status"] == "valid"
    features = {key: np.zeros(size, dtype=np.float32) for key, size in FEATURE_DIMENSIONS.items()}
    valid = {key: accepted for key in FEATURE_DIMENSIONS}
    if accepted:
        features["shape"] = _shape(mask)
        interior = ndimage.binary_erosion(mask, iterations=2)
        valid["color"] = bool(interior.sum() >= 64)
        valid["texture"] = bool(interior.sum() >= 128 and min(mask.shape) >= 16)
        if valid["color"]:
            features["color"] = _color(rgb, interior)
        if valid["texture"]:
            features["texture"] = _texture(rgb, mask)
    diagnostics.update(method=VERSION, image_size=list(image.size), valid=valid)
    return {**features, "features": features, "valid": valid, "validity": valid,
            "mask": mask.astype(np.uint8) * 255, "diagnostics": diagnostics}


def load_visual_parts(directory):
    """Load and verify derived artifact hashes before returning arrays and metadata."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    for name, digest in manifest["artifact_sha256"].items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Visual parts artifact hash mismatch: {name}")
    ids = json.loads((directory / "item_ids.json").read_text(encoding="utf-8"))
    with np.load(directory / "features.npz", allow_pickle=False) as loaded:
        arrays = {name: loaded[name] for name in loaded.files}
    for feature, dimension in FEATURE_DIMENSIONS.items():
        if arrays[feature].shape != (len(ids), dimension):
            raise ValueError(f"Invalid {feature} descriptor shape")
        if arrays[f"valid_{feature}"].shape != (len(ids),):
            raise ValueError(f"Invalid {feature} validity shape")
        if not np.isfinite(arrays[feature]).all():
            raise ValueError(f"Non-finite {feature} descriptors")
    return {"item_ids": ids, "manifest": manifest, "features": arrays, **arrays}
