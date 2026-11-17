"""Deterministic GrabCut foreground for surface-color research, with QA masks."""

import cv2
import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage

VERSION = "surface-grabcut-v13-384"


def extract_surface(path):
    with Image.open(path) as loaded:
        image = ImageOps.exif_transpose(loaded).convert("RGB")
        image.thumbnail((384, 384), Image.Resampling.LANCZOS)
    rgb = np.asarray(image).copy()
    height, width = rgb.shape[:2]
    if min(height, width) < 24:
        raise ValueError("Image too small for surface extraction")
    # Border-only initialization; neither catalogue text nor semantic labels seed
    # the foreground. Single-threaded execution makes the offline fit repeatable.
    cv2.setNumThreads(1)
    cv2.setRNGSeed(0)
    margin = max(2, round(min(height, width) * 0.02))
    mask = np.zeros((height, width), np.uint8)
    cv2.grabCut(
        rgb,
        mask,
        (margin, margin, width - 2 * margin, height - 2 * margin),
        np.zeros((1, 65), np.float64),
        np.zeros((1, 65), np.float64),
        5,
        cv2.GC_INIT_WITH_RECT,
    )
    foreground = (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)
    components, count = ndimage.label(foreground)
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    largest = components == int(np.argmax(sizes)) if count else foreground
    mask = ndimage.binary_fill_holes(largest)
    interior = ndimage.binary_erosion(mask, iterations=2)
    fraction = float(mask.mean())
    valid = bool(0.015 <= fraction <= 0.90 and interior.sum() >= 64)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(float)
    h, s, v = hsv[..., 0] * 2, hsv[..., 1] / 255, hsv[..., 2] / 255
    # Use all foreground area as denominator: a tiny red motif on white should
    # not become a predominantly red object after discarding neutral pixels.
    chromatic = interior & (s >= 0.18) & (v >= 0.12)
    hue = np.histogram(h[chromatic], bins=24, range=(0, 360))[0] / max(1, interior.sum())
    saturation = np.histogram(s[interior], bins=6, range=(0, 1.00001))[0] / max(1, interior.sum())
    value = np.histogram(v[interior], bins=6, range=(0, 1.00001))[0] / max(1, interior.sum())
    features = np.r_[hue, saturation, value].astype(float)
    if not valid:
        features[:] = 0
    return {
        "version": VERSION,
        "features": features.tolist(),
        "valid": valid,
        "foreground_fraction": fraction,
        "rgb": rgb,
        "mask": mask,
        "diagnostic": "area_checked_only; mask quality still requires visual QA",
    }
