"""Conservative multi-evidence object isolation for museum photography.

The estimator uses only pixels.  It combines a robust border background model,
saliency, center geometry, connected components and GrabCut, then abstains when
the evidence does not support a single plausible object.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage

VERSION = "multi-evidence-bg-v4"
PARAMETERS = {
    "working_side": 256,
    "border_fraction": 0.04,
    "accepted_area": [0.025, 0.85],
    "accepted_dominant_component": 0.80,
    "accepted_center_distance": 0.68,
    "accepted_edge_fraction": 0.025,
    "accepted_internal_holes": 0.18,
}


def _open(image):
    if isinstance(image, (str, Path)):
        with Image.open(image) as loaded:
            return np.asarray(ImageOps.exif_transpose(loaded).convert("RGB"))
    return np.asarray(ImageOps.exif_transpose(image).convert("RGB"))


def _lab(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)


def _border_mask(shape, fraction):
    height, width = shape[:2]
    thickness = max(2, round(min(height, width) * fraction))
    border = np.zeros((height, width), dtype=bool)
    border[:thickness] = border[-thickness:] = True
    border[:, :thickness] = border[:, -thickness:] = True
    return border, thickness


def _background_probability(lab, border, cutoff=0.15):
    """Robust covariance model estimated from border; no object intrusion prior."""
    values = lab[border].astype(np.float64)
    center = np.median(values, axis=0)
    # Trim gross intrusions before estimating the background spread.
    delta = values - center
    scale = np.quantile(np.linalg.norm(delta, axis=1), 0.75) + 1e-3
    values = values[np.linalg.norm(delta, axis=1) <= 3 * scale]
    center = np.median(values, axis=0)
    covariance = np.cov(values, rowvar=False) + np.eye(3) * 2.0
    inverse = np.linalg.inv(covariance)
    difference = lab.reshape(-1, 3) - center
    distance2 = np.einsum("...i,ij,...j->...", difference, inverse, difference)
    probability = np.exp(-0.5 * np.clip(distance2, 0, 400)).reshape(lab.shape[:2])
    return probability, float(np.sqrt(np.max(np.diag(covariance))))


def _saliency(rgb):
    lab = _lab(rgb)
    values = cv2.GaussianBlur(lab, (0, 0), 4.0)
    mean = values.reshape(-1, 3).mean(axis=0)
    return np.linalg.norm(values - mean, axis=2)


def _refine(image_partial):
    cv2.setRNGSeed(0)
    try:
        cv2.grabCut(image_partial, None, None, None, None, 3, cv2.GC_INIT_WITH_MASK)
        return None
    except cv2.error:
        return None


def _proposal(rgb, bgprob, border):
    height, width = bgprob.shape
    yy, xx = np.mgrid[-1:1:complex(height), -1:1:complex(width)]
    center = np.exp(-1.7 * (xx * xx / 0.65 + yy * yy / 0.90))
    saliency = _saliency(rgb)
    saliency /= max(float(np.quantile(saliency, 0.98)), 1e-6)
    # Center prior changes ranking inside the candidate evidence; it must not
    # reject a broad textile whose foreground extends to the image margin.
    score = (1 - bgprob) * (0.85 + 0.15 * center) + 0.10 * saliency
    score[border] *= 0.15
    frame_fill = float(((bgprob < 0.25) & ~border).mean())
    planarlike = frame_fill > 0.40 and float((bgprob[~border] > 0.55).mean()) <= 0.42
    threshold = max(0.14, float(np.quantile(score, 0.78)))
    raw = score > threshold
    raw = ndimage.binary_opening(raw, iterations=1)
    raw = ndimage.binary_closing(raw, iterations=2)
    labels, count = ndimage.label(raw)
    if count:
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        raw = labels == int(np.argmax(sizes))
    raw = ndimage.binary_fill_holes(raw).astype(np.uint8)
    if not raw.any() or raw.sum() >= 0.96 * raw.size:
        return raw.astype(bool), score
    sure_fg = ndimage.binary_erosion(raw, iterations=3)
    sure_bg = ndimage.binary_dilation(raw, iterations=3) == 0
    grab = np.full((height, width), cv2.GC_PR_BGD, dtype=np.uint8)
    grab[sure_bg] = cv2.GC_BGD
    grab[sure_fg] = cv2.GC_FGD
    grab[raw.astype(bool) & ~sure_fg] = cv2.GC_PR_FGD
    cv2.setRNGSeed(0)
    try:
        cv2.grabCut(rgb, grab, None, None, None, 4, cv2.GC_INIT_WITH_MASK)
        refined = np.isin(grab, (cv2.GC_FGD, cv2.GC_PR_FGD))
        refined = ndimage.binary_closing(refined, iterations=1)
        refined = ndimage.binary_fill_holes(refined)
        labels, count = ndimage.label(refined)
        if count:
            sizes = np.bincount(labels.ravel())
            sizes[0] = 0
            refined = labels == int(np.argmax(sizes))
        if refined.sum() > 16:
            raw = refined.astype(np.uint8)
    except cv2.error:
        pass
    raw = ndimage.binary_fill_holes(raw)
    if planarlike:
        crop = _planar_crop(rgb, bgprob, border)
        if crop.mean() > 0.20:
            raw = crop
    return raw.astype(bool), score


def _fill_planar_evidence(mask, bgprob, rgb, border):
    """Recover coherent low-texture planes omitted by object-centered priors."""
    height, width = mask.shape
    # Interior region stops before the border; a planar textile may leave a small
    # studio margin while still reaching most of the image.
    interior = ndimage.binary_erosion(~border, iterations=4)
    lab = _lab(rgb)
    smooth = cv2.GaussianBlur(lab, (0, 0), 5.0)
    interior = ~border
    # A flat studio background has tightly concentrated border colors and a
    # substantial inlier region. Profile-based planar override is reserved for
    # frame-filling art where less than half the interior resembles the border.
    if float((bgprob[interior] > 0.55).mean()) > 0.42:
        return np.zeros_like(bgprob, dtype=bool)
    gradient = np.linalg.norm(ndimage.sobel(smooth, axis=0), axis=2) + np.linalg.norm(
        ndimage.sobel(smooth, axis=1), axis=2
    )
    # Boundary support from strong gradients and pixels unlike the border model.
    # A flat textile can be almost featureless: accept the modal interior color
    # region even if it resembles a neutral studio background.  Text on object
    # and textile colors remain differences captured by the same background
    # distribution; the outer border stays excluded.
    candidate = interior & ((bgprob < 0.82) | (gradient > np.quantile(gradient, 0.72)))
    candidate = ndimage.binary_closing(candidate, iterations=3)
    candidate = ndimage.binary_fill_holes(candidate)
    labels, count = ndimage.label(candidate)
    if not count:
        return mask
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    component = labels == int(np.argmax(sizes))
    occupied_bbox = 0.0
    if component.any():
        ys, xs = np.where(component)
        occupied_bbox = (xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1) / mask.size
    # Evidence for a whole plane: substantial bbox, multiple image sides reached,
    # and enough non-border background agreement around it.
    bbox_edges = (
        component[:4, :].mean() + component[-4:, :].mean()
        + component[:, :4].mean() + component[:, -4:].mean()
    )
    if occupied_bbox >= 0.35 and bbox_edges >= 0.45 and component.mean() >= 0.14:
        combined = mask | component
        labels, count = ndimage.label(combined)
        if count:
            sizes = np.bincount(labels.ravel())
            sizes[0] = 0
            combined = labels == int(np.argmax(sizes))
        return ndimage.binary_fill_holes(combined)
    return mask


def _archetype(rgb, mask, bgprob, border):
    """Decide object versus planar from evidence, not catalog object_type."""
    height, width = mask.shape
    ys, xs = np.where(mask)
    if not len(xs):
        return "ambiguous"
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    spans = ((x1 - x0) / width, (y1 - y0) / height)
    # A planar crop must be a substantial, rectangular area spanning both axes.
    # Ordinary bowls and mirrors can also occupy much of one axis, so requiring
    # both axes and high box occupancy avoids the common false override.
    # Do not infer a flat artwork from broad coverage alone: a large full-frame
    # sculptural object can satisfy this. Background-model quality is required.
    if _planar_evidence(rgb, mask, bgprob, border) and min(spans) >= 0.58:
        return "planar"
    return "object"


def _planar_bbox(rgb, bgprob, border):
    """Frozen low-cost planar candidate using robust border-colour evidence."""
    lab = _lab(rgb)
    gradient = np.linalg.norm(ndimage.sobel(cv2.GaussianBlur(lab, (0, 0), 4.0), axis=0), axis=2)
    gradient += np.linalg.norm(ndimage.sobel(cv2.GaussianBlur(lab, (0, 0), 4.0), axis=1), axis=2)
    support = (bgprob < 0.35) | (gradient > np.quantile(gradient, 0.55))
    support[border] = False
    rows = ndimage.gaussian_filter1d(support.mean(axis=1), 2)
    cols = ndimage.gaussian_filter1d(support.mean(axis=0), 2)
    row_hits = np.flatnonzero(rows > max(.10, np.quantile(rows, .35)))
    col_hits = np.flatnonzero(cols > max(.10, np.quantile(cols, .35)))
    if len(row_hits) < .18 * len(rows) or len(col_hits) < .18 * len(cols):
        return np.zeros_like(bgprob, dtype=bool)
    x0, x1 = int(col_hits.min()), int(col_hits.max() + 1)
    y0, y1 = int(row_hits.min()), int(row_hits.max() + 1)
    if (x1 - x0) < .20 * bgprob.shape[1] or (y1 - y0) < .18 * bgprob.shape[0]:
        return np.zeros_like(bgprob, dtype=bool)
    crop = np.zeros_like(bgprob, dtype=bool)
    crop[y0:y1, x0:x1] = True
    return crop


def _planar_evidence(rgb, object_mask, bgprob, border):
    if float((bgprob[~border] > 0.55).mean()) > 0.42:
        return False
    ys, xs = np.where(object_mask)
    if not len(xs):
        return False
    occupancy = object_mask.sum() / max(1, (xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1))
    return occupancy >= 0.78 and object_mask.mean() >= 0.18


def _planar_crop(rgb, bgprob, border):
    """Find a near-rectangular, full-frame textile boundary."""
    height, width = bgprob.shape
    # Working row/column profiles are robust to decorative detail and neutral
    # background because the border model contributes no evidence.
    lab = _lab(rgb)
    smooth = cv2.GaussianBlur(lab, (0, 0), 5.0)
    if float((bgprob[~border] > 0.55).mean()) > 0.42:
        return np.zeros_like(bgprob, dtype=bool)
    gradient = np.linalg.norm(ndimage.sobel(smooth, axis=0), axis=2) + np.linalg.norm(
        ndimage.sobel(smooth, axis=1), axis=2
    )
    evidence = ((bgprob < 0.12) | (gradient > np.quantile(gradient, 0.72))).astype(np.uint8)
    evidence[border] = 0
    row = evidence.mean(axis=1)
    col = evidence.mean(axis=0)
    row_smooth = ndimage.gaussian_filter1d(row.astype(float), 2)
    col_smooth = ndimage.gaussian_filter1d(col.astype(float), 2)
    # A planar image has a broad central run after smoothing, not isolated rows
    # and columns that merely contain decoration.
    def longest_run(values, threshold):
        active = values > threshold
        edges = np.diff(np.r_[False, active, False].astype(np.int8))
        starts = np.flatnonzero(edges == 1)
        ends = np.flatnonzero(edges == -1)
        if not len(starts):
            return None
        index = int(np.argmax(ends - starts))
        return starts[index], ends[index]

    yrun = longest_run(row_smooth, max(0.08, float(np.quantile(row_smooth, 0.12))))
    xrun = longest_run(col_smooth, max(0.08, float(np.quantile(col_smooth, 0.12))))
    if yrun is None or xrun is None:
        return np.zeros_like(bgprob, dtype=bool)
    ys = np.arange(*yrun)
    xs = np.arange(*xrun)
    if not len(xs) or not len(ys):
        return np.zeros_like(bgprob, dtype=bool)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    area = (x1 - x0) * (y1 - y0) / bgprob.size
    if area < 0.20 or area >= 0.97:
        return np.zeros_like(bgprob, dtype=bool)
    if (x1 - x0) < 0.25 * width or (y1 - y0) < 0.18 * height:
        return np.zeros_like(bgprob, dtype=bool)
    # Remove isolated abrupt bands that include studio pixels at a frame edge.
    margins = (x0, width - x1, y0, height - y1)
    if any(margins[i] > 0.42 * size for i, size in ((0, width), (1, width), (2, height), (3, height))):
        return np.zeros_like(bgprob, dtype=bool)
    crop = np.zeros_like(bgprob, dtype=np.uint8)
    crop[y0:y1, x0:x1] = 1
    return crop.astype(bool)


def isolate_object(image):
    rgb = _open(image)
    scale = PARAMETERS["working_side"] / max(rgb.shape[:2])
    if scale < 1:
        rgb = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    lab = _lab(rgb)
    border, _ = _border_mask(lab.shape, PARAMETERS["border_fraction"])
    bgprob, spread = _background_probability(lab, border)
    mask, score = _proposal(rgb, bgprob, border)
    # Retain the initial proposal for abstention and refinement diagnostics; the
    # classifier never needs a confident mask for an ambiguous source image.
    proposal_mask = mask.copy()
    # Planar material is a separate, explicit image archetype.  Mixing its fill
    # with an object mask creates a shape that genuinely describes neither.
    object_mask = mask.copy()
    planar_mask = _fill_planar_evidence(mask, bgprob, rgb, border)
    archetype = _archetype(rgb, object_mask, bgprob, border)
    planar_bbox = _planar_bbox(rgb, bgprob, border)
    # A very wide, shallow planar strip is not a centered object: choose the
    # explicit rectangle branch before an object proposal can collapse it.
    if planar_bbox.mean() > .20 and (
        planar_bbox.sum(axis=1).mean() > .65 * planar_bbox.shape[1]
        or planar_bbox.sum(axis=0).mean() > .65 * planar_bbox.shape[0]
    ):
        archetype = "planar"
    ys, xs = np.where(planar_mask)
    planar_change = float(np.count_nonzero(planar_mask ^ object_mask)) / max(1, planar_mask.size)
    if archetype == "planar" and planar_change > 0.05 and planar_mask.any():
        mask = _planar_crop(rgb, bgprob, border)
        if not mask.any():
            mask = planar_mask
    elif archetype == "planar" and planar_bbox.mean() > .20:
        mask = planar_bbox
    elif not np.any(mask):
        mask = score > max(0.14, float(np.quantile(score, 0.78)))
    else:
        mask = object_mask
    ys, xs = np.where(mask)
    area = float(mask.mean())
    if len(xs):
        bbox = [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]
        center_distance = float(np.hypot(
            (xs.mean() + .5) / mask.shape[1] - .5,
            (ys.mean() + .5) / mask.shape[0] - .5,
        ))
    else:
        bbox = [0, 0, 0, 0]
        center_distance = 10.0
    labels, count = ndimage.label(mask)
    sizes = np.bincount(labels.ravel())[1:] if count else np.array([])
    dominant = float(sizes.max() / max(1, sizes.sum())) if len(sizes) else 0.0
    edge_fraction = float(mask[border].mean())
    score_peak = float(np.quantile(score, 0.99))
    filled = ndimage.binary_fill_holes(mask)
    holes = float((filled.sum() - mask.sum()) / max(1, filled.sum()))
    reasons = []
    if not PARAMETERS["accepted_area"][0] <= area <= PARAMETERS["accepted_area"][1]:
        reasons.append("area")
    if dominant < PARAMETERS["accepted_dominant_component"]:
        reasons.append("fragmented")
    if center_distance > PARAMETERS["accepted_center_distance"] and area < 0.35:
        reasons.append("off_center")
    if edge_fraction > PARAMETERS["accepted_edge_fraction"]:
        reasons.append("touches_border")
    if holes > PARAMETERS["accepted_internal_holes"]:
        reasons.append("large_holes")
    if score_peak < 0.25:
        reasons.append("weak_evidence")
    status = "accepted" if not reasons else "abstain"
    # A rectangular crop exceeding 20% of the frame is evidence of a planar
    # object even when its color is close to the neutral studio background.
    planar_crop = _planar_crop(rgb, bgprob, border)
    if archetype == "planar" and planar_crop.mean() > 0.20 and area < 0.35:
        mask = planar_crop
        area = float(mask.mean())
        ys, xs = np.where(mask)
        bbox = [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]
        reasons = [reason for reason in reasons if reason != "area"]
        status = "accepted" if not reasons else "abstain"
    return {
        "mask": mask.astype(np.uint8) * 255,
        "mask_binary": mask,
        "score": score.astype(np.float32),
        "bbox": bbox,
        "status": status,
        "reasons": reasons,
        "diagnostics": {
            "method": VERSION,
            "archetype": archetype,
            "area": area,
            "dominant_component": dominant,
            "center_distance": center_distance,
            "border_fraction": edge_fraction,
            "hole_fraction": holes,
            "background_spread": spread,
            "score_peak": score_peak,
            "proposal_area": float(proposal_mask.mean()),
            "image_size": [int(rgb.shape[1]), int(rgb.shape[0])],
            "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    }
