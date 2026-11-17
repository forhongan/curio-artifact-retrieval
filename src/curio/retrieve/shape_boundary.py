"""Experimental shape evidence with constrained semantic directions and provenance.

Geometry describes the segmented image, not a guaranteed physical measurement.
Catalogue evidence may veto an explicit cup category, never create visual support.
"""

import math
import re

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize
from scipy.special import expit

from .joint_query import JointQuery

VERSION = "shape-evidence-v14"
PROBES = {
    "meiping": "a meiping (plum) vase with a small mouth, a short neck, high rounded shoulders and sides tapering toward the foot",
    "gourd": "one continuous bottle body has two swollen chambers stacked vertically, one above the other, joined by a narrow waist on the same vertical axis; not two vessels side by side, and not a single belly with a neck collar",
    "bowl": "an open bowl with a broad rim and a visibly deep concave basin, rather than a flat plate or a tall narrow drinking cup",
}
GEOMETRY_NAMES = (
    "mask_valid",
    "log_height_width",
    "widest_height",
    "top_width",
    "bottom_width",
    "upper_width",
    "middle_width",
    "lower_width",
    "vertical_waist_depth",
    "waist_height",
    "horizontal_center_variation",
    "area_fill",
)


def geometry_features(mask, valid=True):
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("Expected a two-dimensional mask")
    yy, xx = np.nonzero(mask)
    if not valid or len(xx) < 64:
        return [0.0] * len(GEOMETRY_NAMES)
    crop = mask[yy.min() : yy.max() + 1, xx.min() : xx.max() + 1]
    height, width = crop.shape
    raw, centers = [], []
    for row in crop:
        xs = np.flatnonzero(row)
        raw.append(float(xs[-1] - xs[0] + 1) if len(xs) else 0.0)
        centers.append(float((xs[-1] + xs[0]) / 2) if len(xs) else width / 2)
    profile = np.interp(np.linspace(0, height - 1, 64), np.arange(height), raw)
    profile = gaussian_filter1d(profile, 1.0) / width
    # Compare upper and lower body peaks across an interior constriction. A neck
    # above one belly alone does not create two large body peaks.
    waist_depth, waist_height = 0.0, 0.0
    for i in range(14, 46):
        above = float(profile[5 : i - 3].max())
        below = float(profile[i + 4 : 59].max())
        depth = max(0.0, min(above, below) - profile[i])
        if depth > waist_depth:
            waist_depth, waist_height = depth, i / 63
    values = [
        1.0,
        float(np.clip(np.log(height / width), -3, 3)),
        float(np.argmax(profile) / 63),
        float(profile[:7].mean()),
        float(profile[-7:].mean()),
        float(profile[7:23].mean()),
        float(profile[23:42].mean()),
        float(profile[42:59].mean()),
        waist_depth,
        waist_height,
        float(np.std(centers) / width),
        float(crop.mean()),
    ]
    return values


def extra_plan(atom):
    return JointQuery(PROBES[atom], False, (VERSION, atom), VERSION)


def feature_vector(old, extra, geometry):
    logits = [
        old["positive"]["yes_no_logit"],
        -old["counter"]["yes_no_logit"],
        extra["yes_no_logit"],
    ]
    x = np.r_[np.clip(logits, -30, 30), geometry].astype(float)
    if len(x) != 3 + len(GEOMETRY_NAMES) or not np.isfinite(x).all():
        raise ValueError("Invalid shape evidence")
    return x.tolist()


def fit_shape(x, y, regularization=2.0):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if (
        x.ndim != 2
        or x.shape[1] != 3 + len(GEOMETRY_NAMES)
        or len(x) != len(y)
        or not np.isfinite(x).all()
        or set(y.tolist()) != {0.0, 1.0}
        or not math.isfinite(regularization)
        or regularization <= 0
    ):
        raise ValueError("Invalid shape training data")
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale < 1e-8] = 1
    z = (x - mean) / scale
    weights = np.where(y == 1, len(y) / (2 * y.sum()), len(y) / (2 * (len(y) - y.sum())))

    def loss(theta):
        w, b = theta[:-1], theta[-1]
        logits = z @ w + b
        residual = weights * (expit(logits) - y)
        value = np.sum(weights * (np.logaddexp(0, logits) - y * logits))
        return value + 0.5 * regularization * (w @ w), np.r_[
            z.T @ residual + regularization * w, residual.sum()
        ]

    result = minimize(
        loss,
        np.zeros(x.shape[1] + 1),
        method="L-BFGS-B",
        jac=True,
        bounds=[(0.0, None)] * 3 + [(None, None)] * (len(GEOMETRY_NAMES) + 1),
        options={"maxiter": 2000, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError("Shape fit failed: " + result.message)
    return {
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "coefficient": result.x[:-1].tolist(),
        "intercept": float(result.x[-1]),
        "regularization": regularization,
    }


def predict_shape(head, features):
    x, w, scale = map(np.asarray, (features, head["coefficient"], head["scale"]))
    if (
        x.shape[-1] != len(w)
        or not np.isfinite(x).all()
        or not np.isfinite(w).all()
        or (scale <= 0).any()
        or (w[:3] < 0).any()
    ):
        raise ValueError("Invalid shape head or evidence")
    return expit(((x - head["mean"]) / scale) @ w + head["intercept"])


def catalogue_conflict(atom, item):
    """Limited category veto with quoted field evidence; never search description."""
    if atom != "bowl":
        return None
    for field in ("object_name", "title"):
        value = str(item.get(field) or "").strip()
        # Stop before decorative subjects, so 'Bowl with cups' is not a cup.
        head = re.split(
            r"\b(?:with|decorated|depicting|in the shape|in the form)\b", value.lower(), maxsplit=1
        )[0]
        if re.search(r"\bcups?\b|酒杯|茶杯|高足杯|耳杯", head) and not re.search(
            r"\bbowls?\b|碗", head
        ):
            return {"reason": "catalogue_cup_category", "field": field, "value": value}
    return None


def decide_shape(atom, head, features, extra, item):
    score = float(predict_shape(head, features))
    threshold = head["threshold"]
    if not math.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("Invalid shape threshold")
    conflict = catalogue_conflict(atom, item)
    topology_no = atom == "gourd" and extra["yes_no_logit"] <= 0
    visual_yes = score >= threshold and not topology_no
    return {
        "score": score,
        "threshold": threshold,
        "visual_yes": visual_yes,
        "yes": visual_yes and conflict is None,
        "reason": "catalogue_conflict"
        if conflict
        else "vertical_topology_rejected"
        if topology_no
        else "supported"
        if visual_yes
        else "insufficient_shape_support",
        "catalogue_evidence": conflict,
    }
