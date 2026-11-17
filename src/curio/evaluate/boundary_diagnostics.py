"""Exact threshold diagnostics and dev-only fitting for set boundary experiments."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from curio.retrieve.condition_boundary import support_scores


def threshold_curve(scores, grades):
    """All distinct >= decisions, including empty; ignore unknowns, preserve ties."""
    scores, grades = np.asarray(scores, dtype=float), np.asarray(grades)
    if scores.ndim != 1 or grades.shape != scores.shape or not np.isfinite(scores).all():
        raise ValueError("Aligned finite score and grade vectors required")
    known = grades >= 0
    s, y = scores[known], grades[known] == 3
    if not len(s):
        return {
            "status": "no_known_labels",
            "points": [],
            "oracle": None,
            "average_precision": None,
        }
    order = np.argsort(-s, kind="stable")
    s, y = s[order], y[order]
    ends = np.r_[np.flatnonzero(s[:-1] != s[1:]), len(s) - 1]
    tp = np.r_[0, np.cumsum(y)[ends]]
    selected = np.r_[0, ends + 1]
    fp, fn = selected - tp, int(y.sum()) - tp
    thresholds = np.r_[np.nextafter(s[0], np.inf), s[ends]]
    points = []
    for t, a, b, c in zip(thresholds, tp, fp, fn):
        points.append(
            {
                "threshold": float(t),
                "tp": int(a),
                "fp": int(b),
                "fn": int(c),
                "precision": float(a / (a + b)) if a + b else None,
                "recall": float(a / (a + c)) if a + c else None,
                "f1": float(2 * a / (2 * a + b + c)) if 2 * a + b + c else 0.0,
            }
        )
    best = max(points, key=lambda p: (p["f1"], p["precision"] or 0, p["threshold"]))
    ap = float(np.sum(np.diff(tp) / y.sum() * tp[1:] / selected[1:])) if y.any() else None
    return {
        "status": "ok" if y.any() else "no_positives",
        "points": points,
        "oracle": best,
        "average_precision": ap,
        "known": len(s),
        "positive": int(y.sum()),
    }


def choose_threshold(scores, grades):
    """Exact macro-F1 maximization on supplied dev scores; ties prefer higher t."""
    scores, grades = np.asarray(scores, dtype=float), np.asarray(grades)
    if scores.ndim != 2 or scores.shape != grades.shape or not np.isfinite(scores).all():
        raise ValueError("Aligned finite item x query matrices required")
    known = grades >= 0
    if not known.any():
        raise ValueError("No known development labels")
    candidates = np.r_[np.unique(scores[known]), np.nextafter(scores[known].max(), np.inf)]
    positive = grades == 3
    best = (-1.0, 0.0)
    for t in candidates:
        yes = scores >= t
        tp = (yes & positive).sum(0)
        fp = (yes & known & ~positive).sum(0)
        fn = (~yes & positive).sum(0)
        den = 2 * tp + fp + fn
        f1 = np.divide(2 * tp, den, out=np.zeros_like(den, dtype=float), where=den > 0)
        candidate = (float(f1[known.any(0)].mean()), float(t))
        if candidate > best:
            best = candidate
    return best[1]


def fit_logistic(features, grades, *, partial_weight=1.0):
    x, g = np.asarray(features), np.asarray(grades)
    if x.shape[:-1] != g.shape or not np.isfinite(x).all():
        raise ValueError("Features and labels do not align")
    known = g >= 0
    x, g = x[known], g[known]
    if len(np.unique(g == 3)) != 2:
        raise ValueError("Training requires positive and negative known examples")
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(C=1.0, max_iter=2000).fit(
        scaler.transform(x), g == 3, sample_weight=np.where(g == 1, partial_weight, 1.0)
    )
    return {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coefficients": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
        "partial_weight": partial_weight,
        "C": 1.0,
        "score_semantics": "uncalibrated support",
    }


def crossfit_boundary(features, grades, groups, *, partial_weight=1.0):
    """Only development arrays enter this function; item groups never cross folds."""
    x, g, groups = np.asarray(features), np.asarray(grades), np.asarray(groups)
    if x.shape[:-1] != g.shape or len(groups) != len(x) or len(np.unique(groups)) < 3:
        raise ValueError("Need aligned dev data and at least three object groups")
    oof, folds = np.zeros_like(g, dtype=float), []
    for train, val in GroupKFold(n_splits=3).split(x, groups=groups):
        model = fit_logistic(x[train], g[train], partial_weight=partial_weight)
        oof[val] = support_scores(x[val].reshape(-1, x.shape[-1]), model).reshape(g[val].shape)
        folds.append(
            {
                "train_groups": sorted(set(groups[train])),
                "validation_groups": sorted(set(groups[val])),
            }
        )
    fitted = fit_logistic(x, g, partial_weight=partial_weight)
    fitted.update(
        threshold=choose_threshold(oof, g), threshold_source="dev grouped out-of-fold", folds=folds
    )
    return fitted
