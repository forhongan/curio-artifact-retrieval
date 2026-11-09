"""Text attributes scored against predictions made from pixels alone.

No gallery descriptions, catalog fields, or ground-truth labels are inputs to
this ranking module. Probabilities are uncalibrated discriminative scores.
"""

from __future__ import annotations

import re

import numpy as np
from scipy.special import expit

VOCAB = {
    "object_type": {
        "vase": ["vase", "瓶形", "瓶子"],
        "bowl": ["bowl", "碗"],
        "plate": ["plate", "盘"],
        "figure": ["figurine", "造像"],
        "cup": ["cup", "杯"],
        "textile": ["textile", "织物"],
    },
    "colors": {
        "blue": ["blue", "蓝"],
        "white": ["white", "白"],
        "green": ["green", "绿"],
        "red": ["red", "红"],
        "brown": ["brown", "褐"],
        "black": ["black", "黑"],
        "yellow": ["yellow", "黄"],
        "pink": ["pink", "粉红"],
    },
    "shape": {
        "narrow_mouth": ["narrow mouth", "narrow neck", "窄口", "细颈"],
        "wide_mouth": ["wide open mouth", "wide mouth", "敞口"],
        "handles": ["handles", "handle", "把手", "有耳"],
        "round_body": ["rounded body", "rounded vessel", "圆鼓"],
    },
}


def parse_attributes(text):
    """A fixed finite-vocabulary bridge, not an LLM or general text encoder."""
    text = text.lower().strip()
    atoms = []
    for task, terms in VOCAB.items():
        for label, aliases in terms.items():
            if any(
                (re.search(r"\b" + re.escape(a) + r"\b", text) if a.isascii() else a in text)
                for a in aliases
            ):
                wanted = 1
                if label == "handles" and any(
                    x in text
                    for x in (
                        "without visible handles",
                        "without handles",
                        "no handle",
                        "没有可见把手",
                        "无柄",
                    )
                ):
                    wanted = 0
                atoms.append((task, label, wanted))
    return atoms


def predict_heads(features, coefficients, intercepts):
    return expit(np.asarray(features) @ np.asarray(coefficients).T + intercepts)


def attribute_scores(probabilities, classes, atoms):
    """Mean log predicted probability; any unsupported atom rejects the query."""
    lookup = {key: i for i, key in enumerate(classes)}
    missing = [f"{task}/{label}" for task, label, _ in atoms if f"{task}/{label}" not in lookup]
    if missing or not atoms:
        return None, missing or ["no_supported_attributes"]
    terms = []
    for task, label, wanted in atoms:
        p = probabilities[:, lookup[f"{task}/{label}"]]
        terms.append(np.log(np.clip(p if wanted else 1 - p, 1e-7, 1)))
    return np.mean(terms, axis=0), []


def stable_order(scores, ids):
    return np.lexsort((np.asarray(ids, dtype=str), -np.asarray(scores)))


def rrf_scores(left, right, ids, right_weight=0.5):
    """Fixed-rank fusion; ranks are not probabilities or cosine values."""
    ranks = []
    for scores in (left, right):
        order = stable_order(scores, ids)
        rank = np.empty(len(ids), dtype=float)
        rank[order] = np.arange(1, len(ids) + 1)
        ranks.append(1 / (60 + rank))
    return (1 - right_weight) * ranks[0] + right_weight * ranks[1]


def cascade_order(coarse, fine, ids, k):
    """Recall by the cheap attribute head; rerank only those K candidates."""
    order = stable_order(coarse, ids)
    selected = order[: min(k, len(order))]
    reranked = selected[stable_order(fine[selected], np.asarray(ids)[selected])]
    return np.r_[reranked, order[len(selected) :]]
