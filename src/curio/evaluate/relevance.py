"""Explicit conjunctive relevance with unknowns and compatible form families.

This module evaluates saved judgments. It is not part of retrieval inference.
"""

from __future__ import annotations

import numpy as np


FORM_FAMILIES = {
    "bottle_vase": {"vase", "bottle"},
    "bowl": {"bowl", "waterpot"},
    "plate_dish": {"plate", "dish", "saucer", "basin"},
    "pouring_vessel": {"ewer", "teapot"},
    "textile": {"textile", "robe", "jacket", "skirt", "shawl", "apron", "pendant"},
}


def condition_state(facts, condition):
    field, value = condition
    forms = facts.get("form")
    forms = [forms] if isinstance(forms, str) else forms
    if field == "form_family":
        if not forms:
            return "unknown"
        return "yes" if set(forms) & FORM_FAMILIES[value] else "no"
    if field in {"form", "exact_form"}:
        return "unknown" if not forms else "yes" if value in forms else "no"
    if field == "tag":
        if value in facts["visible_tags"]:
            return "yes"
        unknown = facts.get("unknown_tag_families", [])
        family = (
            "color"
            if value in {"green", "red", "blue", "white", "blue_white"}
            else "motif"
            if value in {"floral", "figures", "animal", "dragon", "lotus_motif"}
            else "shape"
        )
        return "unknown" if family in unknown else "no"
    if field == "era" and value == "Qianlong" and "qianlong_period_state" in facts:
        return facts["qianlong_period_state"]
    values = facts.get(field)
    if values is None or values == []:
        return "unknown"
    if field == "era" and value in values and len(values) > 1:
        return "unknown"
    if field == "material" and value not in values:
        parent = {
            "bronze": "metal",
            "brass": "metal",
            "copper": "metal",
            "porcelain": "ceramic",
            "earthenware": "ceramic",
            "stoneware": "ceramic",
            "jade": "stone",
            "silk": "textile",
        }
        if parent.get(value) in values:
            return "unknown"
    return "yes" if value in values else "no"


def judge(facts, conditions):
    if not conditions:
        return {"status": "unknown", "grade": -1, "condition_states": []}
    states = [condition_state(facts, c) for c in conditions]
    # A confirmed contradiction resolves a conjunction even if other atoms are unknown.
    if "no" in states:
        status = "partial" if "yes" in states else "irrelevant"
    elif "unknown" in states:
        status = "unknown"
    else:
        status = "relevant"
    return {
        "status": status,
        "grade": {"relevant": 3, "partial": 1, "irrelevant": 0, "unknown": -1}[status],
        "condition_states": states,
    }


def retrieval_metrics(scores, grades, accepted=None):
    scores, grades = np.asarray(scores), np.asarray(grades)
    known = grades >= 0
    order_all = np.argsort(-scores, kind="stable")
    order = order_all[known[order_all]]
    positive = grades == 3
    rel = positive[order]
    npos = int(positive.sum())
    gains = 2.0 ** grades[order[:10]] - 1.0
    discounts = np.log2(np.arange(len(gains)) + 2)
    ideal = np.sort(grades[known])[::-1][:10]
    idcg = float(((2.0**ideal - 1) / np.log2(np.arange(len(ideal)) + 2)).sum())
    out = {
        "positive_count": npos,
        "unknown_count": int((~known).sum()),
        "judged_fraction": float(known.mean()),
        "top10_judged_fraction": float(known[order_all[:10]].mean()),
        "ndcg@10": float((gains / discounts).sum() / idcg) if idcg else 0.0,
        "p@5_known": float(rel[:5].sum() / min(5, len(rel))) if len(rel) else 0.0,
        "recall@20_known": float(rel[:20].sum() / npos) if npos else None,
        "ap_known": float((np.cumsum(rel) / np.arange(1, len(rel) + 1) * rel).sum() / npos)
        if npos
        else None,
    }
    if accepted is not None:
        accepted = np.asarray(accepted, dtype=bool)
        tp = int((accepted & positive).sum())
        fp = int((accepted & known & ~positive).sum())
        fn = int((~accepted & positive).sum())
        up = int((accepted & ~known).sum())
        total = int(accepted.sum())
        out.update(
            tp=tp,
            fp=fp,
            fn=fn,
            selected_count=total,
            unknown_selected=up,
            set_precision=tp / (tp + fp) if tp + fp else None,
            set_recall=tp / npos if npos else None,
            set_f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            precision_lower=tp / total if total else None,
            precision_upper=(tp + up) / total if total else None,
        )
    return out
