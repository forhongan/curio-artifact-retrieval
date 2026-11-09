"""Catalogue-grounded, mutually exclusive image classification tasks.

The catalogue is supervision, never a feature. Ambiguous/composite labels are
retained as unknown instead of manufacturing negatives from incomplete prose.
"""

from __future__ import annotations

from collections import Counter

import numpy as np


PARENTS = {
    "porcelain": "ceramic", "earthenware": "ceramic", "stoneware": "ceramic",
    "bronze": "metal", "copper": "metal", "iron": "metal", "silver": "metal",
    "gold": "metal", "silk": "textile", "cotton": "textile", "satin": "textile",
    "brocade": "textile", "jade": "stone", "agate": "stone", "marble": "stone",
    "rock crystal": "stone", "crystal": "stone", "sandalwood": "wood",
    "boxwood": "wood", "bamboo": "wood",
}
TASKS = ("form", "broad_material", "fine_material", "era", "motif", "technique", "color")


def single_label(row: dict, domain: str) -> str | None:
    ontology = row["derived_ontology"]
    if domain == "form":
        # These are ontology ancestors, not disjoint alternatives.
        values = set(ontology.get("object_type", []))
        if "bottle vase" in values:
            values -= {"bottle vase", "bottle"}
            values.add("vase")
        if "bottle" in values and "vase" in values:
            return None
    elif domain in {"fine_material", "broad_material"}:
        values = set(ontology.get("material", []))
        if domain == "broad_material":
            values = {PARENTS.get(value, value) for value in values}
        else:
            values -= {PARENTS[value] for value in values if value in PARENTS}
            # A generic class cannot adjudicate a fine subtype: e.g. a known
            # ceramic may still be porcelain. Exclude such incomplete labels.
            values -= {"ceramic", "metal", "stone", "textile"}
            broad = set(ontology.get("material", []))
            if not values and broad:
                return None
    elif domain == "era":
        values = set(ontology.get("era", []))
    elif domain == "motif":
        values = set(ontology.get("motifs", []))
    elif domain == "technique":
        values = {
            facet["value"] for facet in ontology.get("technique_facets", [])
            if facet.get("value") and facet.get("status") == "catalogued_technique"
        }
    elif domain == "color":
        values = set(ontology.get("colors", []))
    else:
        raise ValueError(domain)
    return next(iter(values)) if len(values) == 1 else None


def task_manifest(rows: list[dict], domain: str, min_train: int = 20) -> dict:
    labels = [single_label(row, domain) for row in rows]
    train = Counter(label for row, label in zip(rows, labels) if row["split"] == "train" and label)
    classes = sorted(label for label, n in train.items() if n >= min_train)
    split_counts = {
        split: Counter(label for row, label in zip(rows, labels)
                       if row["split"] == split and label in classes)
        for split in ("train", "dev", "test")
    }
    # Evaluation requires each exact class to have train support.  Dev/test
    # are permitted to omit a class; the global classes are still scored with a
    # documented zero-support value, while per-class support remains visible.
    evaluable = bool(classes) and all(
        split_counts["train"][label] >= min_train for label in classes
    ) and bool(split_counts["dev"]) and bool(split_counts["test"])
    return {
        "domain": domain,
        "classes": classes,
        "labels": labels,
        "train_counts": dict(sorted(train.items())),
        "split_class_counts": {split: dict(sorted(counts.items()))
                               for split, counts in split_counts.items()},
        "status": "evaluable" if evaluable else "insufficient_support_for_three_split_classification",
        "coverage": {
            split: {
                "total": sum(row["split"] == split for row in rows),
                "known": sum(row["split"] == split and label is not None for row, label in zip(rows, labels)),
                "supported": sum(row["split"] == split and label in classes for row, label in zip(rows, labels)),
            }
            for split in ("train", "dev", "test")
        },
    }


def conjunction_truth(rows: list[dict], atoms: list[tuple[str, str]]) -> np.ndarray:
    """Tri-state AND: a known mismatch is false even if another facet is unknown."""
    result = []
    for row in rows:
        labels = [single_label(row, domain) for domain, _ in atoms]
        if any(label is not None and label != value for label, (_, value) in zip(labels, atoms)):
            result.append(0)
        elif any(label is None for label in labels):
            result.append(-1)
        else:
            result.append(1)
    return np.asarray(result, dtype=np.int8)


def query_text(atoms: list[tuple[str, str]]) -> str:
    values = dict(atoms)
    parts = []
    if "era" in values:
        parts.append(f"from the {values['era'].title()} dynasty")
    material = values.get("fine_material", values.get("broad_material", ""))
    noun = values.get("form", "artifact")
    descriptors = [material] if material else []
    descriptors.extend(values[key] for key in ("motif", "technique", "color") if key in values)
    phrase = " ".join([*descriptors, noun])
    return " ".join([f"a {phrase}", *parts])
