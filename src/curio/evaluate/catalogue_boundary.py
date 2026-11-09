"""Explicit yes-set evaluation for catalogue-grounded visual attributes.

The query specification is built from development-split catalogue labels before
the test split is scored.  Candidate truth is tri-state: a known mismatch is a
negative, while a missing or ambiguous catalogue facet is excluded from the
explicit-set gallery instead of being guessed as a negative.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import product

import numpy as np

from curio.evaluate.catalogue_tasks import single_label


@dataclass(frozen=True)
class QuerySpec:
    qid: str
    text: str
    atoms: tuple[tuple[str, str], ...]
    family: str
    dev_positives: int

    def as_dict(self) -> dict:
        return {
            "qid": self.qid,
            "text": self.text,
            "atoms": [list(atom) for atom in self.atoms],
            "family": self.family,
            "dev_positives": self.dev_positives,
        }


def query_text(atoms: tuple[tuple[str, str], ...]) -> str:
    """Render a catalogue constraint in the English query style used by CLIP."""
    values = dict(atoms)
    material = values.get("fine_material", values.get("broad_material", ""))
    noun = values.get("form", "artifact")
    descriptors = []
    for key in ("era", "material", "motif", "technique", "color"):
        if key in values:
            descriptors.append(values[key])
    if material and material not in descriptors:
        descriptors.insert(0, material)
    phrase = " ".join([*descriptors, noun]).strip()
    return f"a photograph of a {phrase}"


def truth_for_atoms(row: dict, atoms: tuple[tuple[str, str], ...]) -> int:
    """Return 1 positive, 0 known negative, or -1 unknown."""
    unknown = False
    for domain, value in atoms:
        label = single_label(row, domain)
        if label is None:
            unknown = True
            continue
        if label != value:
            return 0
    return -1 if unknown else 1


def build_query_specs(rows: list[dict], *, max_combinations: int = 36,
                      max_no_match: int = 12) -> list[QuerySpec]:
    """Build a deterministic development-only query suite.

    No test labels are consulted here.  Single constraints use every class with
    at least one development positive.  Combination negatives are selected as
    the least frequent form/material/motif/era conjunctions in development, so
    they stay balanced and are not chosen by their test outcome.
    """
    dev = [row for row in rows if row["split"] == "dev"]
    positives: dict[tuple[str, str], int] = Counter()
    for row in dev:
        for domain in ("form", "fine_material", "broad_material", "era", "motif", "technique"):
            label = single_label(row, domain)
            if label is not None:
                positives[(domain, label)] += 1

    specs: list[QuerySpec] = []
    for (domain, label), count in sorted(positives.items()):
        atoms = ((domain, label),)
        specs.append(QuerySpec(
            qid=f"single_{domain}_{label.replace(' ', '_')}",
            text=query_text(atoms), atoms=atoms, family=f"single_{domain}",
            dev_positives=count,
        ))

    pair_domains = (
        ("form", "fine_material"),
        ("form", "era"),
        ("form", "motif"),
        ("fine_material", "era"),
        ("fine_material", "motif"),
        ("era", "motif"),
    )
    labels = {
        domain: sorted(label for (current, label), n in positives.items()
                      if current == domain and n >= 2)
        for domain in {domain for domain, _ in positives}
    }
    candidates: list[tuple[int, QuerySpec]] = []
    for left, right in pair_domains:
        for left_value, right_value in product(labels.get(left, []), labels.get(right, [])):
            atoms = ((left, left_value), (right, right_value))
            count = sum(truth_for_atoms(row, atoms) == 1 for row in dev)
            if count <= 0:
                continue
            candidates.append((count, QuerySpec(
                qid=f"pair_{left}_{left_value}_and_{right}_{right_value}".replace(" ", "_"),
                text=query_text(atoms), atoms=atoms, family=f"pair_{left}_{right}",
                dev_positives=count,
            )))
    # Always prefer constraints with a meaningful dev positive set, then expose
    # diverse pairs rather than letting one large class dominate all queries.
    candidates.sort(key=lambda item: (-item[0], item[1].qid))
    per_family: Counter[str] = Counter()
    for count, spec in candidates:
        if count < 2 or per_family[spec.family] >= 6:
            continue
        specs.append(spec)
        per_family[spec.family] += 1
        if len(specs) >= len(positives) + max_combinations:
            break

    # The no-match controls are absent from dev but have at least ten wholly
    # known negatives.  They therefore test the decision to return an empty set.
    absent: list[tuple[int, QuerySpec]] = []
    for left, right in pair_domains:
        for left_value, right_value in product(labels.get(left, []), labels.get(right, [])):
            atoms = ((left, left_value), (right, right_value))
            truths = [truth_for_atoms(row, atoms) for row in dev]
            if any(value == 1 for value in truths):
                continue
            known_negatives = sum(value == 0 for value in truths)
            if known_negatives < 10:
                continue
            absent.append((known_negatives, QuerySpec(
                qid=f"nomatch_{left}_{left_value}_and_{right}_{right_value}".replace(" ", "_"),
                text=query_text(atoms), atoms=atoms, family=f"nomatch_{left}_{right}",
                dev_positives=0,
            )))
    absent.sort(key=lambda item: (item[0], item[1].qid))
    seen_families: Counter[str] = Counter()
    for _, spec in absent:
        if seen_families[spec.family] >= 2:
            continue
        specs.append(spec)
        seen_families[spec.family] += 1
        if sum(spec.family.startswith("nomatch_") for spec in specs) >= max_no_match:
            break
    return specs


def query_dataset(rows: list[dict], spec: QuerySpec, split: str) -> dict:
    global_indices = np.array([index for index, row in enumerate(rows)
                               if row["split"] == split], dtype=np.int64)
    items = [rows[int(index)] for index in global_indices]
    truth = np.array([truth_for_atoms(row, spec.atoms) for row in items], dtype=np.int8)
    keep = np.flatnonzero(truth >= 0)
    ids = [items[int(index)]["item_id"] for index in keep]
    indices = global_indices[keep]
    relevant = {item_id for item_id, value in zip(ids, truth[keep]) if value == 1}
    return {
        "indices": indices,
        "ids": ids,
        "truth": truth[keep].astype(bool),
        "relevant": relevant,
        "unknown_excluded": int((truth == -1).sum()),
    }


def explicit_metrics(selected: set[str], relevant: set[str]) -> dict:
    tp = len(selected & relevant)
    fp = len(selected - relevant)
    fn = len(relevant - selected)
    precision = tp / len(selected) if selected else 0.0
    recall = tp / len(relevant) if relevant else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    no_match = not relevant
    return {
        "relevant_size": len(relevant),
        "selected_size": len(selected),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_set_match": int(selected == relevant),
        "empty_prediction": int(not selected),
        "no_match": int(no_match),
        "no_match_correct": int(no_match and not selected),
    }


def aggregate_query_metrics(rows: list[dict]) -> dict:
    positive = [row for row in rows if not row["no_match"]]
    no_match = [row for row in rows if row["no_match"]]
    return {
        "queries": len(rows),
        "positive_queries": len(positive),
        "no_match_queries": len(no_match),
        "macro_precision": float(np.mean([row["precision"] for row in positive])) if positive else 0.0,
        "macro_recall": float(np.mean([row["recall"] for row in positive])) if positive else 0.0,
        "macro_f1": float(np.mean([row["f1"] for row in positive])) if positive else 0.0,
        "exact_set_accuracy": float(np.mean([row["exact_set_match"] for row in rows])),
        "no_match_accuracy": float(np.mean([row["no_match_correct"] for row in no_match])) if no_match else 0.0,
        "empty_rate_positive_queries": float(np.mean([row["empty_prediction"] for row in positive])) if positive else 0.0,
        "macro_set_score": float(np.mean([
            row["f1"] if not row["no_match"] else row["no_match_correct"] for row in rows
        ])),
    }


def rank_percentile(scores: np.ndarray) -> np.ndarray:
    """Convert a candidate score vector to empirical percentiles in [0, 1]."""
    values = np.asarray(scores, dtype=np.float64).ravel()
    if values.size == 0:
        return values
    if values.size == 1:
        return np.ones_like(values)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64) / (values.size - 1)
    return ranks
