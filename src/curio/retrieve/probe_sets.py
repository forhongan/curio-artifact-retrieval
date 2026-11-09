"""Image-only yes/no set search backed by calibrated catalogue probes.

Catalogue fields supervise and evaluate classifiers, but candidate catalogue
text is not a ranking input.  This service is a research boundary mode; it is
separate from the production text-to-image ranking default.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import numpy as np
from scipy.special import expit

from curio.query import parse_query
from curio.query.ontology import ERAS, VOCAB
from curio.query.parser import term_pattern
from curio.evaluate.catalogue_sets import linear_probabilities


DOMAIN_TO_PROBE = {
    "form": "object_type",
    "fine_material": "material",
    "broad_material": "broad_material",
    "era": "era",
    "motif": "motifs",
    "color": "colors",
    "technique": "technique",
}
PROBE_TO_DOMAIN = {value: key for key, value in DOMAIN_TO_PROBE.items()}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ProbeSetService:
    """Load dev-frozen probes and make bounded yes/no set decisions."""

    def __init__(self, items: list[dict], embeddings: np.ndarray, *, probes_path: Path,
                 set_report_path: Path, index_sha256: str | None = None):
        self.items = items
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self.probes_path = Path(probes_path)
        self.set_report_path = Path(set_report_path)
        self.report = json.loads(self.set_report_path.read_text(encoding="utf-8"))
        self.probe_report = json.loads(
            (self.probes_path / "report.json").read_text(encoding="utf-8")
        )
        self.fixed_policy = self.report.get("version") == "catalogue-sets-v5"
        if not self.fixed_policy and self.report.get("version") != "catalogue-probe-sets-v4":
            raise ValueError("Unsupported set report version")
        self.report_hash_key = "head_report_sha256" if self.fixed_policy else "probe_report_sha256"
        if self.report.get(self.report_hash_key) != _sha(
            self.probes_path / "report.json"
        ):
            raise ValueError("Set report does not match probe report")
        if index_sha256 and self.probe_report.get("index_sha256") != index_sha256:
            raise ValueError("Probe set report does not match active image index")
        if self.embeddings.ndim != 2 or len(self.embeddings) != len(items):
            raise ValueError("Probe set embeddings must cover the full catalogue")
        self._probabilities: dict[str, np.ndarray] = {}
        self._classes: dict[str, list[str]] = {}
        self._thresholds: dict[tuple[str, str], float] = {}
        for task, task_report in self.probe_report.get("tasks", {}).items():
            if self.fixed_policy:
                if task_report.get("status") != "trained":
                    continue
                artifact = self.probes_path / task_report["artifact"]
                if _sha(artifact) != task_report["artifact_sha256"]:
                    raise ValueError(f"Head artifact hash mismatch: {artifact.name}")
                task = DOMAIN_TO_PROBE[task]
                classes = list(task_report["classes"])
                with np.load(artifact, allow_pickle=False) as model:
                    probabilities = linear_probabilities(
                        self.embeddings, model["weights"], model["bias"], len(classes)
                    )
                self._classes[task] = classes
                self._probabilities[task] = probabilities.astype(np.float32)
                for value, threshold in task_report["thresholds"].items():
                    self._thresholds[(task, value)] = float(threshold)
                continue
            classes = list(task_report.get("models", {}))
            if not classes:
                continue
            columns = []
            for value in classes:
                model_report = task_report["models"][value]
                artifact = self.probes_path / model_report["artifact"]
                expected = self.probe_report.get("artifacts_sha256", {}).get(artifact.name)
                if expected and _sha(artifact) != expected:
                    raise ValueError(f"Probe artifact hash mismatch: {artifact.name}")
                with np.load(artifact, allow_pickle=False) as model:
                    weights, bias = model["weights"], model["bias"]
                    probabilities = expit(self.embeddings @ weights[-1] + bias[-1])
                columns.append(probabilities)
                self._thresholds[(task, value)] = float(model_report["selected_threshold"])
            self._classes[task] = classes
            self._probabilities[task] = np.stack(columns, axis=1).astype(np.float32)
        gate = self.report.get("gate", {})
        self.gate_threshold = float(gate.get("threshold", 1.))
        self.require_class_winner = bool(gate.get("require_class_winner", False))
        self.by_id = {str(item["item_id"]): item for item in items}
        self.item_ids = [str(item["item_id"]) for item in items]

    @staticmethod
    def _material_task(value: str) -> str:
        return "broad_material" if value in {
            "ceramic", "metal", "stone", "textile", "wood", "glass", "lacquer", "ink"
        } else "material"

    def _atoms(self, query: str) -> tuple[list[tuple[str, str, str]], list[dict], bool]:
        parsed = parse_query(query)
        atoms: list[tuple[str, str, str]] = []
        missing: list[dict] = []
        for value in parsed.slots.get("object_type", []):
            if value == "plate" and value not in self._classes.get("object_type", []) and "dish" in self._classes.get("object_type", []):
                value = "dish"
            atoms.append(("object_type", "form", value))
        for value in parsed.slots.get("material", []):
            task = self._material_task(value)
            atoms.append((task, PROBE_TO_DOMAIN[task], value))
        for value in parsed.slots.get("motifs", []):
            if value == "figure_scene":
                value = "figures"
            atoms.append(("motifs", "motif", value))
        for value in parsed.slots.get("colors", []):
            atoms.append(("colors", "color", value))
        if parsed.era_label:
            atoms.append(("era", "era", parsed.era_label.lower()))
        techniques = {
            "embroidered": ("embroidered", "embroidery", "刺绣"),
            "carved": ("carved", "雕刻"), "weave": ("woven", "weave", "编织"),
        }
        for value, aliases in techniques.items():
            if any(re.search(term_pattern(alias), parsed.positive_text, re.I) for alias in aliases):
                atoms.append(("technique", "technique", value))
        # A parsed shape is a real constraint even though there is no trained
        # probe.  Refuse the query instead of silently dropping it.
        for value in parsed.slots.get("shape", []):
            missing.append({"domain": "shape", "value": value, "reason": "probe_not_trained"})
        for task, domain, value in atoms:
            if value not in self._classes.get(task, []):
                missing.append({"domain": domain, "value": value, "reason": "probe_class_unsupported"})
        if parsed.negations:
            missing.append({"domain": "negation", "value": "query", "reason": "negative_set_not_evaluated"})
        for value in parsed.unsupported:
            missing.append({"domain": value, "value": "query", "reason": "constraint_not_implemented"})
        if self.fixed_policy:
            residual = parsed.positive_text.lower()
            aliases = [alias for values in VOCAB.values() for terms in values.values() for alias in terms]
            aliases += [alias for _, _, terms in ERAS.values() for alias in terms]
            aliases += [alias for terms in techniques.values() for alias in terms]
            for alias in sorted(set(aliases), key=len, reverse=True):
                residual = re.sub(term_pattern(alias), " ", residual, flags=re.I)
            residual = re.sub(r"\b(?:a|an|the|photo|photograph|picture|image|of|with|and|artifact|artifacts|object|objects|show|me|find|some)\b", " ", residual)
            residual = re.sub(r"[\s,.;:!?，。；：！？的有带器]+", "", residual)
            if residual:
                missing.append({"domain": "unparsed", "value": residual, "reason": "query_contains_unhandled_constraints"})
        atoms = list(dict.fromkeys(atoms))
        supported = bool(atoms) and not missing
        return atoms, missing, supported

    def search(self, query: str, *, top_k: int = 20) -> dict:
        atoms, missing, supported = self._atoms(query)
        if not supported:
            raise ValueError({
                "message": "Query cannot be evaluated by the calibrated set probes",
                "missing": missing,
            })
        passes = []
        probabilities = []
        for task, _, value in atoms:
            index = self._classes[task].index(value)
            probability = self._probabilities[task][:, index]
            threshold = self._thresholds[(task, value)]
            probabilities.append(probability)
            passes.append(probability >= threshold)
        matrix = np.asarray(passes, dtype=bool)
        probability_matrix = np.asarray(probabilities, dtype=np.float32).T
        pass_fraction = matrix.mean(axis=0)
        gate_score = float(pass_fraction.max()) if len(pass_fraction) else 0.0
        gate_open = True if self.fixed_policy else gate_score >= self.gate_threshold
        accepted = matrix.all(axis=0) & gate_open
        if self.require_class_winner:
            for domain in dict.fromkeys(atom[1] for atom in atoms):
                task = next(atom[0] for atom in atoms if atom[1] == domain)
                requested = [atom[2] for atom in atoms if atom[1] == domain]
                accepted &= (self._probabilities[task].argmax(axis=1)
                             == self._classes[task].index(requested[0]))
        indices = np.flatnonzero(accepted)
        # This is a margin ordering only inside the yes set; it is not the
        # rejected-vs-accepted queue used by the production retrieval path.
        margins = np.min(
            np.stack([probability_matrix[:, j] - self._thresholds[(task, value)]
                      for j, (task, _, value) in enumerate(atoms)], axis=1),
            axis=1,
        )
        indices = sorted(indices, key=lambda i: (-float(margins[i]), self.item_ids[i]))
        preview = []
        for index in indices[:top_k]:
            item_id = self.item_ids[int(index)]
            item = self.by_id[item_id]
            preview.append({
                "item_id": item_id, "yes": True, "title": item.get("title", ""),
                "thumb_url": f"/api/images/{item_id}?thumb=true",
                "image_url": f"/api/images/{item_id}",
                "support_margin": round(float(margins[index]), 6),
                "atoms": [
                    {"task": task, "value": value,
                     "probability": round(float(probability_matrix[index, j]), 6),
                     "threshold": round(float(self._thresholds[(task, value)]), 6),
                     "yes": bool(matrix[j, index])}
                    for j, (task, _, value) in enumerate(atoms)
                ],
            })
        return {
            "query": query, "variant": "probe_yes_set",
            "retrieval_mode": "image_probes", "matched_count": int(accepted.sum()),
            "rejected_count": int(len(self.items) - accepted.sum()),
            "decision_rule": "intersection of all requested visual predicates",
            "gate": None if self.fixed_policy else {
                "open": bool(gate_open), "score": round(gate_score, 6),
                "threshold": round(self.gate_threshold, 6)},
            "no_match": int(not accepted.any()),
            "atoms": [{"task": task, "domain": domain, "value": value,
                       "threshold": round(float(self._thresholds[(task, value)]), 6),
                       "source": "fixed engineering probability floor" if self.fixed_policy else "dev-frozen visual probe threshold"}
                      for task, domain, value in atoms],
            "matched_item_ids": [self.item_ids[int(index)] for index in indices],
            "results": preview, "returned": len(preview),
            "research_only": True,
            "warnings": [
                "This is an experimental image-only yes/no set, not a manually verified boundary.",
                "Missing or ambiguous catalogue labels are unknown, never negatives.",
                "Candidate catalogue text is used only after selection for display.",
            ],
            "provenance": {
                "set_policy_sha256": _sha(self.set_report_path),
                "labels_sha256": self.report.get("labels_sha256"),
                "probe_report": self.report.get(self.report_hash_key),
                "image_index": self.probe_report.get("index_sha256"),
            },
        }
