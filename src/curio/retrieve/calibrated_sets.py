"""Development-fit set boundaries over image scores; no target text lookup."""

import hashlib
import json
from pathlib import Path
import re

import numpy as np
from scipy.special import expit

from curio.retrieve.attribute_cv import attribute_scores, parse_attributes


def decide(service, query, method, generic):
    root = Path(service.root)
    folder = root / "artifacts/relevance_v5"
    language = "zh" if re.search(r"[\u4e00-\u9fff]", query) else "en"
    config = json.loads(
        (folder / f"boundary_production_lora_{language}.json").read_text(encoding="utf-8")
    )
    if config["index_manifest_sha256"] != service.index_manifest_sha256:
        raise ValueError("Development boundary belongs to a different image index")
    raw = service._dense(query.strip())
    margin = raw - generic
    normalized = (raw - raw.mean()) / max(float(raw.std()), 1e-6)
    diagnostics = {
        "fit_split": config["fit_split"],
        "scope": "same-intent development fit, not correctness probability",
        "language": language,
        "raw_query_used": True,
        "index_manifest_sha256": config["index_manifest_sha256"],
    }
    thresholds = json.loads((folder / "thresholds.json").read_text(encoding="utf-8"))
    if method == "dev_calibrated":
        features = np.column_stack([raw, margin, normalized])
        decision = expit(features @ np.array(config["coefficients"]) + config["intercept"])
        threshold = config["threshold"]
        posterior = decision
    elif method in {"dev_cutoff", "dev_color"}:
        decision = raw.copy()
        key = f"production_lora/dense/{language}"
        if method == "dev_color":
            binding = json.loads((folder / "color_binding.json").read_text(encoding="utf-8"))
            directory = root / "artifacts/cv_research_20261104/fresh"
            for name, digest in binding["hashes"].items():
                if hashlib.sha256((directory / name).read_bytes()).hexdigest() != digest:
                    raise ValueError("Color head cache changed since evaluation")
            heads = np.load(directory / "raw_color_heads.npz")
            predictions = np.load(directory / "raw_color_predictions.npy")
            ids = json.loads(
                (root / "artifacts/classical_cv/item_ids.json").read_text(encoding="utf-8")
            )
            lookup = {key: i for i, key in enumerate(ids)}
            order = [lookup[str(r["item_id"])] for r in service.items]
            auxiliary, missing = attribute_scores(
                predictions[order], list(heads["classes"]), parse_attributes(query)
            )
            weight = binding["weight"]
            decision = normalized
            if auxiliary is not None:
                decision = decision + weight * (auxiliary - auxiliary.mean()) / max(
                    float(auxiliary.std()), 1e-6
                )
            diagnostics.update(
                auxiliary="train-only HSV attribute head",
                auxiliary_weight=weight,
                auxiliary_missing=missing,
            )
            key = f"production_lora/aux_raw_color/{language}"
        threshold = thresholds[key]
        posterior = np.zeros(len(raw))
    else:
        raise ValueError("Unknown development boundary method")
    diagnostics.update(
        status="development_boundary_applied",
        threshold=threshold,
        decision_score_semantics="logistic support"
        if method == "dev_calibrated"
        else "image-derived score",
    )
    return decision >= threshold, posterior, margin, diagnostics, raw
