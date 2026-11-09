"""Evaluate explicit yes/no sets grounded in museum catalogue facets.

Development labels define the query suite and freeze every threshold/weight.
The test split is scored once after that selection.  Missing catalogue fields
remain unknown; only known negatives and positives enter the set metric.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from curio.evaluate.catalogue_boundary import (  # noqa: E402
    QuerySpec,
    aggregate_query_metrics,
    build_query_specs,
    explicit_metrics,
    query_dataset,
    rank_percentile,
)
from curio.evaluate.catalogue_tasks import single_label  # noqa: E402
from curio.represent.encoders import Encoder  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_dataset(labels_path: Path) -> tuple[list[dict], np.ndarray, np.ndarray, list[str]]:
    rows = read_jsonl(labels_path)
    ids = [str(item) for item in json.loads(
        (ROOT / "artifacts/siglip/item_ids.json").read_text(encoding="utf-8"))]
    vectors = np.load(ROOT / "artifacts/siglip/image_embeddings.npy", allow_pickle=False)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    by_id = {row["item_id"]: row for row in rows}
    aligned = [by_id[item_id] for item_id in ids]
    return aligned, vectors, np.asarray(ids), ids


def train_attribute_models(rows: list[dict], vectors: np.ndarray) -> tuple[dict, dict]:
    models: dict[str, dict] = {}
    reports = {}
    domains = ("form", "fine_material", "broad_material", "era", "motif", "technique", "color")
    for domain in domains:
        labels = np.array([single_label(row, domain) for row in rows], dtype=object)
        train_mask = np.array([row["split"] == "train" and label is not None
                               for row, label in zip(rows, labels)])
        dev_mask = np.array([row["split"] == "dev" and label is not None
                             for row, label in zip(rows, labels)])
        train_counts = {value: int(((labels == value) & train_mask).sum())
                        for value in set(labels[train_mask])}
        classes = sorted(value for value, count in train_counts.items() if count >= 20)
        train_mask &= np.isin(labels, classes)
        dev_mask &= np.isin(labels, classes)
        if len(classes) < 2 or not train_mask.any() or not dev_mask.any():
            reports[domain] = {
                "status": "insufficient_three_split_support",
                "classes": classes,
                "train_counts": train_counts,
            }
            continue
        trials = []
        for C in (0.1, 1.0, 10.0):
            model = LogisticRegression(C=C, class_weight="balanced", max_iter=1500,
                                       random_state=0)
            model.fit(vectors[train_mask], labels[train_mask])
            prediction = model.predict(vectors[dev_mask])
            score = float(f1_score(labels[dev_mask], prediction, average="macro",
                                   labels=classes, zero_division=0))
            trials.append((score, -C, C, model))
        score, _, C, model = max(trials, key=lambda value: value[:2])
        models[domain] = {"model": model, "classes": list(model.classes_)}
        reports[domain] = {
            "status": "evaluable",
            "classes": classes,
            "train_counts": train_counts,
            "selected_C": C,
            "dev_macro_f1": score,
        }
    return models, reports


def classify_attributes(models: dict, vectors: np.ndarray,
                        atoms: tuple[tuple[str, str], ...]) -> np.ndarray | None:
    probabilities = []
    for domain, value in atoms:
        model = models.get(domain)
        if model is None or value not in model["classes"]:
            return None
        index = model["classes"].index(value)
        probabilities.append(model["model"].predict_proba(vectors)[:, index])
    values = np.asarray(probabilities, dtype=np.float64)
    return values


def scores_for_query(models: dict, vectors: np.ndarray,
                     atoms: tuple[tuple[str, str], ...]) -> dict[str, np.ndarray]:
    probability = classify_attributes(models, vectors, atoms)
    if probability is None:
        return {}
    return {
        "attribute_min": probability.min(axis=0),
        "attribute_mean": probability.mean(axis=0),
        "attribute_product": np.prod(probability, axis=0),
        "attribute_mean_min": (probability.mean(axis=0) + probability.min(axis=0)) / 2,
    }


def prepare_query_scores(rows: list[dict], vectors: np.ndarray, models: dict,
                         text_vectors: dict[str, np.ndarray], specs: list[QuerySpec],
                         split: str) -> list[dict]:
    """Attach split-specific dense scores without consulting candidate truth."""
    prepared = []
    for spec in specs:
        data = query_dataset(rows, spec, split)
        data["vectors"] = vectors[data["indices"]]
        dense = data["vectors"] @ text_vectors[spec.qid]
        data["dense_relative"] = rank_percentile(dense)
        data["dense"] = dense
        prepared.append({"spec": spec, "data": data})
    return prepared


def evaluate_prepared(prepared: list[dict], models: dict, method: str,
                      threshold: float, dense_percentile: float,
                      dense_weight: float) -> list[dict]:
    output = []
    for item in prepared:
        spec, data = item["spec"], item["data"]
        attributes = scores_for_query(models, data["vectors"], spec.atoms)
        if method == "dense":
            yes = data["dense_relative"] >= threshold
        else:
            attribute = attributes.get(method)
            if method == "dense_attribute":
                attribute = attributes.get("attribute_mean_min")
            if attribute is None:
                yes = np.zeros(len(data["ids"]), dtype=bool)
            elif method == "dense_attribute":
                combined = ((1 - dense_weight) * data["dense_relative"]
                            + dense_weight * attribute)
                yes = ((data["dense_relative"] >= dense_percentile)
                       & (combined >= threshold))
            else:
                yes = attribute >= threshold
        selected = {item_id for item_id, keep in zip(data["ids"], yes) if keep}
        output.append({"qid": spec.qid, **explicit_metrics(selected, data["relevant"])})
    return output


def selection_key(report: dict) -> tuple:
    return (
        report["macro_set_score"],
        report["exact_set_accuracy"],
        report["macro_f1"],
        report["no_match_accuracy"],
        -report["empty_rate_positive_queries"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default=(
        "artifacts/cv_v4_20261108/catalogue_v3_primary_plus_candidates/"
        "labels_catalogue_v2.jsonl"
    ))
    parser.add_argument("--output", default="artifacts/cv_v4_20261108/boundary_catalogue")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    labels_path = ROOT / args.labels
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    rows, vectors, item_ids, ids = load_dataset(labels_path)
    specs = build_query_specs(rows)
    models, model_report = train_attribute_models(rows, vectors)
    encoder = Encoder("siglip", device=args.device)
    text_vectors = encoder.encode_texts([spec.text for spec in specs])
    text_vectors = {spec.qid: vector for spec, vector in zip(specs, text_vectors)}

    dev = prepare_query_scores(rows, vectors, models, text_vectors, specs, "dev")
    test = prepare_query_scores(rows, vectors, models, text_vectors, specs, "test")
    configurations = []
    for method in ("dense", "attribute_min", "attribute_mean", "attribute_product",
                   "attribute_mean_min", "dense_attribute"):
        if method.startswith("attribute_") and not models:
            continue
        thresholds = tuple(np.linspace(.05, .95, 19))
        percentile_values = tuple(np.linspace(.70, .995, 20)) if method == "dense_attribute" else (0.0,)
        weights = (.5, .75, 1.0) if method == "dense_attribute" else (0.0,)
        for threshold in thresholds:
            for percentile in percentile_values:
                for weight in weights:
                    report = aggregate_query_metrics(evaluate_prepared(
                        dev, models, method, float(threshold), float(percentile), float(weight)
                    ))
                    configurations.append({
                        "method": method,
                        "threshold": float(threshold),
                        "dense_percentile": float(percentile),
                        "attribute_weight": float(weight),
                        "dev": report,
                    })
    best = max(configurations, key=lambda item: selection_key(item["dev"]))
    test_rows = evaluate_prepared(test, models, best["method"], best["threshold"],
                                  best["dense_percentile"], best["attribute_weight"])
    test_summary = aggregate_query_metrics(test_rows)
    dev_rows = evaluate_prepared(dev, models, best["method"], best["threshold"],
                                 best["dense_percentile"], best["attribute_weight"])
    report = {
        "version": "catalogue-explicit-boundary-v4",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "labels_path": labels_path.relative_to(ROOT).as_posix(),
        "labels_sha256": digest(labels_path),
        "index_sha256": digest(ROOT / "artifacts/siglip/image_embeddings.npy"),
        "query_source": "development catalogue labels only; selected query atoms frozen before test",
        "truth_policy": "known mismatch negative; missing or ambiguous catalogue property excluded",
        "candidate_policy": "image embeddings and visual attribute probes only; no candidate catalogue text enters ranking",
        "scope_note": "fixed supplementary evaluation because this catalogue has appeared in earlier studies; not a new blind holdout",
        "queries": [spec.as_dict() for spec in specs],
        "attribute_models": model_report,
        "selection": best,
        "dev": {"summary": aggregate_query_metrics(dev_rows), "rows": dev_rows},
        "test": {"summary": test_summary, "rows": test_rows},
        "test_diagnostics": {
            "unknown_excluded": {
                split: sum(query_dataset(rows, spec, split)["unknown_excluded"]
                           for spec in specs)
                for split in ("dev", "test")
            },
            "note": "unknown counts sum across queries and are not unique image counts",
        },
    }
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
    (output / "query_specs.json").write_text(json.dumps([spec.as_dict() for spec in specs],
                                                        ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"queries": len(specs), "selection": best, "test": test_summary},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
