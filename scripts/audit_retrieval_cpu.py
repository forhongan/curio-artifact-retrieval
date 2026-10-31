"""CPU scaling diagnostics and temporary silver-query feasibility audit.

This is deliberately not a retrieval-quality experiment: the 5k gallery repeats
real pilot embeddings under synthetic IDs, query vectors are cached image vectors,
and optional probe weights are random. No model is loaded or downloaded.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import platform
import tempfile
import time

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from curio.evaluate.benchmark import ERAS, ZH_MATERIALS, ZH_MOTIFS, ZH_TYPES, build, candidate_pool, grade
from curio.retrieve.engine import SearchService, load_items


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def predicate_audit(data):
    """Run the unchanged benchmark builder in a temporary output directory."""
    data_bytes = Path(data).read_bytes()
    items = [json.loads(line) for line in data_bytes.decode("utf-8").splitlines() if line.strip()]
    predicates, joint = [], []
    for (typ, mat), count in sorted(Counter((x.get("object_type"), x.get("material")) for x in items).items()):
        if count < 3 or not typ or not mat or typ in {"other", "unknown", "object"} or mat in {"other", "unknown"}:
            continue
        base = {"object_type": typ, "material": mat}
        predicates.append(base)
        subset = [x for x in items if x.get("object_type") == typ and x.get("material") == mat]
        motifs = Counter(itertools.chain.from_iterable(set(x.get("motifs", [])) for x in subset))
        supported_motifs = [motif for motif, n in sorted(motifs.items()) if n >= 3]
        predicates.extend({**base, "motif": motif} for motif in supported_motifs)
        predicates.extend({**base, "era": era} for era in ERAS
                          if sum(grade(x, {**base, "era": era}) == 3 for x in subset) >= 3)
        for motif in supported_motifs:
            for era in ERAS:
                candidate = {**base, "motif": motif, "era": era}
                matches = sum(grade(x, candidate) == 3 for x in subset)
                if matches >= 3:
                    joint.append({"predicate": candidate, "exact_grade3_items": matches})
    with tempfile.TemporaryDirectory(prefix="curio_predicate_audit_") as temporary:
        snapshot = Path(temporary) / "items.jsonl"
        snapshot.write_bytes(data_bytes)
        try:
            manifest = build(snapshot, Path(temporary) / "eval", n_dev=40, n_test=80)
            build_result = {"possible": True, "manifest": manifest}
        except ValueError as exc:
            build_result = {"possible": False, "error": str(exc)}
    records = []
    missing_zh = Counter()
    signatures = Counter()
    label_groups = {}
    grade3_sets = {}
    for predicate in predicates:
        grades = [grade(item, predicate) for item in items]
        distribution = Counter(grades)
        relevant = sum(value >= 2 for value in grades)
        grade_fingerprint = hashlib.sha256(json.dumps(sorted(zip((x["item_id"] for x in items), grades)), separators=(",", ":")).encode()).hexdigest()
        exact_ids = sorted(item["item_id"] for item, value in zip(items, grades) if value == 3)
        exact_fingerprint = hashlib.sha256(json.dumps(exact_ids, separators=(",", ":")).encode()).hexdigest()
        label_groups.setdefault(grade_fingerprint, []).append(predicate)
        grade3_sets.setdefault(exact_fingerprint, []).append(predicate)
        records.append({"predicate": predicate, "exact_grade3_items": distribution[3],
                        "grade_distribution": {str(value): distribution[value] for value in range(4)},
                        "relevant_grade2_or_3_items": relevant,
                        "exact_grade3_fraction": distribution[3] / len(items),
                        "relevant_grade2_or_3_fraction": relevant / len(items),
                        "graded_qrels_sha256": grade_fingerprint,
                        "grade3_item_set_sha256": exact_fingerprint})
        signatures["+".join(sorted(predicate))] += 1
        if predicate["object_type"] not in ZH_TYPES:
            missing_zh["object_type:" + predicate["object_type"]] += 1
        if predicate["material"] not in ZH_MATERIALS:
            missing_zh["material:" + predicate["material"]] += 1
        if predicate.get("motif") and predicate["motif"] not in ZH_MOTIFS:
            missing_zh["motif:" + predicate["motif"]] += 1
    pool = candidate_pool(items)
    usable_hashes = {row["graded_qrels_sha256"] for row in pool["candidates"]}
    additional_hashes = set()
    for row in joint:
        labels = {item["item_id"]: grade(item, row["predicate"]) for item in items}
        fingerprint = hashlib.sha256(json.dumps(sorted(labels.items()), separators=(",", ":")).encode()).hexdigest()
        row["graded_qrels_sha256"] = fingerprint
        row["adds_distinct_informative_labels"] = any(value < 2 for value in labels.values()) and fingerprint not in usable_hashes
        if row["adds_distinct_informative_labels"]:
            additional_hashes.add(fingerprint)
    return {"data_sha256": hashlib.sha256(data_bytes).hexdigest(), "real_items": len(items),
            "requested_semantic_predicates": 120, "existing_supported_predicates": len(predicates),
            "supported_informative_unique_queries": pool["distinct_graded_qrels"],
            "excluded_all_relevant_predicates": pool["excluded_all_relevant_predicates"],
            "builder": build_result, "output_policy": "temporary outputs removed; formal eval untouched",
            "minimum_exact_grade3_support": 3, "predicate_signatures": dict(signatures),
            "predicates": records, "supported_motif_plus_era_extensions": joint,
            "distinct_graded_qrels": len(label_groups),
            "identical_graded_qrels_groups": [group for group in label_groups.values() if len(group) > 1],
            "distinct_grade3_item_sets": len(grade3_sets),
            "identical_grade3_item_set_groups": [group for group in grade3_sets.values() if len(group) > 1],
            "queries_with_all_items_relevant": [row["predicate"] for row in records if row["relevant_grade2_or_3_items"] == len(items)],
            "queries_with_grade3_fraction_at_least_25_percent": [row["predicate"] for row in records if row["exact_grade3_fraction"] >= .25],
            "joint_distinct_additional_label_vectors": len(additional_hashes),
            "extended_total_if_adopted": pool["distinct_graded_qrels"] + len(additional_hashes),
            "untranslated_ontology_tokens": dict(missing_zh),
            "recommendation": "Keep only distinct supported intents; do not count translations/paraphrases as new predicates. "
                              "Motif+era combinations below have at least three actual metadata matches but require "
                              "a documented protocol change and bilingual ontology review. If still below 120, "
                              "use a smaller transparent benchmark or collect more museum objects/independent queries. "
                              "These remain metadata-silver labels, not visual relevance judgments."}


def quantiles(values):
    return {"p50_ms": float(np.quantile(values, .5)), "p95_ms": float(np.quantile(values, .95)),
            "min_ms": float(np.min(values)), "max_ms": float(np.max(values))}


def cpu_scaling(data, index, *, size=5000, repetitions=30):
    index = Path(index)
    vectors = np.asarray(np.load(index / "image_embeddings.npy", allow_pickle=False), dtype=np.float32)
    ids = json.loads((index / "item_ids.json").read_text(encoding="utf-8"))
    catalog = {item["item_id"]: item for item in load_items(data)}
    if vectors.ndim != 2 or vectors.shape[0] != len(ids) or not ids or not np.isfinite(vectors).all():
        raise ValueError("Invalid source vector gallery")
    usable_rows = [row for row, key in enumerate(ids) if key in catalog]
    excluded_ids = sorted(set(ids) - catalog.keys())
    if not usable_rows:
        raise ValueError("Pilot index has no IDs present in the current catalog")
    source_rows = np.asarray(usable_rows, dtype=np.int64)[np.arange(size) % len(usable_rows)]
    gallery_vectors = vectors[source_rows].copy()
    synthetic = [{**catalog[ids[row]], "item_id": f"synthetic_{number:05d}__{ids[row]}"}
                 for number, row in enumerate(source_rows)]
    synthetic_ids = [item["item_id"] for item in synthetic]
    fingerprint = hashlib.sha256(gallery_vectors.tobytes() + json.dumps(synthetic_ids).encode()).hexdigest()
    synthetic_data_hash = hashlib.sha256(json.dumps(synthetic, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    dimension = vectors.shape[1]
    manifest = {"encoder": "clip", "index_version": "synthetic-cpu-fixture", "index_sha256": fingerprint,
                "data_sha256": synthetic_data_hash, "dimension": dimension, "items": size}

    class CachedImageVectorEncoder:
        def encode_texts(self, texts):
            row = int(hashlib.sha256(texts[0].encode()).hexdigest()[:8], 16) % len(vectors)
            return vectors[row:row + 1]

    class SyntheticProbes:
        calls = 0

        def predict(self, embeddings):
            self.calls += 1
            weights = np.random.default_rng(5523).normal(size=(dimension, 2)).astype(np.float32)
            logits = embeddings @ weights
            logits -= logits.max(axis=1, keepdims=True)
            probabilities = np.exp(logits)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            return {"material": {"classes": ["bronze", "porcelain"], "probabilities": probabilities}}

    probe_report = {"index_sha256": fingerprint, "data_sha256": synthetic_data_hash,
                    "index_version": "synthetic-cpu-fixture", "feature_dimension": dimension,
                    "tasks": {"material": {"status": "trained", "classes": ["bronze", "porcelain"]}}}
    probes = SyntheticProbes()
    query_texts = ["porcelain vase decorated with dragon", "bronze bowl from the Ming dynasty",
                   "\u9752\u82b1\u74f7\u74f6\uff0c\u5e26\u6709\u83b2\u82b1\u7eb9\u9970"]
    with threadpool_limits(limits=1, user_api="blas"):
        started = time.perf_counter()
        service = SearchService(synthetic, embeddings=gallery_vectors, item_ids=synthetic_ids,
                                encoder=CachedImageVectorEncoder(), probes=probes, probe_report=probe_report,
                                index_manifest=manifest)
        startup_ms = (time.perf_counter() - started) * 1000
        runtime = threadpool_info()
        results = []
        for variant in ("baseline", "hybrid", "visual"):
            for text in query_texts:
                service.search(text, variant=variant, top_k=20)
            rows = []
            for number in range(repetitions):
                response = service.search(query_texts[number % len(query_texts)], variant=variant, top_k=20)
                if response["degraded"] or len(response["results"]) != min(20, size):
                    raise ValueError("Synthetic scaling fixture unexpectedly degraded")
                rows.append(response["timings"])
            results.append({"variant": variant, "requests": repetitions,
                            "stages": {key: quantiles([row[key] for row in rows])
                                       for key in ("parse_ms", "encode_ms", "retrieve_ms", "fusion_ms", "total_ms")}})
    return {"synthetic": True, "purpose": "CPU retrieval/fusion scaling only; not relevance or real end-to-end latency",
            "gallery_count": size, "dimension": dimension, "source_real_index_rows": len(ids),
            "unique_source_rows": len(usable_rows), "excluded_stale_index_ids": excluded_ids,
            "source_index": str(index),
            "source_embeddings_sha256": digest(index / "image_embeddings.npy"),
            "source_item_ids_sha256": digest(index / "item_ids.json"),
            "source_manifest_sha256": digest(index / "manifest.json"),
            "current_catalog_sha256": digest(data), "synthetic_index_sha256": fingerprint,
            "gallery_construction": "Repeat genuine pilot image vectors and catalog rows under unique synthetic IDs",
            "query_vectors": "Cached image vectors selected by a stable query hash; no text model, GPU, or downloads",
            "probe_weights": "Seed-5523 random two-class weights; startup prediction cached once; no quality claim",
            "probe_predict_calls": probes.calls, "startup_ms": startup_ms, "probe_load_ms": service.probe_load_ms,
            "warmup_requests_per_variant": len(query_texts), "top_k": 20, "blas_threads": 1,
            "threadpools": runtime, "queries": query_texts, "results": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/snapshots/collection_v2.jsonl")
    parser.add_argument("--index", default="artifacts/pilot_siglip")
    parser.add_argument("--output", default="artifacts/diagnostics/backend_review_20261030.json")
    parser.add_argument("--size", type=int, default=5000)
    parser.add_argument("--repetitions", type=int, default=30)
    args = parser.parse_args()
    if args.size < 1 or args.repetitions < 1:
        parser.error("size and repetitions must be positive")
    data_bytes = Path(args.data).read_bytes()
    with tempfile.TemporaryDirectory(prefix="curio_diagnostic_snapshot_") as temporary:
        snapshot = Path(temporary) / "items.jsonl"
        snapshot.write_bytes(data_bytes)
        report = {"created_at": datetime.now(timezone.utc).isoformat(), "source_catalog": args.data,
                  "environment": {"python": platform.python_version(), "platform": platform.platform(), "processor": platform.processor()},
                  "predicate_audit": predicate_audit(snapshot),
                  "synthetic_cpu_scaling": cpu_scaling(snapshot, args.index, size=args.size, repetitions=args.repetitions)}
    if Path(args.data).read_bytes() != data_bytes:
        raise ValueError("Catalog changed during diagnosis; wait for ingestion to finish and rerun")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "supported_predicates": report["predicate_audit"]["existing_supported_predicates"],
                      "informative_unique_queries": report["predicate_audit"]["supported_informative_unique_queries"],
                      "supported_with_joint_extensions": report["predicate_audit"]["extended_total_if_adopted"],
                      "synthetic_cpu_scaling": report["synthetic_cpu_scaling"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
