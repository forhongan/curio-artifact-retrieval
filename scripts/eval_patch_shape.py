"""Predeclared weak object-form audit of frozen DINO top100 patch matching.

No blind eight-intent judgments, training, threshold selection, or visual labels
are used. Catalogue class matches are auxiliary weak labels, not image truth.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

from curio.represent.patch_match import DinoPatchMatcher, PARAMETERS, VERSION, rerank_patch_candidates
from curio.retrieve.engine import SearchService, load_items


FORMS = ("bowl", "plate", "cup", "ewer", "censer", "figure")
SEED = "patch-form-audit-20261030-v1"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def freeze_protocol(items, count, catalogue_sha):
    labels = {str(item["item_id"]): sorted(SearchService._attributes(item)["object_type"]) for item in items}
    queries = []
    for form in FORMS:
        eligible = [item for item in items if item.get("split") == "test"
                    and set(labels[str(item["item_id"])]).intersection(FORMS) == {form}]
        eligible.sort(key=lambda item: hashlib.sha256(f"{SEED}:{item['item_id']}".encode()).hexdigest())
        used = set()
        for item in eligible:
            family = item.get("object_group") or str(item["item_id"])
            if family in used:
                continue
            used.add(family)
            queries.append({"qid": f"{form}_{len(used):02d}", "form": form,
                            "reference_id": str(item["item_id"]), "reference_family": family,
                            "reference_path": item["image_path"], "split": "test"})
            if len(used) == count:
                break
        if len(used) != count:
            raise ValueError(f"Not enough distinct test families for {form}")
    excluded_families = {row["reference_family"] for row in queries}
    excluded = sorted(str(item["item_id"]) for item in items
                      if (item.get("object_group") or str(item["item_id"])) in excluded_families)
    return {"protocol": SEED, "catalogue_sha256": catalogue_sha, "forms": list(FORMS),
            "query_count_per_form": count, "queries": queries, "excluded_ids": excluded,
            "exclusion": "all query reference families excluded from every gallery ranking",
            "weak_labels": "object_type aliases from title/object_name/classification/object_type; metadata only",
            "candidate_k": 100, "report_k": 10, "parameters": PARAMETERS,
            "methods": ["dinov2_cls", "local_score", "shape_similarity", "combined_score"],
            "model_selection": "none; fixed algorithm/thresholds/weights before ranking",
            "limitations": "Class aliases are incomplete catalogue labels, not independently judged visible form. No material/date/authenticity claim. Test split is diagnostic only; no tuning on results."}, labels


def weak_metrics(order, target_ids, known_ids, k=10):
    top = list(order[:k])
    gains = np.array([int(key in target_ids) for key in top], dtype=float)
    discounts = 1 / np.log2(np.arange(2, 2 + len(top)))
    ideal = discounts[:min(len(target_ids), len(top))].sum()
    return {"metadata_match_at10": float(gains.sum() / len(top)) if top else 0,
            "metadata_ndcg_at10": float((gains * discounts).sum() / ideal) if ideal else None,
            "unknown_metadata_fraction_at10": sum(key not in known_ids for key in top) / len(top) if top else None}


def save_feature(path, feature):
    np.savez_compressed(path, tokens=feature["tokens"], weights=feature["weights"], profile=feature["profile"],
                        log_aspect=feature["log_aspect"], foreground_weights=feature["foreground_weights"],
                        diagnostics=json.dumps(feature["diagnostics"]))


def load_feature(path):
    with np.load(path, allow_pickle=False) as data:
        return {"tokens": data["tokens"], "weights": data["weights"], "profile": data["profile"],
                "log_aspect": float(data["log_aspect"]), "foreground_weights": data["foreground_weights"],
                "diagnostics": json.loads(str(data["diagnostics"]))}


def run(args):
    started = time.perf_counter()
    root = Path.cwd()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    cache = output / "feature_cache"
    cache.mkdir(exist_ok=True)
    items = load_items(args.data)
    catalogue_sha = digest(args.data)
    protocol, labels = freeze_protocol(items, args.queries_per_form, catalogue_sha)
    encoded_protocol = json.dumps(protocol, sort_keys=True, ensure_ascii=False).encode()
    protocol_sha = hashlib.sha256(encoded_protocol).hexdigest()
    protocol_file = output / "protocol.json"
    if protocol_file.exists() and json.loads(protocol_file.read_text(encoding="utf-8")) != protocol:
        raise ValueError("Existing output has a different frozen protocol; use a fresh output directory")
    write_json(protocol_file, protocol)  # Written BEFORE model loading or rankings.
    manifest = json.loads((args.index / "manifest.json").read_text(encoding="utf-8"))
    ids = json.loads((args.index / "item_ids.json").read_text(encoding="utf-8"))
    vectors = np.load(args.index / "image_embeddings.npy", allow_pickle=False)
    index_sha = hashlib.sha256(vectors.tobytes() + json.dumps(ids).encode()).hexdigest()
    if manifest.get("data_sha256") != catalogue_sha or manifest.get("index_sha256") != index_sha:
        raise ValueError("DINO index/catalogue provenance mismatch")
    if len(set(ids)) != len(ids) or set(ids) != set(labels):
        raise ValueError("DINO IDs must uniquely cover the full catalogue")
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    lookup = {key: i for i, key in enumerate(ids)}
    by_id = {str(item["item_id"]): item for item in items}
    excluded = set(protocol["excluded_ids"])
    eligible = [i for i, key in enumerate(ids) if key not in excluded]
    queries = []
    feature_ids = set()
    with threadpool_limits(limits=1):
        for query in protocol["queries"]:
            similarity = vectors @ vectors[lookup[query["reference_id"]]]
            order = sorted(eligible, key=lambda i: (-float(similarity[i]), ids[i]))
            candidate_ids = [ids[i] for i in order[:100]]
            queries.append({**query, "candidates": candidate_ids,
                            "base_scores": [float(similarity[i]) for i in order[:100]],
                            "full_order": [ids[i] for i in order]})
            feature_ids.update(candidate_ids)
            feature_ids.add(query["reference_id"])
    matcher = DinoPatchMatcher()
    if manifest["encoder_metadata"]["revision"] != matcher.metadata["revision"]:
        raise ValueError("DINO feature revision differs from global CLS index")
    features = {}
    pending = []
    cache_paths = {}
    model_key = json.dumps(matcher.metadata, sort_keys=True)
    for key in sorted(feature_ids):
        image_path = root / by_id[key]["image_path"]
        fingerprint = hashlib.sha256((digest(image_path) + model_key).encode()).hexdigest()
        cache_path = cache / f"{fingerprint}.npz"
        cache_paths[key] = cache_path
        if cache_path.exists():
            features[key] = load_feature(cache_path)
        else:
            pending.append(key)
    extraction_started = time.perf_counter()
    for start in range(0, len(pending), args.batch_size):
        batch_ids = pending[start:start + args.batch_size]
        extracted = matcher.encode_images([root / by_id[key]["image_path"] for key in batch_ids], batch_size=args.batch_size)
        for key, feature in zip(batch_ids, extracted):
            features[key] = feature
            save_feature(cache_paths[key], feature)
        print(f"patch extraction {min(start + len(batch_ids), len(pending))}/{len(pending)}", flush=True)
    extraction_seconds = time.perf_counter() - extraction_started
    classical_dir = root / "artifacts/visual_parts"
    if (classical_dir / "manifest.json").exists():
        from curio.represent.visual_parts import load_visual_parts

        classical = load_visual_parts(classical_dir)
        if classical["manifest"]["data_sha256"] != catalogue_sha:
            raise ValueError("Classical coverage comparison catalogue mismatch")
        classical_ids = classical["item_ids"]
        valid_shape = classical["features"]["valid_shape"].astype(bool)
        accepted = {key for key, valid in zip(classical_ids, valid_shape) if valid}
        reference_ids = {row["reference_id"] for row in protocol["queries"]}
        patch_accepted = {key for key in feature_ids if features[key]["diagnostics"]["status"] == "valid"}
        write_json(output / "coverage_comparison.json", {
            "purpose": "descriptive coverage on identical frozen audit images; no threshold selection",
            "image_count": len(feature_ids), "reference_count": len(reference_ids),
            "classical_accepted_images": len(feature_ids.intersection(accepted)),
            "patch_accepted_images": len(patch_accepted),
            "classical_accepted_references": len(reference_ids.intersection(accepted)),
            "patch_accepted_references": len(reference_ids.intersection(patch_accepted)),
            "both_accepted_images": len(patch_accepted.intersection(accepted)),
            "patch_only_accepted_images": len(patch_accepted - accepted),
            "classical_only_accepted_images": len(feature_ids.intersection(accepted) - patch_accepted),
            "classical_manifest_sha256": digest(classical_dir / "manifest.json"),
            "protocol_sha256": protocol_sha,
        })
    known_ids = {key for key, value in labels.items() if value} - excluded
    records = []
    with threadpool_limits(limits=1):
        for query in queries:
            target_ids = {key for key, value in labels.items() if query["form"] in value} - excluded
            reference = features[query["reference_id"]]
            candidates = [features[key] for key in query["candidates"]]
            base = query["full_order"]
            common = {"qid": query["qid"], "form": query["form"],
                      "reference_id": query["reference_id"], "reference_diagnostics": reference["diagnostics"],
                      "weak_positive_count": len(target_ids),
                      "candidate_recall100_metadata": len(target_ids.intersection(query["candidates"])) / len(target_ids)}
            records.append({**common, "method": "dinov2_cls", "ranking": base[:10],
                            **weak_metrics(base, target_ids, known_ids)})
            for mode in ("local_score", "shape_similarity", "combined_score"):
                match_started = time.perf_counter()
                result = rerank_patch_candidates(reference, candidates, query["base_scores"], query["candidates"], mode=mode)
                order = [query["candidates"][i] for i in result["order"]]
                records.append({**common, "method": mode, "ranking": order[:10],
                                "status": result["status"], "accepted_candidate_count": result["accepted_candidate_count"],
                                "coverage": result["coverage"], "matching_ms": (time.perf_counter() - match_started) * 1000,
                                **weak_metrics(order, target_ids, known_ids)})
            print(query["qid"], reference["diagnostics"]["status"], flush=True)
    summaries = {}
    for method in protocol["methods"]:
        rows = [row for row in records if row["method"] == method]
        summaries[method] = {"queries": len(rows),
            **{metric: float(np.mean([row[metric] for row in rows])) for metric in
               ("metadata_match_at10", "metadata_ndcg_at10", "unknown_metadata_fraction_at10", "candidate_recall100_metadata")},
            "reference_accepted": sum(row["reference_diagnostics"]["status"] == "valid" for row in rows)}
        if method != "dinov2_cls":
            summaries[method]["mean_candidate_pair_coverage"] = float(np.mean([row["coverage"] for row in rows]))
            summaries[method]["fallback_queries"] = sum(row["status"] == "fallback_cls" for row in rows)
    write_json(output / "results.json", {"protocol_sha256": protocol_sha, "summary": summaries, "records": records})
    write_json(output / "run_manifest.json", {"created_at": datetime.now(timezone.utc).isoformat(),
        "method": VERSION, "model": matcher.metadata, "protocol_sha256": protocol_sha,
        "gallery_count": len(ids), "eligible_gallery_count": len(eligible), "excluded_family_item_count": len(excluded),
        "catalogue_sha256": catalogue_sha, "index_sha256": index_sha,
        "implementation_sha256": digest(root / "src/curio/represent/patch_match.py"),
        "evaluator_sha256": digest(__file__), "features_examined": len(features), "new_features_extracted": len(pending),
        "feature_status": dict(Counter(row["diagnostics"]["status"] for row in features.values())),
        "rejection_reasons": dict(Counter(reason for row in features.values() for reason in row["diagnostics"]["reasons"])),
        "extraction_seconds": extraction_seconds, "total_seconds": time.perf_counter() - started,
        "human_judgments": False, "label_source": "weak_catalogue_alias_labels_not_visual_gold",
        "blind_pool_used": False, "tuning_performed": False})
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/snapshots/collection_v2.jsonl"))
    parser.add_argument("--index", type=Path, default=Path("artifacts/dinov2"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/patch_shape_eval"))
    parser.add_argument("--queries-per-form", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    run(parser.parse_args())
