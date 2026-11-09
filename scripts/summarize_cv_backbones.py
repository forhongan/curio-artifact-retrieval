"""Verify supplementary backbone artifacts and produce a compact research summary."""

import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from compare_cv_backbones import BASE, OLD, OUT, ROOT, read, vectors
from curio.retrieve.attribute_cv import predict_heads
from run_cv_research import dump, rows, sha


def main():
    protocol = read(OUT / "protocol.json")
    assert protocol["parent_protocol_sha256"] == sha(BASE / "protocol.json")
    assert protocol["parent_selection_sha256"] == sha(OLD / "selection.json")
    assert protocol["intent_sha256"] == sha(BASE / "queries.json")
    catalog = rows(ROOT / "data/snapshots/collection_v2.jsonl")
    catalog_by_id = {r["item_id"]: r for r in catalog}
    test = read(OLD / "test_annotations.json")
    assert all(catalog_by_id[r["item_id"]]["split"] == "test" for r in test)
    old_train = [
        r for r in rows(ROOT / "eval/visual_labels/annotations.jsonl") if r["split"] == "train"
    ]
    assert all(catalog_by_id[r["item_id"]]["split"] == "train" for r in old_train)
    ids = read(ROOT / "artifacts/classical_cv/item_ids.json")
    counts = {}
    with threadpool_limits(limits=1):
        for name in ("clip", "lora", "siglip1"):
            h = np.load(OUT / f"{name}_heads.npz")
            assert h["classes"].tolist() == read(OLD / "selection.json")["classes"]
            fresh = predict_heads(vectors(name, ids), h["coefficients"], h["intercepts"])
            saved = np.load(OUT / f"{name}_predictions.npy")
            assert fresh.shape == (4692, 21)
            assert np.allclose(saved, fresh, atol=2e-6)
            counts[name] = int(fresh.size)
    pixels = rows(BASE / "pixels.jsonl")
    enc = read(OUT / "encoding.json")
    for name in ("clip", "siglip2", "lora", "dino", "siglip1"):
        v = np.load(OUT / f"{name}_timed_embeddings.npy")
        assert len(v) == len(pixels) == 4692
        assert np.isfinite(v).all() and np.allclose(np.linalg.norm(v, axis=1), 1, atol=2e-5)
        assert enc["models"][name]["full_gallery"]["count"] == len(v)
        if name != "siglip1":
            expected = vectors(name, [r["item_id"] for r in pixels])
            assert np.min(np.sum(expected * v, axis=1)) > 0.9999
        else:
            assert np.array_equal(v, vectors(name, [r["item_id"] for r in pixels]))
    accuracy = read(OUT / "accuracy.json")
    primary = {
        name: accuracy[f"{name}_raw_en"]["test"]["mean_all_intents"]
        for name in ("clip", "siglip1", "siglip2", "lora")
    }
    attribute = {
        name: accuracy[f"{name}_attributes"]["test"]["mean_all_intents"]
        for name in ("clip", "siglip1", "lora")
    }
    parent = read(OLD / "test_results.json")
    for name in ("siglip", "dino", "raw_all", "gc_all"):
        attribute["siglip2" if name == "siglip" else name] = parent[name]["mean_all_intents"]
    assert (
        abs(primary["siglip2"]["P@5"] - parent["siglip_dot_en"]["mean_all_intents"]["P@5"]) < 1e-12
    )
    assert (
        abs(primary["lora"]["P@5"] - parent["production_dot_en"]["mean_all_intents"]["P@5"]) < 1e-12
    )
    pf = read(OUT / "prefilters.json")
    dense = accuracy["lora_rewritten_en"]["test"]["mean_all_intents"]
    for name, item in pf["quality"]["test"].items():
        if item["fraction"] == 1:
            assert item["macro_retention"] == 1 and item["total_lost_positives"] == 0
            assert abs(item["P@5"] - dense["P@5"]) < 1e-12
        for q in item["per_query"]:
            if not q["supported"]:
                assert q["candidates"] == 96 and q["lost_known_positives"] == 0
    dump(
        OUT / "summary.json",
        {
            "limits": "21 fixed intents, restricted96-image gallery; AI pixel labels; unknowns excluded from positive hits. Attribute methods trained on218 labels, finite vocabulary and20/21 query coverage; dense baselines have21/21. Test previously exposed; supplementary fixed controls, no new independent test claim.",
            "dense_raw_english": primary,
            "attribute_controls": attribute,
            "encoding": {
                n: {
                    "full_gallery": r["full_gallery"],
                    "single_image_p50_ms": r["single_image_end_to_end"]["p50_ms"],
                    "precision": r["precision"],
                }
                for n, r in enc["models"].items()
            },
            "prefilter_half": {
                n: {k: v for k, v in pf["quality"]["test"][f"{n}_0.5"].items() if k != "per_query"}
                for n in ("raw_all", "gc_all", "dino", "fusion")
            },
        },
    )
    dump(
        OUT / "acceptance.json",
        {
            "status": "passed",
            "frozen_parent_hashes": "unchanged",
            "lora_test_items_in_original_catalog_train": 0,
            "heads_predictions_verified": counts,
            "encoding_models_verified": list(enc["models"]),
            "timed_images_per_model": 4692,
            "prefilter_unsupported_passthrough": "verified",
            "unpruned_reranker_matches_full_gallery": "verified",
        },
    )
    manifest = []
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and p.name != "artifact_manifest.json":
            manifest.append(
                {"path": str(p.relative_to(ROOT)), "sha256": sha(p), "bytes": p.stat().st_size}
            )
    dump(
        OUT / "artifact_manifest.json",
        {
            "artifacts": manifest,
            "source": {
                str(p.relative_to(ROOT)): sha(p)
                for p in [Path(__file__), ROOT / "scripts/compare_cv_backbones.py"]
            },
        },
    )
    print(json.dumps({"dense": primary, "attributes": attribute, "checks": "passed"}, indent=2))


if __name__ == "__main__":
    main()
