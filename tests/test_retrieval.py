import numpy as np
import json
import hashlib
import time
import sys
from types import SimpleNamespace

import pytest

from curio.index.build import build_index

from curio.retrieve.engine import SearchService
from curio.retrieve.lexical import BM25


def items():
    return [
        {"item_id": "met_1", "title": "Dragon vase", "object_name": "Vase", "medium": "Porcelain", "tags": ["dragon"], "year_from": 1400, "year_to": 1500, "source": "met", "image_path": "data/processed/images/met_1.jpg"},
        {"item_id": "cma_2", "title": "Lotus bowl", "object_name": "Bowl", "medium": "Bronze", "tags": ["lotus"], "year_from": 1600, "year_to": 1700, "source": "cma", "image_path": "data/processed/images/cma_2.jpg"},
    ]


class FakeEncoder:
    def encode_texts(self, texts):
        return np.array([[1.0, 0.0]], dtype=np.float32)


def test_bm25_uses_document_frequency_and_ranks_term_match():
    scores = BM25(["rare dragon vase", "common blue bowl", "common blue bowl"]).score("dragon")
    assert scores[0] > 0 and scores[1] == 0


def test_hybrid_uses_vectors_and_metadata_evidence_not_visual_claims():
    service = SearchService(items(), embeddings=np.array([[1, 0], [0, 1]], dtype=np.float32), encoder=FakeEncoder())
    result = service.search("Ming porcelain vase with dragon", variant="hybrid")
    assert result["retrieval_mode"] == "image_embeddings"
    assert result["results"][0]["item_id"] == "met_1"
    assert all(evidence["kind"] == "metadata" for evidence in result["results"][0]["evidence"] if "slot" in evidence)
    assert "未进行图像可见性核验" in result["results"][0]["evidence"][0]["text"]


def test_rrf_equal_channel_scores_share_average_rank():
    catalog = [{"item_id": item_id, "title": "Porcelain vase", "object_type": "vase", "material": "porcelain"}
               for item_id in ("met_1", "met_2")]
    service = SearchService(catalog, embeddings=np.array([[1, 0], [1, 0]], dtype=np.float32), encoder=FakeEncoder())
    rows = service.search("porcelain vase", variant="hybrid")["results"]
    assert rows[0]["score"] == rows[1]["score"]
    for row in rows:
        assert row["score_breakdown"]["rrf_ranks"] == {"bm25": 1.5, "slots": 1.5, "dense": 1.5}


def test_rrf_score_gap_is_invariant_to_swapped_item_ids():
    vectors = np.array([[1, 0], [.8, .6]], dtype=np.float32)
    gaps = []
    for ids in (("met_1", "met_2"), ("met_2", "met_1")):
        catalog = [{"item_id": item_id, "title": "Porcelain vase", "object_type": "vase", "material": "porcelain"}
                   for item_id in ids]
        service = SearchService(catalog, embeddings=vectors, encoder=FakeEncoder())
        rows = {row["item_id"]: row for row in service.search("porcelain vase", variant="hybrid")["results"]}
        gaps.append(rows[ids[0]]["score"] - rows[ids[1]]["score"])
    assert gaps[0] > 0
    assert gaps[0] == pytest.approx(gaps[1])


def test_negative_attribute_never_earns_positive_slot_score():
    service = SearchService(items())
    result = service.search("porcelain vase not dragon", variant="hybrid")
    assert "dragon" not in result["parsed"]["slots"]["motifs"]
    assert result["parsed"]["negations"]
    assert all("dragon" not in [hit["value"] for hit in row["matched_slots"]] for row in result["results"])


@pytest.mark.parametrize("title", ["Covered Jar", "Lidded Jar"])
def test_negated_lid_penalizes_explicit_covered_metadata_without_certifying_absence(title):
    catalog = [{"item_id": "met_1", "title": title, "object_type": "jar", "material": "porcelain"},
               {"item_id": "met_2", "title": "Storage Jar", "object_type": "jar", "material": "porcelain"}]
    service = SearchService(catalog, embeddings=np.array([[1, 0], [1, 0]], dtype=np.float32), encoder=FakeEncoder())
    neutral = service.search("罐子", variant="hybrid")
    assert neutral["results"][0]["item_id"] == "met_1"
    assert neutral["results"][0]["score"] == neutral["results"][1]["score"]
    response = service.search("没有盖的罐子", variant="hybrid")
    assert response["results"][0]["item_id"] == "met_2"
    by_id = {row["item_id"]: row for row in response["results"]}
    assert by_id["met_1"]["score_breakdown"]["negative_conflicts"] == 1
    assert by_id["met_2"]["score_breakdown"]["negative_conflicts"] == 0
    assert all(hit["value"] != "lid" for row in response["results"] for hit in row["matched_slots"])
    assert any("缺失标签不能证明" in warning for warning in response["warnings"])
    assert all("核验" in entry["text"] for entry in response["results"][0]["evidence"])


def test_user_filters_are_strict_and_never_auto_relaxed():
    service = SearchService(items(), embeddings=np.array([[1, 0], [0, 1]], dtype=np.float32), encoder=FakeEncoder())
    result = service.search("vase", filters={"object_type": ["bowl"]})
    assert [row["item_id"] for row in result["results"]] == ["cma_2"]
    result = service.search("vase", filters={"source": ["not-a-museum"]})
    assert result["results"] == []
    assert any("未自动放宽" in warning for warning in result["warnings"])


@pytest.mark.parametrize("variant", ["baseline", "hybrid", "visual", "lexical", "image_only"])
def test_allowed_gallery_restricts_ranking_without_changing_index(variant):
    service = SearchService(items(), embeddings=np.eye(2, dtype=np.float32), encoder=FakeEncoder())
    fingerprint = service.index_sha256
    response = service.search("bronze bowl", variant=variant, allowed_item_ids={"cma_2"})
    assert [row["item_id"] for row in response["results"]] == ["cma_2"]
    assert service.index_sha256 == fingerprint and service.health()["indexed_images"] == 2
    assert service.search("vase", variant="baseline", allowed_item_ids=set())["results"] == []
    with pytest.raises(ValueError, match="Unknown allowed_item_ids"):
        service.search("vase", allowed_item_ids={"missing"})


def test_missing_vectors_is_explicit_metadata_diagnostic():
    service = SearchService(items())
    result = service.search("vase")
    assert result["degraded"] is True
    assert result["retrieval_mode"] == "metadata_only"
    assert service.health()["retrieval_mode"] == "metadata_only"


def test_confidence_summary_separates_visual_coverage_from_rank_margin():
    catalog = [
        {"item_id": "met_1", "title": "Porcelain vase", "object_type": "vase", "material": "porcelain"},
        {"item_id": "met_2", "title": "Bronze bowl", "object_type": "bowl", "material": "bronze"},
    ]
    service = SearchService(catalog, embeddings=np.array([[1, 0], [0, 1]], dtype=np.float32), encoder=FakeEncoder())
    response = service.search("porcelain vase", variant="image_only")
    confidence = response["confidence"]
    assert confidence["basis"] and confidence["level"] == "limited"
    assert confidence["visual_coverage"]["status"] == "unknown"
    assert confidence["signal_availability"] == {"dense": True, "probes": False, "probes_applied": False}
    assert confidence["signal_agreement"]["status"] == "not_measured"
    assert confidence["calibration"]["correctness_probability"] is None
    assert confidence["ranking_separation"]["top1_top2_margin"] is not None


def test_confidence_summary_is_unknown_without_visual_index():
    response = SearchService(items()).search("vase", variant="image_only")
    confidence = response["confidence"]
    assert confidence["level"] == "unknown"
    assert confidence["visual_coverage"]["status"] == "unknown"
    assert confidence["signal_agreement"]["status"] == "not_measured"
    assert confidence["signal_availability"]["dense"] is False


def test_baseline_passes_raw_query_and_does_not_penalize_negation():
    class Recorder(FakeEncoder):
        def encode_texts(self, texts):
            self.texts = texts
            return super().encode_texts(texts)
    encoder = Recorder()
    service = SearchService(items(), embeddings=np.array([[1, 0], [0, 1]], dtype=np.float32), encoder=encoder)
    result = service.search("vase without dragon", variant="baseline")
    assert encoder.texts == ["vase without dragon"]
    assert result["results"][0]["score"] == 1.0


def test_partial_index_does_not_mix_unindexed_metadata_into_visual_results():
    service = SearchService(items(), embeddings=np.array([[1, 0]], dtype=np.float32), item_ids=["met_1"], encoder=FakeEncoder())
    result = service.search("lotus bronze bowl", variant="hybrid")
    assert [row["item_id"] for row in result["results"]] == ["met_1"]
    assert result["degraded"]
    assert service.health()["unindexed_items"] == 1


def test_wrong_encoder_index_is_rejected_even_with_same_dimension(tmp_path):
    np.save(tmp_path / "image_embeddings.npy", np.array([[1, 0], [0, 1]], dtype=np.float32))
    (tmp_path / "item_ids.json").write_text(json.dumps(["met_1", "cma_2"]))
    (tmp_path / "manifest.json").write_text(json.dumps({"encoder": "siglip", "dimension": 2}))
    service = SearchService(items(), index_dir=tmp_path, encoder_name="clip", encoder=FakeEncoder())
    assert service.embeddings is None
    assert "does not match" in service.health()["warnings"][0]


def test_description_does_not_create_attribute_evidence():
    sample = items()[0]
    sample.update(tags=[], description="This object is unlike a blue dragon bowl.")
    service = SearchService([sample])
    assert service.attributes[0]["motifs"] == set()
    assert service.attributes[0]["colors"] == set()


def test_build_index_records_encoder_revision_and_adapter_metadata(tmp_path):
    sample = items()[0]
    image = tmp_path / sample["image_path"]
    image.parent.mkdir(parents=True)
    image.write_bytes(b"fixture")
    (tmp_path / "items.jsonl").write_text(json.dumps(sample), encoding="utf-8")
    class FakeImageEncoder:
        metadata = {"encoder": "clip", "revision": "revision123", "adapter_path": "adapter-dir", "adapter_hash": "abc"}
        def encode_images(self, paths, batch_size=16):
            return np.array([[3.0, 4.0]], dtype=np.float32)
    manifest = build_index("items.jsonl", "index", encoder=FakeImageEncoder(), root=tmp_path)
    assert manifest["adapted"] and manifest["adapter_hash"] == "abc"
    assert manifest["encoder_metadata"]["revision"] == "revision123"
    assert manifest["catalog_sha256"] and manifest["data_sha256"]
    np.testing.assert_allclose(np.load(tmp_path / "index/image_embeddings.npy"), [[.6, .8]])


class FakeProbes:
    def __init__(self):
        self.calls = 0

    def predict(self, embeddings):
        self.calls += 1
        assert embeddings.shape == (2, 2)
        return {
            "material": {"classes": ["bronze", "porcelain"], "probabilities": np.array([[.95, .05], [.05, .95]], dtype=np.float32)},
            "object_type": {"classes": ["bowl", "vase"], "probabilities": np.array([[.95, .05], [.05, .95]], dtype=np.float32)},
        }


def make_probe_service(probes=None, report_overrides=None, encoder=None):
    catalog = [dict(items()[0]), {**items()[0], "item_id": "met_2"}]
    vectors = np.eye(2, dtype=np.float32)
    ids = [item["item_id"] for item in catalog]
    fingerprint = hashlib.sha256(vectors.tobytes() + json.dumps(ids).encode()).hexdigest()
    manifest = {"encoder": "clip", "data_sha256": "catalog-file-sha", "index_sha256": fingerprint,
                "index_version": "index-v1", "dimension": 2}
    report = {"index_sha256": fingerprint, "data_sha256": "catalog-file-sha", "index_version": "index-v1", "feature_dimension": 2,
              "tasks": {"material": {"status": "trained", "classes": ["bronze", "porcelain"]},
                        "object_type": {"status": "trained", "classes": ["bowl", "vase"]}}}
    report.update(report_overrides or {})
    return SearchService(catalog, embeddings=vectors, item_ids=ids, encoder=encoder or FakeEncoder(),
                         index_manifest=manifest, probes=probes or FakeProbes(), probe_report=report)


def test_visual_probe_changes_rank_and_precomputes_gallery_probabilities_once():
    probes = FakeProbes()
    service = make_probe_service(probes)
    baseline = service.search("porcelain vase", variant="hybrid", top_k=1)
    visual = service.search("porcelain vase", variant="visual", top_k=1)
    assert baseline["results"][0]["item_id"] == "met_1"
    assert visual["results"][0]["item_id"] == "met_2"
    assert visual["effective_variant"] == "visual" and not visual["degraded"]
    assert visual["results"][0]["score_breakdown"]["probe"] == pytest.approx(.95)
    service.search("vase", variant="visual")
    assert probes.calls == 1


def test_visual_probe_evidence_is_a_prediction_not_metadata_or_confirmation():
    result = make_probe_service().search("porcelain vase", variant="visual")
    row = result["results"][0]
    visual_evidence = [entry for entry in row["evidence"] if entry["kind"] == "visual_probe"]
    assert visual_evidence and all(0 <= entry["probability"] <= 1 for entry in visual_evidence)
    assert all("弱监督视觉分类预测" in entry["text"] and "并非人工确认" in entry["text"] for entry in visual_evidence)
    assert any(entry["kind"] == "metadata" for entry in row["evidence"])
    assert all(slot["basis"] == "metadata" for slot in row["matched_slots"])


@pytest.mark.parametrize("field,value", [("index_sha256", "another-space"), ("data_sha256", "another-dataset"), ("feature_dimension", 3), ("index_version", "another-index")])
def test_visual_rejects_probe_from_different_embedding_space_or_data(field, value):
    probes = FakeProbes()
    service = make_probe_service(probes, {field: value})
    assert not service.health()["probes_available"]
    response = service.search("porcelain vase", variant="visual")
    assert response["degraded"] and response["effective_variant"] == "hybrid"
    assert probes.calls == 0
    assert all(entry["kind"] != "visual_probe" for row in response["results"] for entry in row["evidence"])


def test_visual_without_probes_is_explicit_hybrid_fallback_with_identical_scores():
    service = SearchService(items(), embeddings=np.eye(2, dtype=np.float32), encoder=FakeEncoder())
    hybrid = service.search("vase", variant="hybrid")
    visual = service.search("vase", variant="visual")
    assert visual["degraded"] and visual["effective_variant"] == "hybrid"
    assert [(row["item_id"], row["score"]) for row in hybrid["results"]] == [(row["item_id"], row["score"]) for row in visual["results"]]
    assert any("降级" in warning for warning in visual["warnings"])


def test_baseline_remains_raw_dense_even_when_probes_are_loaded():
    class Recorder(FakeEncoder):
        def encode_texts(self, texts):
            self.seen = texts
            return super().encode_texts(texts)
    encoder = Recorder()
    result = make_probe_service(encoder=encoder).search("porcelain vase without bronze", variant="baseline")
    assert encoder.seen == ["porcelain vase without bronze"]
    assert result["results"][0]["item_id"] == "met_1"
    assert all(row["score_breakdown"]["probe"] is None for row in result["results"])


def test_stage_timings_measure_warm_encode_and_retrieval_without_repredicting():
    class TimedEncoder(FakeEncoder):
        def encode_texts(self, texts):
            time.sleep(.003)
            return super().encode_texts(texts)
    service = make_probe_service(encoder=TimedEncoder())
    response = service.search("porcelain vase", variant="visual")
    timings = response["timings"]
    assert timings["encode_ms"] >= 2
    assert timings["retrieve_ms"] >= 0 and timings["fusion_ms"] >= 0
    assert timings["warm_total_ms"] == timings["total_ms"]
    assert not timings["cold_start"] and timings["model_load_ms"] == 0
    assert sum(timings[key] for key in ("parse_ms", "encode_ms", "retrieve_ms", "fusion_ms")) <= timings["total_ms"] + .01


def test_cold_model_load_is_separate_from_warm_time_and_latency_samples(monkeypatch):
    class LazyEncoder(FakeEncoder):
        def __init__(self, *args, **kwargs):
            time.sleep(.005)
        def encode_texts(self, texts):
            time.sleep(.002)
            return super().encode_texts(texts)
    monkeypatch.setitem(sys.modules, "curio.represent.encoders", SimpleNamespace(Encoder=LazyEncoder))
    service = SearchService(items(), embeddings=np.eye(2, dtype=np.float32))
    cold = service.search("vase")["timings"]
    warm = service.search("vase")["timings"]
    assert cold["cold_start"] and cold["model_load_ms"] >= 4
    assert cold["warm_total_ms"] + cold["model_load_ms"] == pytest.approx(cold["total_ms"], abs=.01)
    assert not warm["cold_start"] and warm["model_load_ms"] == 0
    assert service.metrics()["latency_sample_count"] == 1
    assert service.metrics()["cold_start_searches"] == 1


def test_real_probe_loader_requires_exact_index_manifest_and_prepares_probabilities(tmp_path):
    catalog = items()
    ids = [item["item_id"] for item in catalog]
    vectors = np.eye(2, dtype=np.float32)
    index = tmp_path / "index"
    probes = tmp_path / "probes"
    index.mkdir()
    probes.mkdir()
    fingerprint = hashlib.sha256(vectors.tobytes() + json.dumps(ids).encode()).hexdigest()
    manifest = {"encoder": "clip", "data_sha256": "data-file", "index_sha256": fingerprint,
                "index_version": "v1", "dimension": 2}
    manifest_bytes = json.dumps(manifest).encode()
    (index / "manifest.json").write_bytes(manifest_bytes)
    (index / "item_ids.json").write_text(json.dumps(ids))
    np.save(index / "image_embeddings.npy", vectors)
    report = {"index_sha256": fingerprint, "data_sha256": "data-file", "index_version": "v1", "feature_dimension": 2,
              "index_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
              "tasks": {"material": {"status": "trained", "classes": ["bronze", "porcelain"]}}}
    (probes / "report.json").write_text(json.dumps(report))
    np.savez(probes / "material.npz", weights=np.array([[-1, 1], [1, -1]], dtype=np.float32), bias=np.zeros(2), classes=np.array(["bronze", "porcelain"]))
    service = SearchService(catalog, index_dir=index, probes_path=probes, encoder=FakeEncoder())
    assert service.health()["probes_available"]
    result = service.search("porcelain vase", variant="visual")
    assert any(entry["kind"] == "visual_probe" for entry in result["results"][0]["evidence"])
    report["index_manifest_sha256"] = "mismatched-manifest"
    (probes / "report.json").write_text(json.dumps(report))
    rejected = SearchService(catalog, index_dir=index, probes_path=probes, encoder=FakeEncoder())
    assert not rejected.health()["probes_available"]
    assert "index_manifest_sha256" in rejected.probe_warning


def test_negated_probe_class_does_not_receive_positive_visual_bonus():
    result = make_probe_service().search("vase not porcelain", variant="visual")
    entries = [entry for row in result["results"] for entry in row["evidence"] if entry["kind"] == "visual_probe"]
    assert entries and all(entry["slot"] == "object_type" and entry["value"] == "vase" for entry in entries)


def test_5000_item_gallery_reuses_probe_predictions_across_warm_requests():
    n, dimension = 5000, 16
    rng = np.random.default_rng(5523)
    vectors = rng.normal(size=(n, dimension)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    catalog = [{"item_id": f"synthetic_{index}", "title": "Porcelain vase", "object_type": "vase", "material": "porcelain"} for index in range(n)]
    ids = [item["item_id"] for item in catalog]
    fingerprint = hashlib.sha256(vectors.tobytes() + json.dumps(ids).encode()).hexdigest()
    report = {"index_sha256": fingerprint, "data_sha256": "synthetic-data", "index_version": "synthetic-v1", "feature_dimension": dimension,
              "tasks": {"material": {"status": "trained", "classes": ["bronze", "porcelain"]}}}
    class GalleryProbes:
        calls = 0
        def predict(self, embeddings):
            self.calls += 1
            probability = 1 / (1 + np.exp(-embeddings[:, 0]))
            return {"material": {"classes": ["bronze", "porcelain"], "probabilities": np.column_stack((1 - probability, probability))}}
    class GalleryEncoder:
        def encode_texts(self, texts):
            return np.eye(1, dimension, dtype=np.float32)
    probes = GalleryProbes()
    service = SearchService(catalog, embeddings=vectors, item_ids=ids, encoder=GalleryEncoder(), probes=probes, probe_report=report,
                            index_manifest={"encoder": "clip", "index_version": "synthetic-v1", "data_sha256": "synthetic-data", "index_sha256": fingerprint})
    for _ in range(3):
        response = service.search("porcelain vase", variant="visual", top_k=10)
        assert len(response["results"]) == 10 and response["effective_variant"] == "visual"
        assert response["timings"]["retrieve_ms"] > 0 and response["timings"]["fusion_ms"] > 0
    assert probes.calls == 1
    assert service.health()["indexed_images"] == n
    assert service.metrics()["latency_sample_count"] == 3


def test_image_only_fuses_dense_and_cached_probe_predictions_without_metadata_scoring(monkeypatch):
    probes = FakeProbes()
    service = make_probe_service(probes)
    def forbidden(*args, **kwargs):
        raise AssertionError("image_only used metadata scoring")
    monkeypatch.setattr(service.lexical, "score", forbidden)
    monkeypatch.setattr(service, "_slot_score", forbidden)
    response = service.search("porcelain vase", variant="image_only")
    assert response["effective_variant"] == "image_only" and not response["degraded"]
    assert [row["item_id"] for row in response["results"]] == ["met_2", "met_1"]
    for row in response["results"]:
        details = row["score_breakdown"]
        assert details["bm25"] is None and details["slots"] is None and details["negative_conflicts"] is None
        assert details["rrf_ranks"].keys() == {"dense"}
        assert details["rrf_normalized"] is None and details["scoring_mode"] == "image_only"
        assert row["score"] == pytest.approx(details["dense_rank_normalized"] + details["probe_bonus"], abs=1e-6)
        assert row["matched_slots"] == []
        assert {entry["kind"] for entry in row["evidence"]} == {"similarity", "visual_probe"}
    service.search("porcelain vase", variant="image_only")
    assert probes.calls == 1


def test_image_only_metadata_mutation_does_not_change_rank_or_score_but_explicit_facets_filter():
    first, second = make_probe_service(), make_probe_service()
    second.items[0].update(title="Bronze bowl", medium="Bronze", tags=["lotus"])
    second.items[1].update(title="Black textile", medium="Silk", tags=["fish"])
    second.attributes = [second._attributes(item) for item in second.items]
    query = "porcelain vase not dragon"
    left, right = first.search(query, variant="image_only"), second.search(query, variant="image_only")
    assert [(row["item_id"], row["score"]) for row in left["results"]] == [(row["item_id"], row["score"]) for row in right["results"]]
    response = second.search(query, variant="image_only", filters={"material": ["bronze"]})
    assert [row["item_id"] for row in response["results"]] == ["met_1"]
    assert all("仅依据元数据冲突" not in warning for warning in left["warnings"] + left["parsed"]["warnings"])


@pytest.mark.parametrize("with_probes", [False, True])
@pytest.mark.parametrize("query", ["porcelain vase with dragon", "porcelain vase not bronze, without dragon"])
def test_image_only_direct_embeddings_are_invariant_to_all_catalog_text(with_probes, query, monkeypatch):
    original = items()
    scrambled = [{**item, "title": "Unrelated textile", "description": "bronze dragon covered bowl",
                  "medium": "Silk", "object_type": "textile", "object_name": "Textile", "classification": "Fabric",
                  "material": "silk", "motifs": ["fish"], "tags": ["fish"], "colors": ["black"], "shape": ["handles"],
                  "culture": "unrelated", "period": "unrelated"} for item in original]
    vectors, ids = np.eye(2, dtype=np.float32), [item["item_id"] for item in original]
    def forbidden(*args, **kwargs):
        raise AssertionError("image_only read a metadata ranking channel")
    services = [SearchService(catalog, embeddings=vectors.copy(), item_ids=ids, encoder=FakeEncoder())
                for catalog in (original, scrambled)]
    for service in services:
        assert service.manifest == {}
        if with_probes:
            service.probe_predictions = FakeProbes().predict(vectors)
        monkeypatch.setattr(service.lexical, "score", forbidden)
        monkeypatch.setattr(service, "_slot_score", forbidden)
    left, right = [service.search(query, variant="image_only") for service in services]
    assert [(row["item_id"], row["rank"], row["score"], row["score_breakdown"]) for row in left["results"]] == [
        (row["item_id"], row["rank"], row["score"], row["score_breakdown"]) for row in right["results"]]
    assert left["results"][0]["title"] != right["results"][0]["title"]


@pytest.mark.parametrize("configured,query", [(False, "vase not dragon"), (True, "a green object not dragon")])
def test_image_only_without_applicable_probes_is_exact_rewritten_dense(configured, query):
    service = make_probe_service() if configured else SearchService(items(), embeddings=np.eye(2, dtype=np.float32), encoder=FakeEncoder())
    response = service.search(query, variant="image_only")
    assert response["effective_variant"] == "dense_rewritten"
    assert response["degraded"] is (not configured)
    assert [row["score"] for row in response["results"]] == [1.0, 0.0]
    assert all(row["matched_slots"] == [] and row["score_breakdown"]["probe"] is None for row in response["results"])
    assert all(entry["kind"] == "similarity" for row in response["results"] for entry in row["evidence"])


def test_image_only_has_no_metadata_fallback_when_image_embeddings_are_unavailable(monkeypatch):
    service = SearchService(items())
    def forbidden(*args, **kwargs):
        raise AssertionError("image_only used metadata fallback")
    monkeypatch.setattr(service.lexical, "score", forbidden)
    monkeypatch.setattr(service, "_slot_score", forbidden)
    response = service.search("porcelain vase", variant="image_only")
    assert response["results"] == [] and response["degraded"]
    assert response["effective_variant"] == "unavailable" and response["retrieval_mode"] == "unavailable"
    assert any("不返回元数据替代" in warning for warning in response["warnings"])


def test_image_only_dense_rank_ties_are_not_broken_by_item_ids():
    service = make_probe_service()
    service.embeddings[:] = np.array([[1, 0], [1, 0]], dtype=np.float32)
    service.probe_predictions["material"]["probabilities"][:] = .5
    service.probe_predictions["object_type"]["probabilities"][:] = .5
    response = service.search("porcelain vase", variant="image_only")
    rows = response["results"]
    assert rows[0]["score"] == rows[1]["score"]
    assert all(row["score_breakdown"]["rrf_ranks"] == {"dense": 1.5} for row in rows)


def test_image_only_zero_weight_preserves_dense_ordering():
    service = make_probe_service()
    service.probe_weight = 0
    response = service.search("porcelain vase", variant="image_only")
    assert [row["item_id"] for row in response["results"]] == ["met_1", "met_2"]
    assert all(row["score_breakdown"]["probe_bonus"] == 0 for row in response["results"])


def test_image_only_accepts_independent_sigmoid_heads_for_all_supported_query_groups():
    class DetailProbes:
        def predict(self, embeddings):
            return {task: {"classes": classes, "probabilities": np.array([[.2, .3], [.9, .8]], dtype=np.float32)}
                    for task, classes in {"colors": ["blue", "white"], "shape": ["narrow_mouth", "round_body"], "motifs": ["dragon", "lotus"]}.items()}
    tasks = {task: {"status": "trained", "classes": classes, "activation": "sigmoid"}
             for task, classes in {"colors": ["blue", "white"], "shape": ["narrow_mouth", "round_body"], "motifs": ["dragon", "lotus"]}.items()}
    service = make_probe_service(DetailProbes(), {"tasks": tasks})
    response = service.search("blue narrow-necked vase with a dragon", variant="image_only")
    assert response["effective_variant"] == "image_only" and not response["degraded"]
    assert response["results"][0]["item_id"] == "met_2"
    evidence = [entry for entry in response["results"][0]["evidence"] if entry["kind"] == "visual_probe"]
    assert {entry["slot"] for entry in evidence} == {"colors", "shape", "motifs"}
    assert response["results"][0]["score_breakdown"]["probe"] == pytest.approx(.9)


def test_image_only_rejects_out_of_range_sigmoid_probabilities():
    class InvalidDetailProbes:
        def predict(self, embeddings):
            return {"shape": {"classes": ["narrow_mouth"], "probabilities": np.array([[1.1], [.3]])}}
    tasks = {"shape": {"status": "trained", "classes": ["narrow_mouth"], "activation": "sigmoid"}}
    service = make_probe_service(InvalidDetailProbes(), {"tasks": tasks})
    response = service.search("narrow mouth", variant="image_only")
    assert response["effective_variant"] == "dense_rewritten" and response["degraded"]
    assert not service.health()["probes_available"]


def test_image_only_negation_uses_complement_of_covered_image_probability_and_reports_unknown():
    service = make_probe_service()
    response = service.search("not bronze, without dragon", variant="image_only")
    assert response["effective_variant"] == "image_only"
    assert response["results"][0]["item_id"] == "met_2"
    assert response["probe_coverage"]["negations"] == [
        {"slot": "material", "value": "bronze", "status": "covered"},
        {"slot": "motifs", "value": "dragon", "status": "unknown"},
    ]
    evidence = [entry for entry in response["results"][0]["evidence"] if entry["kind"] == "visual_probe"]
    assert len(evidence) == 1 and evidence[0]["negated"] is True
    assert evidence[0]["value"] == "bronze" and evidence[0]["probability"] == pytest.approx(.05)
    assert evidence[0]["support_probability"] == pytest.approx(.95)
    assert "不能证明属性不存在" in evidence[0]["text"]
    assert response["results"][0]["score_breakdown"]["probe"] == pytest.approx(.95)


def test_image_only_unknown_negation_never_becomes_absence_evidence():
    response = make_probe_service().search("without handles", variant="image_only")
    assert response["effective_variant"] == "dense_rewritten"
    assert response["probe_coverage"]["negations"] == [{"slot": "shape", "value": "handles", "status": "unknown"}]
    assert all(entry["kind"] == "similarity" for row in response["results"] for entry in row["evidence"])


def make_lid_probe_service():
    class LidProbes:
        def predict(self, embeddings):
            # A lidded jar can have a low independent-lid class probability.
            return {"object_type": {"classes": ["jar", "lid"],
                                    "probabilities": np.array([[.99, .01], [.1, .9]], dtype=np.float32)}}
    tasks = {"object_type": {"status": "trained", "classes": ["jar", "lid"]}}
    return make_probe_service(LidProbes(), {"tasks": tasks})


@pytest.mark.parametrize("query,group", [("没有盖的罐子", "negations"), ("jar without a lid", "negations"),
                                        ("带盖的罐子", "positive"), ("covered jar", "positive")])
def test_image_only_lid_component_never_uses_independent_object_class(query, group):
    response = make_lid_probe_service().search(query, variant="image_only")
    assert response["effective_variant"] == "image_only" and not response["degraded"]
    assert {"slot": "object_type", "value": "lid", "status": "unknown",
            "reason": "component_presence_not_trained"} in response["probe_coverage"][group]
    assert any("仅表示独立盖子类别" in warning for warning in response["warnings"])
    assert any("该部件条件保持未知" in warning for warning in response["parsed"]["warnings"])
    rows = {row["item_id"]: row for row in response["results"]}
    assert rows["met_1"]["score_breakdown"]["probe"] == pytest.approx(.99)
    assert rows["met_2"]["score_breakdown"]["probe"] == pytest.approx(.1)
    for row in rows.values():
        evidence = [entry for entry in row["evidence"] if entry["kind"] == "visual_probe"]
        assert len(evidence) == 1 and evidence[0]["value"] == "jar" and not evidence[0]["negated"]


@pytest.mark.parametrize("query,group", [("without a lid", "negations"), ("covered object", "positive"),
                                        ("an object with a lid", "positive"), ("有盖器物", "positive"),
                                        ("带盖器物", "positive"), ("lidded", "positive")])
def test_image_only_lid_component_only_query_falls_back_to_dense(query, group):
    response = make_lid_probe_service().search(query, variant="image_only")
    assert response["effective_variant"] == "dense_rewritten" and not response["degraded"]
    assert response["probe_coverage"][group] == [{"slot": "object_type", "value": "lid", "status": "unknown",
                                                 "reason": "component_presence_not_trained"}]
    assert [row["score"] for row in response["results"]] == [1.0, 0.0]
    assert all(row["score_breakdown"]["probe"] is None and row["score_breakdown"]["probe_bonus"] == 0
               for row in response["results"])
    assert all(entry["kind"] == "similarity" for row in response["results"] for entry in row["evidence"])


@pytest.mark.parametrize("query", ["a lid", "a porcelain lid", "盖子", "瓷盖"])
def test_image_only_independent_lid_positive_intent_retains_object_class_support(query):
    response = make_lid_probe_service().search(query, variant="image_only")
    assert response["effective_variant"] == "image_only"
    assert {"slot": "object_type", "value": "lid", "status": "covered"} in response["probe_coverage"]["positive"]
    assert response["results"][0]["item_id"] == "met_2"
    evidence = [entry for entry in response["results"][0]["evidence"] if entry["kind"] == "visual_probe"]
    assert len(evidence) == 1 and evidence[0]["value"] == "lid" and not evidence[0]["negated"]
    assert response["results"][0]["score_breakdown"]["probe"] == pytest.approx(.9)


def test_diagnostic_visual_keeps_existing_lid_class_behavior():
    response = make_lid_probe_service().search("covered jar", variant="visual")
    assert response["effective_variant"] == "visual" and response["probe_coverage"] is None
    evidence = [entry for entry in response["results"][0]["evidence"] if entry["kind"] == "visual_probe"]
    assert {entry["value"] for entry in evidence} == {"jar", "lid"}
    assert response["results"][0]["score_breakdown"]["probe"] == pytest.approx(.5)


def test_evidence_keeps_actual_cosine_gap_when_zero_weight_still_uses_rank_scores():
    class CloseEncoder(FakeEncoder):
        def encode_texts(self, texts):
            return np.array([[.8, .6]], dtype=np.float32)

    service = make_probe_service(encoder=CloseEncoder())
    service.probe_weight = 0
    response = service.search("porcelain vase", variant="image_only", top_k=1)
    evidence = response["confidence"]
    assert evidence["ranking_separation"]["candidate_count"] == 2
    assert evidence["ranking_separation"]["top1_top2_margin"] == pytest.approx(.2)
    assert evidence["ranking_separation"]["top1"] == pytest.approx(.8)
    assert response["results"][0]["score"] == 1.0
    assert evidence["signal_availability"]["probes_applied"] is False
    assert evidence["calibration"]["correctness_probability"] is None


def test_evidence_measures_constraint_agreement_and_identifies_different_fused_winner():
    service = make_probe_service()
    response = service.search("porcelain vase", variant="image_only")
    evidence = response["confidence"]
    assert evidence["signal_agreement"]["status"] == "disagree"
    assert evidence["signal_agreement"]["compared"] == 2
    assert evidence["scope"] == "dense_recall_before_fusion"
    assert evidence["dense_top_item_id"] == "met_1"
    assert evidence["final_top_item_id"] == "met_2"
    agreeing = service.search("bronze bowl", variant="image_only")["confidence"]
    assert agreeing["signal_agreement"]["status"] == "agree"
    # Agreement can occur on weak support when it is the only eligible candidate.
    restricted = service.search("porcelain vase", variant="image_only", allowed_item_ids={"met_1"})
    assert restricted["confidence"]["signal_agreement"]["status"] == "agree"
    assert restricted["confidence"]["ranking_separation"]["top1_top2_margin"] is None


def test_search_evidence_counts_era_unsupported_and_unparsed_query_as_unverified():
    service = make_probe_service()
    response = service.search("Ming vase two around the rim", variant="image_only")
    coverage = response["confidence"]["visual_coverage"]
    assert coverage["covered"] == 1 and coverage["total"] == 4
    assert coverage["ratio"] == .25
    assert response["confidence"]["level"] == "limited"
    unknown = service.search("an ineffable mysterious thing", variant="image_only")["confidence"]
    assert unknown["level"] == "limited"
    assert unknown["visual_coverage"]["ratio"] is None
    assert unknown["parser"]["confidence_calibrated"] is False


@pytest.mark.parametrize("query", ["Ming dynasty porcelain vase", "明代青花瓷瓶", "1400-1500 porcelain vase"])
def test_level3_strips_era_before_visual_rewrite_and_records_original_constraints(query):
    class Recorder(FakeEncoder):
        def encode_texts(self, texts):
            self.texts = texts
            return super().encode_texts(texts)

    encoder = Recorder()
    service = SearchService(items(), embeddings=np.eye(2, dtype=np.float32), encoder=encoder)
    response = service.level3_search(query)
    assert response["parsed"]["era"] is not None
    assert response["stage1_parsed"]["era"] is None
    assert "Ming" not in encoder.texts[0] and "明代" not in encoder.texts[0]
    assert "1400" not in encoder.texts[0] and "1500" not in encoder.texts[0]
    assert response["stage1_query"] == response["stage1_parsed"]["query"]
    assert response["confidence"]["scope"] == "level3_mixed_ranking"
    assert response["stage1_confidence"]["scope"] == "dense_recall_before_fusion"
    assert response["confidence"]["ranking_separation"]["top1_top2_margin"] is None


def test_level3_missing_labels_are_unknown_and_explicit_excluded_labels_are_violated():
    catalog = [
        {"item_id": "a", "title": "Unknown object"},
        {"item_id": "b", "title": "Bronze bowl", "tags": ["dragon"]},
    ]
    service = SearchService(catalog, embeddings=np.eye(2, dtype=np.float32), encoder=FakeEncoder())
    response = service.level3_search("porcelain vase without dragon")
    rows = {row["item_id"]: row for row in response["results"]}
    assert all(entry["status"] == "unknown" for entry in rows["a"]["level3_evidence"])
    missing_positive = [entry for entry in rows["b"]["level3_evidence"] if not entry["negated"]]
    assert all(entry["status"] == "unknown" for entry in missing_positive)
    excluded = [entry for entry in rows["b"]["level3_evidence"] if entry["negated"]]
    assert excluded[0]["status"] == "violated"
    assert excluded[0]["reason"] == "excluded_catalogue_label_present"
    assert all(entry["basis"] == "catalogue" for row in rows.values() for entry in row["level3_evidence"])


@pytest.mark.parametrize("start,end,status", [
    (None, 1500, "unknown"), (1400, None, "unknown"), ("1400", 1500, "unknown"),
    (1500, 1400, "unknown"), (float("nan"), 1500, "unknown"), (True, 1500, "unknown"),
    (1200, 1600, "satisfied"), (1700, 1800, "violated"),
])
def test_level3_date_evidence_requires_valid_complete_catalogue_range(start, end, status):
    from curio.query import parse_query

    actual, reason = SearchService._level3_catalogue_status(
        {"year_from": start, "year_to": end}, {}, parse_query("Ming vase"), "era", "Ming")
    assert actual == status
    if status == "satisfied":
        assert reason == "catalogue_date_range_overlap"


def test_level3_fusion_weight_is_explicit_unvalidated_and_tied_visual_scores_stay_tied():
    catalog = [{"item_id": "a", "title": "Unknown"}, {"item_id": "b", "title": "Vase"}]
    service = SearchService(catalog, embeddings=np.array([[1, 0], [1, 0]], dtype=np.float32), encoder=FakeEncoder())
    response = service.level3_search("vase", catalogue_weight=0)
    assert response["fusion_config"] == {"catalogue_weight": 0, "visual_weight": 1, "validation_status": "unvalidated_prototype"}
    assert response["results"][0]["score"] == response["results"][1]["score"]
    reranked = service.level3_search("vase", catalogue_weight=1)
    assert reranked["results"][0]["item_id"] == "b"
    assert reranked["confidence"]["level"] == "limited"
    with pytest.raises(ValueError):
        service.level3_search("vase", catalogue_weight=1.1)


def test_level3_end_to_end_timing_includes_stage2_and_exposes_unverified_visual_conditions(monkeypatch):
    service = SearchService(items(), embeddings=np.eye(2, dtype=np.float32), encoder=FakeEncoder())
    clock = [0.0]
    monkeypatch.setattr("curio.retrieve.engine.time.perf_counter", lambda: clock[0])
    original = service._level3_catalogue_status

    def delayed_evidence(*args, **kwargs):
        clock[0] += .125
        return original(*args, **kwargs)

    monkeypatch.setattr(service, "_level3_catalogue_status", delayed_evidence)
    response = service.level3_search("Ming blue vase two around the rim")
    assert response["timings"]["stage1_total_ms"] == 0
    assert response["timings"]["stage2_ms"] >= 125
    assert response["timings"]["total_ms"] >= 125
    assert response["confidence"] != response["stage1_confidence"]
    unverified = response["results"][0]["level3_unverified_constraints"]
    assert {entry["slot"] for entry in unverified} == {"colors", "unsupported"}
    assert all(entry["status"] == "unknown" for entry in unverified)


def test_level3_era_only_does_not_return_arbitrary_visual_results():
    service = SearchService(items(), embeddings=np.eye(2, dtype=np.float32), encoder=FakeEncoder())
    response = service.level3_search("Ming dynasty")
    assert response["stage1_query"] == ""
    assert response["results"] == []
    assert response["confidence"]["level"] == "unknown"
