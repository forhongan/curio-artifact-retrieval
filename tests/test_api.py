import json
import hashlib

from fastapi.testclient import TestClient
import numpy as np
import pytest

from curio.retrieve.engine import SearchService
from curio.serve.app import create_app


class FakeEncoder:
    def encode_texts(self, texts):
        return np.array([[1.0, 0.0]], dtype=np.float32)


def catalog(tmp_path):
    image = tmp_path / "data" / "processed" / "images" / "met_1.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"not really jpeg but route only streams")
    items = [{"item_id": "met_1", "title": "Dragon vase", "object_name": "Vase", "medium": "Porcelain", "tags": ["dragon"], "year_from": 1400, "year_to": 1500, "source": "met", "image_path": "data/processed/images/met_1.jpg", "thumb_path": "data/processed/images/met_1.jpg"}]
    data = tmp_path / "items.jsonl"
    data.write_text("\n".join(json.dumps(item) for item in items), encoding="utf-8")
    return items


def test_api_search_validation_health_items_and_images(tmp_path):
    items = catalog(tmp_path)
    service = SearchService(items, embeddings=np.array([[1, 0]], dtype=np.float32), encoder=FakeEncoder(), root=tmp_path)
    client = TestClient(create_app(service=service, root=tmp_path, feedback_path=tmp_path / "feedback.jsonl"))
    assert client.get("/api/health").json()["dense_available"] is True
    response = client.post("/api/search", json={"query": "porcelain vase", "top_k": 2})
    assert response.status_code == 200
    assert response.json()["results"][0]["item_id"] == "met_1"
    assert client.post("/api/search", json={"query": "vase", "top_k": 51}).status_code == 422
    assert "image_path" not in client.get("/api/items/met_1").json()
    assert client.get("/api/images/met_1").content == b"not really jpeg but route only streams"


def test_image_route_blocks_unlisted_paths_and_path_traversal(tmp_path):
    items = catalog(tmp_path)
    items[0]["image_path"] = str(tmp_path / "secret.txt")
    (tmp_path / "secret.txt").write_text("secret")
    service = SearchService(items, root=tmp_path)
    client = TestClient(create_app(service=service, root=tmp_path))
    assert client.get("/api/images/met_1").status_code == 404
    assert client.get("/api/images/../../secret.txt").status_code == 404


def test_image_route_only_serves_processed_image_directory(tmp_path):
    items = catalog(tmp_path)
    external = tmp_path / "data" / "processed" / "metadata.txt"
    external.write_text("metadata")
    items[0]["image_path"] = "data/processed/metadata.txt"
    service = SearchService(items, root=tmp_path)
    client = TestClient(create_app(service=service, root=tmp_path))
    assert client.get("/api/images/met_1").status_code == 404


def test_feedback_persists_and_metrics_count(tmp_path):
    items = catalog(tmp_path)
    target = tmp_path / "feedback.jsonl"
    client = TestClient(create_app(service=SearchService(items), root=tmp_path, feedback_path=target))
    result = client.post("/api/feedback", json={"request_id": "r1", "item_id": "met_1", "relevant": True})
    assert result.status_code == 201 and result.json()["persisted"] is True
    assert json.loads(target.read_text()) ["item_id"] == "met_1"
    assert client.get("/api/metrics").json()["feedback_count"] == 1


def test_visual_variant_is_accepted_and_explicitly_degrades_without_probes(tmp_path):
    service = SearchService(catalog(tmp_path), embeddings=np.array([[1, 0]], dtype=np.float32), encoder=FakeEncoder(), root=tmp_path)
    client = TestClient(create_app(service=service, root=tmp_path))
    response = client.post("/api/search", json={"query": "porcelain vase", "variant": "visual"})
    assert response.status_code == 200
    assert response.json()["degraded"]
    assert response.json()["effective_variant"] == "hybrid"
    assert {"encode_ms", "retrieve_ms", "fusion_ms", "warm_total_ms"} <= response.json()["timings"].keys()


def test_api_defaults_to_image_only_and_explicitly_reports_dense_fallback(tmp_path):
    service = SearchService(catalog(tmp_path), embeddings=np.array([[1, 0]], dtype=np.float32), encoder=FakeEncoder(), root=tmp_path)
    client = TestClient(create_app(service=service, root=tmp_path))
    response = client.post("/api/search", json={"query": "porcelain vase"})
    assert response.status_code == 200
    data = response.json()
    assert data["variant"] == "image_only" and data["effective_variant"] == "dense_rewritten"
    assert data["results"][0]["matched_slots"] == []
    assert {entry["kind"] for entry in data["results"][0]["evidence"]} == {"similarity"}


def test_api_image_only_does_not_return_metadata_when_vectors_are_missing(tmp_path):
    client = TestClient(create_app(service=SearchService(catalog(tmp_path), root=tmp_path), root=tmp_path))
    response = client.post("/api/search", json={"query": "vase", "variant": "image_only"})
    assert response.status_code == 200
    assert response.json()["results"] == [] and response.json()["effective_variant"] == "unavailable"


def test_level3_research_endpoint_separates_visual_and_catalogue_evidence(tmp_path):
    items = catalog(tmp_path)
    service = SearchService(items, embeddings=np.array([[1, 0]], dtype=np.float32), encoder=FakeEncoder(), root=tmp_path)
    client = TestClient(create_app(service=service, root=tmp_path))
    response = client.post("/api/search/level3", json={"query": "Ming porcelain vase", "top_k": 1, "recall_k": 12})
    assert response.status_code == 200
    data = response.json()
    assert data["variant"] == "level3_research"
    assert data["retrieval_mode"] == "two_stage_research"
    assert data["results"][0]["level3_evidence"]
    assert all(entry["basis"] == "catalogue" for entry in data["results"][0]["level3_evidence"])


@pytest.mark.parametrize("mismatch", [None, "manifest", "index", "weight", "probe_files", "configured"])
def test_api_only_exposes_evaluation_for_loaded_index_and_probe_files(tmp_path, monkeypatch, mismatch):
    items = catalog(tmp_path)
    index = tmp_path / "index"
    index.mkdir()
    vectors, ids = np.array([[1, 0]], dtype=np.float32), ["met_1"]
    index_sha = hashlib.sha256(vectors.tobytes() + json.dumps(ids).encode()).hexdigest()
    manifest = {"encoder": "clip", "index_sha256": index_sha, "data_sha256": "synthetic-data",
                "dimension": 2, "items": 1, "index_version": "synthetic-index"}
    np.save(index / "image_embeddings.npy", vectors)
    (index / "item_ids.json").write_text(json.dumps(ids))
    (index / "manifest.json").write_text(json.dumps(manifest))
    probes = tmp_path / "probes"
    probes.mkdir()
    np.savez(probes / "material.npz", weights=np.array([[1, 0], [0, 1]]),
             bias=np.zeros(2), classes=np.array(["porcelain", "bronze"]))
    (probes / "report.json").write_text(json.dumps({
        "data_sha256": "synthetic-data", "index_sha256": index_sha,
        "index_manifest_sha256": hashlib.sha256((index / "manifest.json").read_bytes()).hexdigest(),
        "index_version": "synthetic-index", "feature_dimension": 2,
        "tasks": {"material": {"status": "trained", "classes": ["porcelain", "bronze"]}},
    }))
    service = SearchService(items, index_dir=index, encoder=FakeEncoder(), probes_path=probes, root=tmp_path)
    assert service.health()["probes_available"]
    report = {"index_manifest_sha256": service.index_manifest_sha256, "index": {"index_sha256": service.index_sha256},
              "probes": {"configured": True, "weight": service.probe_weight,
                         "files_sha256": dict(service.probe_files_sha256)}, "summary": []}
    if mismatch == "manifest":
        report["index_manifest_sha256"] = "other"
    elif mismatch == "index":
        report["index"]["index_sha256"] = "other"
    elif mismatch == "weight":
        report["probes"]["weight"] = .3
    elif mismatch == "probe_files":
        report["probes"]["files_sha256"]["material.npz"] = "other"
    elif mismatch == "configured":
        report["probes"]["configured"] = False
    report_path = tmp_path / "evaluation.json"
    report_path.write_text(json.dumps(report))
    monkeypatch.setenv("CURIO_METRICS", str(report_path))
    client = TestClient(create_app(service=service, root=tmp_path))
    response = client.get("/api/metrics").json()
    runtime = response["runtime_config"]
    assert runtime["index_manifest_sha256"] == service.index_manifest_sha256
    assert runtime["index_sha256"] == service.index_sha256 and runtime["data_sha256"] == "synthetic-data"
    assert runtime["probes"]["files_sha256"] == service.probe_files_sha256
    assert runtime["probes"]["weight"] == service.probe_weight and runtime["default_variant"] == "image_only"
    assert "process_id" in response["runtime_environment"] and "python" in response["runtime_environment"]
    assert str(tmp_path) not in json.dumps(runtime) + json.dumps(response["runtime_environment"])
    if mismatch:
        assert response["evaluation"] is None and response["evaluation_warning"]
    else:
        assert response["evaluation"] == report
