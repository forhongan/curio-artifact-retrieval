from io import BytesIO
import hashlib
import json

from fastapi.testclient import TestClient
import numpy as np
from PIL import Image
import pytest

from curio.retrieve.engine import SearchService
from curio.retrieve.image_search import ImageSearchService, reference_similarity, unit_rows
from curio.serve.app import create_app


class PixelEncoder:
    def encode_images(self, images):
        return np.tile([1., 0.], (len(images), 1))


def service(root):
    items = [{"item_id": "a", "title": "unrelated words", "source": "met"},
             {"item_id": "b", "title": "perfect query match", "source": "met"}]
    return SearchService(items, embeddings=np.array([[1., 0.], [0., 1.]]),
                         encoder=PixelEncoder(), root=root)


def test_reference_aggregations_and_centroid_ranking_equivalence():
    gallery = np.array([[1., 0.], [0., 1.], [.5, .5]])
    refs = np.array([[1., 0.], [.8, .2]])
    mean, details = reference_similarity(gallery, refs, "mean")
    maximum, _ = reference_similarity(gallery, refs, "max")
    centroid, _ = reference_similarity(gallery, refs, "centroid")
    assert np.allclose(mean, details.mean(1))
    assert np.allclose(maximum, details.max(1))
    # Normalizing a single mean prototype changes scale, not order.
    assert np.array_equal(np.argsort(mean), np.argsort(centroid))


@pytest.mark.parametrize("matrix", [[], [[0, 0]], [[float('nan'), 1]], [1, 2]])
def test_bad_features_rejected(matrix):
    with pytest.raises(ValueError):
        unit_rows(matrix)


def test_visual_ranking_ignores_metadata_and_supports_exclusion(tmp_path):
    backend = service(tmp_path)
    images = ImageSearchService(backend)
    first = images.search([Image.new("RGB", (20, 20))])
    assert first["results"][0]["item_id"] == "a"
    backend.by_id["a"]["title"], backend.by_id["b"]["title"] = "perfect query match", "anything"
    assert images.search([Image.new("RGB", (20, 20))])["results"][0]["item_id"] == "a"
    assert images.search([Image.new("RGB", (20, 20))], exclude_ids={"a"})["results"][0]["item_id"] == "b"
    assert first["results"][0]["matched_slots"] == []
    assert first["confidence"]["level"] == "unknown"


def test_image_api_invalid_upload_missing_index_and_valid_pixels(tmp_path):
    backend = service(tmp_path)
    client = TestClient(create_app(service=backend, root=tmp_path))
    assert client.post("/api/search/image", content=b"invalid").status_code == 422
    assert client.post("/api/search/image?method=invented", content=b"invalid").status_code == 422
    buffer = BytesIO()
    Image.new("RGB", (20, 20)).save(buffer, "PNG")
    response = client.post("/api/search/image", content=buffer.getvalue())
    assert response.status_code == 200 and response.json()["results"][0]["item_id"] == "a"
    backend.embeddings = None
    assert client.post("/api/search/image", content=buffer.getvalue()).status_code == 503


def test_bank_selection_checks_image_hash_and_outside_paths(tmp_path):
    bank = tmp_path / "data/references"
    bank.mkdir(parents=True)
    path = bank / "ref.png"
    Image.new("RGB", (20, 20)).save(path)
    row = {"reference_id": "ref", "image_path": "data/references/ref.png",
           "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    target = bank / "bank.jsonl"
    target.write_text(json.dumps(row), encoding="utf-8")
    search = ImageSearchService(service(tmp_path))
    assert search.search_query("vase", reference_ids=["ref"])["reference_count"] == 1
    row["image_sha256"] = "tampered"
    target.write_text(json.dumps(row), encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint"):
        search.search_query("vase", reference_ids=["ref"])
    row["image_path"] = "pyproject.toml"
    target.write_text(json.dumps(row), encoding="utf-8")
    with pytest.raises(ValueError, match="outside"):
        search.search_query("vase", reference_ids=["ref"])


def test_reference_missing_is_explicit_and_never_text_fallback(tmp_path):
    response = ImageSearchService(service(tmp_path)).search_query("明代青花瓷")
    assert response["retrieval_mode"] == "reference_unavailable"
    assert response["results"] == []


def test_expanded_gallery_requires_matching_base_and_has_separate_scope(tmp_path):
    backend = service(tmp_path)
    catalog = tmp_path / "data/snapshots/collection_v3_expanded.jsonl"
    catalog.parent.mkdir(parents=True)
    extra = {"item_id": "new", "title": "calligraphy", "source": "met"}
    rows = backend.items + [extra]
    catalog.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    index = tmp_path / "artifacts/siglip_expanded"
    index.mkdir(parents=True)
    np.save(index / "image_embeddings.npy", [[1., 0.], [0., 1.], [.9, .1]])
    (index / "item_ids.json").write_text(json.dumps(["a", "b", "new"]))
    manifest = {"parent_index_sha256": backend.index_sha256, "encoder": "clip"}
    (index / "manifest.json").write_text(json.dumps(manifest))
    # PixelEncoder has no text API; expanded lexical comparison must still be
    # explicitly scoped and not present a stale frozen evaluation.
    client = TestClient(create_app(service=backend, root=tmp_path))
    response = client.post("/api/search/expanded", json={"query": "calligraphy", "variant": "lexical"})
    assert response.status_code == 200
    assert response.json()["gallery_scope"] == "expanded_v3_research"
    assert response.json()["results"][0]["item_id"] == "new"
    assert client.get("/api/health").json()["expanded_catalogue_count"] == 3
    assert client.get("/api/items/new").status_code == 200
    manifest["parent_index_sha256"] = "other index"
    (index / "manifest.json").write_text(json.dumps(manifest))
    other = TestClient(create_app(service=backend, root=tmp_path))
    assert other.post("/api/search/expanded", json={"query": "vase"}).status_code == 503
    assert other.get("/api/items/new").status_code == 404


def test_specialist_abstention_is_explicit(tmp_path, monkeypatch):
    search = ImageSearchService(service(tmp_path))
    monkeypatch.setattr(search, "_specialist", lambda *args: ([{"validity": {"shape": False}}], {}))
    image = Image.new("RGB", (20, 20))
    with pytest.raises(ValueError, match="rejected"):
        search.search([image], method="shape")
    mixed = search.search([image], method="siglip_shape")
    assert mixed["scoring"] == "cosine_similarity"
    assert any("rejected" in warning for warning in mixed["warnings"])
