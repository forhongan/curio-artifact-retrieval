from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

from fastapi.testclient import TestClient
import numpy as np
import pytest

from curio.retrieve.condition_boundary import FEATURE_NAMES, prompt_plan
from curio.retrieve.condition_parser import parse_conditions
from curio.retrieve.condition_runtime import ConditionBoundary, vocabulary_sha256
from curio.retrieve.engine import SearchService
from curio.retrieve.visual_sets import VisualSetService
from curio.serve.app import SetRequest, create_app


class Encoder:
    metadata = {"model_id": "test-model", "revision": "test-revision"}

    def __init__(self):
        self.calls = []

    def encode_texts(self, texts):
        self.calls.extend(texts)
        return np.array([[1.0, 0.0] for _ in texts], dtype=np.float32)


@pytest.fixture
def runtime(tmp_path):
    items = [{"item_id": str(i), "title": "irrelevant metadata"} for i in range(10)]
    vectors = np.array([[1, i + 1] if i < 5 else [-1, i + 1] for i in range(10)], dtype=np.float32)
    service = SearchService(items, embeddings=vectors, encoder=Encoder(), root=tmp_path)
    models = {}
    for name, width in [("conditions", 9), ("refit_base", 3)]:
        for lang in ["en", "zh"]:
            models[f"{name}/{lang}"] = {
                "mean": [0] * width,
                "scale": [1] * width,
                "coefficients": [8] + [0] * (width - 1),
                "intercept": 0,
                "threshold": 0.5,
            }
    bundle = {
        "schema": "condition-boundary-v7",
        "feature_names": FEATURE_NAMES,
        "vocabulary_sha256": vocabulary_sha256(),
        "index_sha256": service.index_sha256,
        "gallery_count": len(items),
        "text_encoder": Encoder.metadata,
        "models": models,
    }
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps(bundle))
    sets = VisualSetService(service)
    sets._condition_boundary = ConditionBoundary(service, path)
    return service, sets, path


def test_parser_reproduces_all_frozen_query_plans_without_reading_candidate_metadata():
    base = Path(__file__).parents[1] / "eval/relevance_v5/queries.jsonl"
    for line in base.read_text(encoding="utf-8").splitlines():
        query = json.loads(line)
        for lang in ["en", "zh"]:
            parsed = parse_conditions(query[f"text_{lang}"])
            assert not parsed["fallback_reasons"], (query["qid"], lang, parsed)
            assert prompt_plan(parsed["conditions"], lang) == prompt_plan(query["conditions"], lang)


@pytest.mark.parametrize(
    "query",
    [
        "not a red vase",
        "没有龙纹的瓷瓶",
        "green or red vase",
        "two porcelain vases",
        "a painting of a vase",
        "梅瓶",
        "a vase with elephants",
    ],
)
def test_unsupported_semantics_route_to_raw_text(query, runtime):
    service, sets, _ = runtime
    result = sets.search(query, method="condition_boundary")
    assert result["boundary"]["effective_model"] == "refit_base"
    assert result["encoded_query"] == query
    assert service.encoder.calls == [query, "a photograph of a museum object"]
    assert result["notices"]


def test_narrow_mouth_is_not_silently_promoted_to_long_neck():
    parsed = parse_conditions("绿色窄口瓷瓶")
    assert ["shape", "narrow_mouth"] in parsed["conditions"]
    assert ["tag", "long_neck"] not in parsed["conditions"]
    assert prompt_plan(parsed["conditions"], "zh")["missing"] == ["shape/narrow_mouth"]


def test_v7_full_gallery_preview_and_support_semantics(runtime):
    service, sets, _ = runtime
    result = sets.search("green vase", top_k=1, method="condition_boundary")
    assert result["method"] == "condition_boundary"
    assert result["matched_count"] == result["returned"] == 5
    assert len(result["preview"]) == 1
    assert result["matched_count"] + result["rejected_count"] == 10
    assert result["results"][0]["score"] == result["results"][0]["support_score"]
    assert "support_mode_posterior" not in result["results"][0]
    assert not result["provenance"]["candidate_catalogue_used_for_decisions"]
    for item in service.items:
        item["title"] = "conflicting catalogue"
        item["era"] = "wrong"
    assert (
        sets.search("green vase", method="condition_boundary")["matched_item_ids"]
        == result["matched_item_ids"]
    )


def test_api_without_method_uses_new_default(runtime, tmp_path, monkeypatch):
    monkeypatch.delenv("CURIO_SET_PROBES", raising=False)
    service, sets, _ = runtime
    app = create_app(service=service, root=tmp_path)
    app.state.visual_set_service._condition_boundary = sets._condition_boundary
    app.state.visual_set_service._strict_boundary.runtime = sets._condition_boundary
    response = TestClient(app).post("/api/search/set", json={"query": "green vase", "top_k": 1})
    assert SetRequest(query="vase").method == "strict_conditions"
    assert response.status_code == 200
    assert response.json()["method"] == "strict_conditions"
    # Identical fake embeddings provide no per-attribute contrast evidence.
    assert response.json()["matched_count"] == 0
    assert response.json()["uncertain_count"] == 5
    assert not response.json()["preview"]


def test_missing_condition_is_visible(runtime):
    _, sets, _ = runtime
    result = sets.search("乾隆时期瓷器", method="condition_boundary")
    assert result["boundary"]["missing_conditions"] == ["era/Qianlong"]
    assert result["notices"]


@pytest.mark.parametrize(
    "mismatch", ["index", "schema", "vocabulary", "features", "scale", "encoder", "missing_file"]
)
def test_wrong_runtime_artifacts_fail_closed(runtime, mismatch):
    service, sets, path = runtime
    bundle = json.loads(path.read_text())
    if mismatch == "index":
        service.index_sha256 = "wrong"
    elif mismatch == "schema":
        bundle["schema"] = "wrong"
    elif mismatch == "vocabulary":
        bundle["vocabulary_sha256"] = "wrong"
    elif mismatch == "features":
        bundle["feature_names"] = []
    elif mismatch == "scale":
        bundle["models"]["conditions/en"]["scale"][0] = 0
    elif mismatch == "encoder":
        service.encoder.metadata = {"model_id": "wrong", "revision": "wrong"}
    path.write_text(json.dumps(bundle))
    if mismatch == "missing_file":
        path.unlink()
    with pytest.raises((OSError, ValueError)):
        sets.search("green vase", method="condition_boundary")


def test_cached_prompt_scores_are_reused_and_concurrent_results_match(runtime):
    service, sets, _ = runtime
    first = sets.search("green vase", method="condition_boundary")
    calls = len(service.encoder.calls)
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(
            pool.map(lambda _: sets.search("green vase", method="condition_boundary"), range(3))
        )
    assert all(r["matched_item_ids"] == first["matched_item_ids"] for r in results)
    assert len(service.encoder.calls) == calls + 3  # only the original query is re-encoded


def test_frozen_text_cache_preserves_decisions_and_validates_hash(runtime):
    service, sets, path = runtime
    cache = path.with_name("texts.npz")
    np.savez_compressed(
        cache, texts=np.array(["green vase"]), vectors=np.array([[1.0, 0.0]], dtype=np.float32)
    )
    bundle = json.loads(path.read_text())
    bundle["text_cache"] = {
        "file": cache.name,
        "sha256": hashlib.sha256(cache.read_bytes()).hexdigest(),
    }
    path.write_text(json.dumps(bundle))
    original = service._dense
    service._dense = lambda text: np.full(10, -0.1) if text == "green vase" else original(text)
    result = sets.search("green vase", method="condition_boundary")
    assert result["matched_count"] == 5
    assert result["boundary"]["frozen_query_vector"]
    cache.write_bytes(b"changed")
    sets._condition_boundary = ConditionBoundary(service, path)
    with pytest.raises(ValueError, match="text cache changed"):
        sets.search("green vase", method="condition_boundary")


def test_frozen_text_scores_follow_catalog_order(runtime):
    service, sets, _ = runtime
    boundary = sets._condition_boundary
    boundary._text_vectors = {"cached": np.array([1.0, 0.0], dtype=np.float32)}
    service.vector_rows = list(reversed(service.vector_rows))
    expected = (service.embeddings @ np.array([1.0, 0.0], dtype=np.float32))[::-1]
    np.testing.assert_allclose(boundary._score("cached"), expected)
