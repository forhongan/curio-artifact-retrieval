import hashlib
import json

from fastapi.testclient import TestClient

from curio.retrieve.engine import SearchService
from curio.serve.app import create_app


class FakeRuntime:
    def __init__(self, root):
        self.root = root
        self.calls = []

    def run(self, query, output, **kwargs):
        self.calls.append((query, kwargs))
        output.mkdir(parents=True)
        image = output / "reference.jpg"
        image.write_bytes(b"audited reference bytes")
        selected = [
            {
                "reference_id": "cma_99",
                "title": "External dish",
                "source": "cleveland",
                "date": "1736-1795",
                "license": "CC0",
                "image_path": str(image.relative_to(self.root)),
                "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
            }
        ]
        (output / "direct_acquisition.json").write_text(
            json.dumps({"accepted": selected, "request_summary": {"live": 2, "cached": 0}}),
            encoding="utf-8",
        )
        return {
            "status": "ranked",
            "gallery_count": 1,
            "reference_ids": ["cma_99"],
            "results": [{"item_id": "met_1", "rank": 1}],
        }


def test_dynamic_api_passes_user_description_and_serves_only_audited_reference(tmp_path):
    runtime = FakeRuntime(tmp_path)
    service = SearchService([{"item_id": "met_1", "title": "Local image", "source": "met"}], root=tmp_path)
    client = TestClient(create_app(service=service, root=tmp_path, dynamic_runtime=runtime))
    response = client.post(
        "/api/search/reference/dynamic",
        json={"query": "Qianlong famille rose porcelain", "reference_limit": 3, "top_k": 7},
    )
    assert response.status_code == 200
    result = response.json()
    assert runtime.calls[0][0] == "Qianlong famille rose porcelain"
    assert runtime.calls[0][1]["reference_limit"] == 3
    assert runtime.calls[0][1]["top_k"] == 7
    assert runtime.calls[0][1]["mode"] == "direct"
    assert runtime.calls[0][1]["search_text"] is None
    assert result["candidate_catalogue_used_for_ranking"] is False
    assert result["results"][0]["image_url"] == "/api/images/met_1"
    assert result["results"][0]["source"] == "met"
    url = result["references"][0]["image_url"]
    assert client.get(url).content == b"audited reference bytes"
    assert "image_path" not in result["references"][0]
    assert client.get(url.replace("cma_99", "cma_100")).status_code == 404
    image = tmp_path / "artifacts/dynamic_reference_api" / result["audit_id"] / "reference.jpg"
    image.write_bytes(b"changed after audit")
    assert client.get(url).status_code == 503


def test_dynamic_api_rejects_empty_description_and_local_paths(tmp_path):
    runtime = FakeRuntime(tmp_path)
    client = TestClient(
        create_app(service=SearchService([], root=tmp_path), root=tmp_path, dynamic_runtime=runtime)
    )
    assert client.post("/api/search/reference/dynamic", json={"query": "   "}).status_code == 422
    assert (
        client.post(
            "/api/search/reference/dynamic",
            json={"query": "vase", "acquisition": "some-local-file.json"},
        ).status_code
        == 422
    )
    assert not runtime.calls


def test_qualified_reference_mode_and_aggregation_forwarded(tmp_path):
    runtime = FakeRuntime(tmp_path)
    client = TestClient(
        create_app(
            service=SearchService([{"item_id": "met_1"}], root=tmp_path),
            root=tmp_path,
            dynamic_runtime=runtime,
        )
    )
    response = client.post(
        "/api/search/reference/dynamic",
        json={
            "query": "Qianlong famille rose vase",
            "mode": "qualified",
            "aggregation": "query_aligned",
        },
    )
    assert response.status_code == 200
    assert runtime.calls[0][1]["mode"] == "qualified"
    assert runtime.calls[0][1]["aggregation"] == "query_aligned"
