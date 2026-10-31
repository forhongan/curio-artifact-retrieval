import importlib.util
import json
from pathlib import Path
from urllib.parse import urlsplit

from fastapi.testclient import TestClient
import httpx
import numpy as np
from PIL import Image
import pytest

from curio.retrieve.engine import SearchService
from curio.serve.app import create_app

SPEC = importlib.util.spec_from_file_location("system_check", Path(__file__).resolve().parents[1] / "scripts/system_check.py")
SCRIPT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCRIPT)


@pytest.fixture
def fake_api(tmp_path, monkeypatch):
    image = tmp_path / "data/processed/images/met_1.jpg"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (32, 32), "white").save(image)
    frontend = tmp_path / "web/dist/index.html"
    frontend.parent.mkdir(parents=True)
    frontend.write_text("<html>fixture</html>")
    rows = [{"item_id": "met_1", "title": "Fixture", "image_path": "data/processed/images/met_1.jpg",
             "license": "CC0", "source_url": "https://example.invalid/item"}]

    class Encoder:
        metadata = {"device": "cpu", "model_id": "synthetic", "revision": "synthetic",
                    "adapter_path": "private/path/never/expose"}

        def encode_texts(self, texts):
            return np.array([[1, 0]], dtype=np.float32)

    service = SearchService(rows, embeddings=np.array([[1, 0]], dtype=np.float32), encoder=Encoder(), root=tmp_path,
                            index_manifest={"data_sha256": "synthetic-data", "encoder_metadata": {"model_id": "synthetic"}})
    service.index_manifest_sha256 = "synthetic-manifest"
    service.probe_configured = True
    service.probe_files_sha256 = {"material.npz": "synthetic-head"}
    service.probe_predictions = {"material": {"classes": ["porcelain"], "probabilities": np.ones((1, 1))}}
    client = TestClient(create_app(service=service, root=tmp_path))

    def path(url):
        parts = urlsplit(url)
        return parts.path + ("?" + parts.query if parts.query else "")

    def get(url, **kwargs):
        return client.get(path(url))

    monkeypatch.setattr(SCRIPT.requests, "get", get)
    monkeypatch.setattr(SCRIPT.requests, "post", lambda url, **kwargs: client.post(path(url), json=kwargs["json"]))
    return get


def test_acceptance_records_real_request_payloads_ids_configuration_and_concurrency(fake_api):
    report = SCRIPT.check("http://fixture.invalid", count=8, workers=2)
    assert report["schema"] == "curio-system-acceptance-v2"
    assert report["runtime_config"]["data_sha256"] == "synthetic-data"
    assert report["runtime_config"]["index_manifest_sha256"] == "synthetic-manifest"
    assert report["runtime_environment_after"]["encoder_device"] == "cpu"
    assert "private/path" not in json.dumps(report)
    assert report["concurrent_image_only"]["requests"] == report["requested"]["count_per_phase"] == 8
    records = report["request_records"]
    assert len(records) == 2 + 4 + 8 + 8
    assert len({row["request_id"] for row in records}) == len(records)
    assert len(report["measurements"]) == 8
    concurrent = [row for row in records if row["phase"] == "concurrent_image_only"]
    assert sorted(row["sequence"] for row in concurrent) == list(range(8))
    assert all(row["request"]["query"] in SCRIPT.QUERIES and row["request"]["variant"] == "image_only" for row in concurrent)
    assert all(row["wall_ms"] >= 0 and row["result_ids"] == ["met_1"] for row in concurrent)


@pytest.mark.parametrize("changed", ["configuration", "process"])
def test_acceptance_rejects_configuration_or_process_changes_mid_run(fake_api, monkeypatch, changed):
    calls = 0

    def get(url, **kwargs):
        nonlocal calls
        response = fake_api(url, **kwargs)
        if url.endswith("/api/metrics"):
            calls += 1
            if calls == 2:
                body = response.json()
                if changed == "configuration":
                    body["runtime_config"]["data_sha256"] = "other"
                else:
                    body["runtime_environment"]["process_id"] += 1
                return httpx.Response(200, json=body, request=response.request)
        return response

    monkeypatch.setattr(SCRIPT.requests, "get", get)
    with pytest.raises(ValueError, match="changed during acceptance"):
        SCRIPT.check("http://fixture.invalid", count=4, workers=2, variants=("image_only",))
