import gzip
import hashlib
import json
import subprocess
import sys

from fastapi.testclient import TestClient
import pytest

from curio.retrieve.catalogue_query import compile_catalogue
from curio.serve.boundary_snapshot import create_snapshot_app


def fixture_bundle(tmp_path, *, corrupt=False):
    plan = compile_catalogue("梅瓶")
    items = [{"item_id": "one", "title": "Meiping vase"}, {"item_id": "two", "title": "Cup"}]
    report = {
        "job_id": "opaque-job",
        "complete": True,
        "status": "complete",
        "total": 2,
        "processed": 2,
        "matched_count": 1,
        "rejected_count": 1,
        "uncertain_count": 0,
        "plan": plan.payload(),
        "results": [{"item_id": "one", "yes": True}],
        "decisions": [{"item_id": "one", "yes": True}, {"item_id": "two", "yes": False}],
    }
    if corrupt:
        report["decisions"].pop()
    bundle = {
        "schema": "curio-boundary-snapshot-v1",
        "version": "test",
        "items": items,
        "queries": [{"aliases": ["梅瓶", "meiping vase"], "report": report}],
    }
    data = gzip.compress(json.dumps(bundle).encode())
    (tmp_path / "snapshot.json.gz").write_bytes(data)
    (tmp_path / "manifest.json").write_text(
        json.dumps({"snapshot_sha256": hashlib.sha256(data).hexdigest()})
    )
    (tmp_path / "images").mkdir()
    (tmp_path / "images/one.jpg").write_bytes(b"image fixture")


def test_snapshot_contract_bilingual_and_no_fake_unknown_results(tmp_path):
    fixture_bundle(tmp_path)
    with TestClient(create_snapshot_app(tmp_path)) as client:
        zh = client.post(
            "/api/search/set/jobs", json={"query": "梅瓶", "method": "catalogue_visual"}
        )
        en = client.post("/api/search/set/jobs", json={"query": "meiping vase"})
        assert zh.status_code == 202 and zh.json()["job_id"] == en.json()["job_id"]
        assert zh.json()["execution_mode"] == "snapshot_replay" and not zh.json()["live_inference"]
        assert "decisions" not in zh.json()
        assert (
            len(client.get("/api/search/set/jobs/opaque-job?decisions=true").json()["decisions"])
            == 2
        )
        assert client.post("/api/search/set/jobs/opaque-job/cancel").json()["complete"]
        assert client.post("/api/search/set/jobs", json={"query": "红色碗"}).status_code == 422
        assert client.post("/api/search/set/jobs", json={"query": "  "}).status_code == 422
        assert (
            client.post(
                "/api/search/set/jobs", json={"query": "梅瓶", "method": "joint_vlm"}
            ).status_code
            == 422
        )
        assert client.get("/api/items/one").json()["title"] == "Meiping vase"
        assert client.get("/api/images/one").status_code == 200
        assert client.get("/api/images/missing").status_code == 404
        assert client.get("/api/search/set/jobs/missing").status_code == 404
        response = client.options(
            "/api/search/set/jobs",
            headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"},
        )
        assert response.headers["access-control-allow-origin"] == "http://localhost:5173"
        assert client.get("/api/boundary/capabilities").json()["gallery_count"] == 2


def test_snapshot_rejects_incomplete_data_and_checksum_change(tmp_path):
    fixture_bundle(tmp_path, corrupt=True)
    with pytest.raises(ValueError, match="Incomplete"):
        create_snapshot_app(tmp_path)
    (tmp_path / "snapshot.json.gz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        create_snapshot_app(tmp_path)


def test_snapshot_import_does_not_load_search_engine_or_torch():
    command = "import sys; import curio.serve.boundary_snapshot; assert 'curio.retrieve.engine' not in sys.modules; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", command], check=True)
