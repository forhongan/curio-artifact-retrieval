"""Real HTTP smoke checks for the independently versioned visual routes."""
import argparse
from datetime import datetime, timezone
from io import BytesIO
import json
from pathlib import Path
import time

from PIL import Image
import requests


def main(base):
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "health": requests.get(base + "/api/health", timeout=10).json(), "checks": []}
    cases = [
        ("default", "/api/search", {"query": "绿色窄口瓷瓶", "top_k": 5}),
        ("reference_kangxi", "/api/search/reference", {"query": "康熙梅瓶", "top_k": 5}),
        ("reference_jiajing", "/api/search/reference", {"query": "嘉靖青花葫芦瓶", "top_k": 5, "method": "dinov2"}),
        ("reference_unavailable", "/api/search/reference", {"query": "万历青花瓷瓶", "top_k": 5}),
        ("reference_qualified", "/api/search/reference", {"query": "万历青花五彩瓶", "top_k": 5}),
        ("reference_patch", "/api/search/reference", {"query": "康熙梅瓶", "top_k": 5, "method": "dinov2_patch"}),
        ("expanded", "/api/search/expanded", {"query": "Chinese calligraphy album page", "top_k": 5}),
    ]
    for name, route, payload in cases:
        started = time.perf_counter()
        response = requests.post(base + route, json=payload, timeout=180)
        response.raise_for_status()
        data = response.json()
        if name == "reference_unavailable":
            assert data["retrieval_mode"] == "reference_unavailable" and not data["results"]
        elif name.startswith("reference"):
            assert data["retrieval_mode"] == "image_to_image" and data["reference_count"] >= 1
            assert all(not row["matched_slots"] for row in data["results"])
        if name == "reference_qualified":
            assert data["reference_constraint_status"] == "qualified_metadata" and data["reference_qualifications"]
        if name == "reference_patch":
            assert data["patch_diagnostics"]["recall_count"] == 100
            assert data["confidence"]["scope"] == "dinov2_cls_recall_before_patch_rerank"
        if name == "expanded":
            assert data["gallery_count"] == 4793 and data["gallery_scope"] == "expanded_v3_research"
        report["checks"].append({"name": name, "route": route, "payload": payload,
                                 "wall_ms": round((time.perf_counter() - started) * 1000, 3), "response": data})
        print(name, response.status_code, len(data["results"]), flush=True)
    uploaded = requests.post(base + "/api/search/image?method=siglip&top_k=5",
                             data=Path("data/references/images/met_46527.jpg").read_bytes(), timeout=120)
    uploaded.raise_for_status()
    report["checks"].append({"name": "raw_image_upload", "response": uploaded.json()})
    assert requests.post(base + "/api/search/image", data=b"not image", timeout=10).status_code == 422
    assert requests.post(base + "/api/search/reference", json={"query": " "}, timeout=10).status_code == 422
    extra = json.loads(Path("data/snapshots/expansion_v3.jsonl").read_text(encoding="utf-8").splitlines()[0])
    served = requests.get(base + f"/api/images/{extra['item_id']}", timeout=10)
    served.raise_for_status()
    Image.open(BytesIO(served.content)).verify()
    assert requests.get(base + f"/api/items/{extra['item_id']}", timeout=10).status_code == 200
    assert requests.get(base + "/api/references/met_44749/image", timeout=10).status_code == 200
    metrics = requests.get(base + "/api/metrics", timeout=10).json()
    assert metrics["runtime_config"]["probes"]["weight"] == 0
    report["runtime_config"] = metrics["runtime_config"]
    report["status"] = "passed"
    Path("artifacts/reference_eval/api_acceptance.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    main(parser.parse_args().base.rstrip("/"))
