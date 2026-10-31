"""Exercise a real running local API and save measured acceptance evidence."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from io import BytesIO
import json
from pathlib import Path
import time
import threading

import numpy as np
from PIL import Image
import requests

QUERIES = (
    "青花瓷碗",
    "a blue and white porcelain bowl",
    "有龙纹的瓷瓶",
    "a bronze mirror",
    "没有盖的罐子",
    "莲花纹的碗",
    "a pale green jade ring",
    "a standing figure",
    "红色的细颈瓶",
    "a textile with floral patterns",
)


def check(base_url, count=200, workers=4, variants=("image_only", "baseline", "hybrid", "visual")):
    if not variants or len(variants) != len(set(variants)):
        raise ValueError("Choose distinct variants")
    if count < len(variants) or workers < 1:
        raise ValueError("Need at least one measured request per variant and positive workers")
    base_url = base_url.rstrip("/")
    request_records, record_lock = [], threading.Lock()

    def get(path):
        response = requests.get(base_url + path, timeout=60)
        response.raise_for_status()
        return response

    def search(query, variant="image_only", *, phase="check", sequence=None, **kwargs):
        start = time.perf_counter()
        payload = {"query": query, "variant": variant, "top_k": 12, **kwargs}
        response = requests.post(
            base_url + "/api/search",
            timeout=120,
            json=payload,
        )
        response.raise_for_status()
        body = response.json()
        elapsed = (time.perf_counter() - start) * 1000
        if body["degraded"]:
            raise RuntimeError(f"Degraded real search: {body.get('warnings')}")
        ids = [x["item_id"] for x in body["results"]]
        assert len(ids) == len(set(ids)), "Duplicate result IDs"
        assert len(ids) <= payload["top_k"], "Too many results"
        assert body["variant"] == variant, "Server used a different requested variant"
        assert all(np.isfinite(x["score"]) for x in body["results"])
        assert all(not x.get("verified", False) for x in body["results"]), (
            "Unsubstantiated verification flag"
        )
        with record_lock:
            request_records.append({"phase": phase, "sequence": sequence, "request": payload,
                                    "request_id": body["request_id"], "effective_variant": body["effective_variant"],
                                    "result_ids": ids, "result_count": len(ids), "wall_ms": elapsed,
                                    "timings": body["timings"], "degraded": body["degraded"],
                                    "warnings": body.get("warnings", []), "probe_coverage": body.get("probe_coverage")})
        return body, elapsed

    before = get("/api/health").json()
    metrics_before = get("/api/metrics").json()
    config_before = metrics_before.get("runtime_config")
    if not isinstance(config_before, dict) or not config_before.get("index_manifest_sha256"):
        raise ValueError("API lacks fingerprinted runtime configuration; update the server before acceptance")
    assert before["dense_available"], before
    assert before["items"] == before["indexed_images"] > 0, before
    first, first_wall = search(QUERIES[0], phase="first_request")
    assert len(first["results"]) > 0, "Empty gallery search"
    checked_images = []
    for result in first["results"][:5]:
        item = get("/api/items/" + result["item_id"]).json()
        assert "image_path" not in item and "thumb_path" not in item
        assert item["license"] == "CC0" and item["source_url"].startswith("https://")
        with Image.open(BytesIO(get(item["image_url"]).content)) as image:
            image.load()
            checked_images.append({"item_id": item["item_id"], "size": image.size})
    empty, _ = search("碗", phase="strict_empty_facet", filters={"object_type": ["__impossible__"]})
    assert empty["results"] == [], "Explicit facet was silently relaxed"
    for query in ("", "   "):
        assert (
            requests.post(base_url + "/api/search", json={"query": query}, timeout=60).status_code
            == 422
        )
    assert get("/").headers["content-type"].startswith("text/html")
    facets = get("/api/facets").json()
    assert facets
    measurements = []
    for variant in variants:
        # Explicit warmup excluded from the following measurements.
        search(QUERIES[0], variant, phase="warmup")
    for i in range(count):
        variant = variants[i % len(variants)]
        body, elapsed = search(QUERIES[i % len(QUERIES)], variant, phase="warm_sequential", sequence=i)
        measurements.append({"variant": variant, "query": QUERIES[i % len(QUERIES)], "request_id": body["request_id"],
                             "effective_variant": body["effective_variant"], "wall_ms": elapsed, **body["timings"]})

    def concurrent_call(i):
        _, elapsed = search(QUERIES[i % len(QUERIES)], "image_only", phase="concurrent_image_only", sequence=i)
        return elapsed

    concurrent_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        concurrent = list(pool.map(concurrent_call, range(count)))
    concurrent_seconds = time.perf_counter() - concurrent_start
    metrics_after = get("/api/metrics").json()
    if metrics_after.get("runtime_config") != config_before:
        raise ValueError("API runtime configuration changed during acceptance")
    if metrics_after.get("runtime_environment", {}).get("process_id") != metrics_before.get("runtime_environment", {}).get("process_id"):
        raise ValueError("API server process changed during acceptance")

    def stats(values):
        return {
            "requests": len(values),
            "p50_ms": float(np.median(values)),
            "p95_ms": float(np.quantile(values, 0.95)),
            "max_ms": float(max(values)),
        }

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_url": base_url,
        "schema": "curio-system-acceptance-v2",
        "runtime_config": config_before,
        "runtime_environment_before": metrics_before.get("runtime_environment"),
        "runtime_environment_after": metrics_after.get("runtime_environment"),
        "requested": {"count_per_phase": count, "workers": workers, "variants": list(variants), "queries": list(QUERIES), "top_k": 12},
        "health_before": before,
        "health_after": get("/api/health").json(),
        "first_request": {
            "wall_ms": first_wall,
            "timings": first["timings"],
            "note": "Only cold if encoder_loaded was false before this check",
        },
        "checks": {
            "images_decode": checked_images,
            "strict_empty_facet": True,
            "invalid_input": True,
            "no_local_paths": True,
            "unique_finite_results": True,
        },
        "warm_sequential": {
            variant: stats([x["wall_ms"] for x in measurements if x["variant"] == variant])
            for variant in variants
        },
        "concurrent_image_only": {
            **stats(concurrent),
            "workers": workers,
            "seconds": concurrent_seconds,
            "requests_per_second": count / concurrent_seconds,
        },
        "measurements": measurements,
        "request_records": request_records,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--count", type=int, default=200)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--variants", default="image_only,baseline,hybrid,visual")
    p.add_argument("--output", default="artifacts/system_acceptance.json")
    args = p.parse_args()
    report = check(args.url, args.count, args.workers, tuple(args.variants.split(",")))
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {k: v for k, v in report.items() if k != "measurements"}, ensure_ascii=False, indent=2
        )
    )


if __name__ == "__main__":
    main()
