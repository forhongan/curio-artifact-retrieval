"""Verify the portable HTTP contract using Python's standard library only."""

import argparse
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:8015")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    base = args.base.rstrip("/")

    def request(path, payload=None, method=None, headers=None):
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        req = Request(
            base + path,
            data=data,
            method=method,
            headers={"Content-Type": "application/json", **(headers or {})},
        )
        with urlopen(req, timeout=30) as response:
            return response.status, json.load(response), response.headers

    _, health, _ = request("/api/health")
    assert health["items"] == 4692 and health["live_inference"] is False
    _, capabilities, _ = request("/api/boundary/capabilities")
    assert len(capabilities["supported_queries"]) == 8
    expected = {
        "青花瓷": 295,
        "梅瓶": 90,
        "葫芦瓶": 39,
        "双腹葫芦瓶": 36,
        "三节葫芦瓶": 1,
        "红色碗": 3,
        "绿色瓷瓶": 29,
        "绿色窄口瓷瓶": 10,
    }
    checks, sets = [], {}
    for query in capabilities["supported_queries"]:
        previous = None
        for alias in query["aliases"]:
            status, result, headers = request(
                "/api/search/set/jobs",
                {"query": alias, "method": "catalogue_visual"},
                headers={"Origin": "http://localhost:5173"},
            )
            assert status == 202 and result["complete"] and result["status"] == "complete"
            assert headers["Access-Control-Allow-Origin"] == "http://localhost:5173"
            assert result["live_inference"] is False
            assert result["execution_mode"] == "snapshot_replay"
            assert result["total"] == result["processed"] == 4692
            assert result["matched_count"] == query["yes"]
            assert result["rejected_count"] == query["no"]
            assert result["uncertain_count"] == 0
            assert "decisions" not in result
            current = {r["item_id"] for r in result["results"]}
            assert len(current) == len(result["results"]) == result["matched_count"]
            if previous is not None:
                assert previous == (result["job_id"], current)
            previous = (result["job_id"], current)
            if alias in expected:
                assert len(current) == expected[alias]
                sets[alias] = current
        job_id = result["job_id"]
        _, full, _ = request(f"/api/search/set/jobs/{job_id}?decisions=true")
        decisions = full["decisions"]
        assert len(decisions) == len({r["item_id"] for r in decisions}) == 4692
        assert all(type(r["yes"]) is bool for r in decisions)
        assert {r["item_id"] for r in decisions if r["yes"]} == current
        _, cancelled, _ = request(f"/api/search/set/jobs/{job_id}/cancel", method="POST")
        assert cancelled["complete"] and cancelled["status"] == "complete"
        for item in result["results"]:
            assert item["yes"] and item["condition_evidence"]
        item = result["results"][0]
        _, detail, _ = request("/api/items/" + item["item_id"])
        assert detail["item_id"] == item["item_id"] and detail["title"]
        with urlopen(base + item["image_url"], timeout=30) as response:
            assert response.status == 200
            assert response.headers["Content-Type"] == "image/jpeg"
            assert response.read().startswith(b"\xff\xd8")
        checks.append({"aliases": query["aliases"], "yes": len(current), "no": query["no"]})
    assert set(sets) == set(expected)
    assert sets["绿色窄口瓷瓶"] <= sets["绿色瓷瓶"]
    assert sets["双腹葫芦瓶"] <= sets["葫芦瓶"]
    assert sets["三节葫芦瓶"] <= sets["葫芦瓶"]
    try:
        request("/api/search/set/jobs", {"query": "紫色花瓶", "method": "catalogue_visual"})
    except HTTPError as error:
        assert error.code == 422
    else:
        raise AssertionError("Unknown query must not return fabricated results")
    report = {
        "passed": True,
        "queries": checks,
        "binary_decisions": 37536,
        "bilingual": True,
        "cors": True,
        "details_and_images": True,
        "unknown_query_422": True,
        "completed_cancel_noop": True,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
