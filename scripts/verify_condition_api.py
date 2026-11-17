"""Verify the explicit v7 comparison HTTP route against all forty frozen experiment sets."""

import argparse
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    expected = json.loads((ROOT / "artifacts/set_boundary_v7/selected_ids.json").read_text())
    queries = [
        json.loads(s) for s in (ROOT / "eval/relevance_v5/queries.jsonl").read_text().splitlines()
    ]
    checks = []
    for lang in ["en", "zh"]:
        for query in queries:
            request = urllib.request.Request(
                args.url + "/api/search/set",
                data=json.dumps(
                    {"query": query[f"text_{lang}"], "top_k": 1, "method": "condition_boundary"}
                ).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=180) as response:
                result = json.load(response)
            assert result["method"] == "condition_boundary"
            assert result["boundary"]["effective_model"] == "conditions"
            assert result["gallery_count"] == 4692
            assert result["returned"] == result["matched_count"] == len(result["results"])
            assert len(result["preview"]) == min(1, result["matched_count"])
            assert set(result["matched_item_ids"]) == set(
                expected[f"conditions/{lang}"][query["qid"]]
            ), (query["qid"], lang)
            checks.append(
                {
                    "qid": query["qid"],
                    "language": lang,
                    "selected": result["matched_count"],
                    "latency_ms": result["latency_ms"],
                    "matches_offline": True,
                }
            )
            print(query["qid"], lang, result["matched_count"], "matched", flush=True)
    output = ROOT / "docs/research/set_boundary_v7/runtime_verification.json"
    output.write_text(
        json.dumps(
            {
                "real_http": True,
                "method": "condition_boundary",
                "explicit_comparison": True,
                "checks": checks,
            },
            indent=2,
        )
        + "\n"
    )
    print(output)


if __name__ == "__main__":
    main()
