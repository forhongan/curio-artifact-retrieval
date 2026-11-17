"""Turn 120 directly inspected images into 720 query-specific assisted judgments."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from curio.evaluate.assisted_review import judge_assisted
from curio.retrieve.joint_pixels import color_observability

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--web-output", type=Path, help="Optional frontend JSON export path")
    args = parser.parse_args()
    base = ROOT / "eval/boundary_assisted_v10"
    bundle_path = base / "review_queue.json"
    bundle = json.loads(bundle_path.read_text())
    gallery_path = ROOT / "data/snapshots/collection_v2.jsonl"
    items = {r["item_id"]: r for r in map(json.loads, gallery_path.read_text().splitlines())}
    fact_path = base / "visual_facts.jsonl"
    raw_facts = list(map(json.loads, fact_path.read_text().splitlines()))
    facts = {r["item_id"]: r for r in raw_facts}
    assert len(facts) == len(raw_facts) == 120
    assert set(facts) == {r["item_id"] for r in bundle["rows"]}
    colors = {
        ident: color_observability(ROOT / items[ident]["image_path"])["observable"]
        for ident in facts
    }
    labels = []
    for row in bundle["rows"]:
        ident = row["item_id"]
        assert (
            hashlib.sha256((ROOT / items[ident]["image_path"]).read_bytes()).hexdigest()
            == row["image_sha256"]
        )
        labels.append(
            {
                **row,
                **judge_assisted(
                    facts[ident], items[ident], row["qid"], color_observable=colors[ident]
                ),
            }
        )
    summary = {
        "total": len(labels),
        "images_reviewed": len(facts),
        "positive": sum(r["grade"] == 3 for r in labels),
        "negative": sum(r["grade"] == 0 for r in labels),
        "color_excluded": sum(r["reason"] == "color_not_observable" for r in labels),
        "needs_review": sum(r["needs_review"] for r in labels),
        "per_query": {
            qid: dict(Counter(r["reason"] for r in labels if r["qid"] == qid))
            for qid in sorted({r["qid"] for r in labels})
        },
    }
    result = {
        "schema": "curio-assisted-review-v1",
        "bundle_id": bundle["bundle_id"],
        "summary": summary,
        "labels": labels,
        "provenance": {
            "method": "root AI directly viewed all six 20-image contact sheets and enlarged ambiguous motifs; catalogue medium used for material only; no student predictions used",
            "facts_sha256": hashlib.sha256(fact_path.read_bytes()).hexdigest(),
            "gallery_sha256": hashlib.sha256(gallery_path.read_bytes()).hexdigest(),
            "human_confirmed": False,
            "test_policy": "original test split is preserved; do not use test labels to fit or calibrate; AI-assisted evaluation is not human validation",
        },
    }
    if args.web_output:
        args.web_output.parent.mkdir(parents=True, exist_ok=True)
        args.web_output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    (base / "labels.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in labels)
    )
    (base / "manifest.json").write_text(
        json.dumps({k: v for k, v in result.items() if k != "labels"}, ensure_ascii=False, indent=2)
        + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
