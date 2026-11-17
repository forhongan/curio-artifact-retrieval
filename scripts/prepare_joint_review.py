"""Publish the already frozen, unlabeled follow-up as a blind review queue.

No labels or model predictions are inferred or overwritten by this export.
"""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / "artifacts/set_boundary_followup"
    queries = json.loads((source / "new_queries.json").read_text())
    items = json.loads((source / "new_items.json").read_text())
    manifest = json.loads((source / "manifest.json").read_text())
    for name in ["new_queries.json", "new_items.json"]:
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != manifest["hashes"][name]:
            raise RuntimeError("Frozen follow-up sample changed: " + name)
    rows = [
        {
            "key": q["qid"] + ":" + r["item_id"],
            "qid": q["qid"],
            "item_id": r["item_id"],
            "query_zh": q["text_zh"],
            "query_en": q["text_en"],
            "object_group": r["object_group"],
            "split": r["split"],
            "image_url": "/api/images/" + r["item_id"],
            "image_sha256": r["image_sha256"],
        }
        for q in queries
        for r in items
        if q["split"] == r["split"]
    ]
    data = {"schema": "curio-blind-review-v1", "source_hashes": manifest["hashes"], "rows": rows}
    data["bundle_id"] = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
    target = ROOT / "web/public/assets/boundary-review.json"
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    print(f"Exported {len(rows)} unlabelled review pairs to {target}")


if __name__ == "__main__":
    main()
