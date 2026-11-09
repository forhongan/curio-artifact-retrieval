"""Record changed catalogue facets with their original museum evidence."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "artifacts/cv_v4_20261108/catalogue_v3_primary_plus_candidates/labels_catalogue_v2.jsonl"
NEW = ROOT / "artifacts/cv_v4_20261108/catalogue_v4_semantic_review/labels_catalogue_v2.jsonl"


def read(path):
    return {row["item_id"]: row for row in
            (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)}


def main():
    old, new = read(OLD), read(NEW)
    if old.keys() != new.keys():
        raise ValueError("Catalogue versions contain different item IDs")
    changes, counts = [], Counter()
    for item_id in old:
        before, after = old[item_id]["derived_ontology"], new[item_id]["derived_ontology"]
        delta = {key: {"old": before[key], "new": after[key]}
                 for key in ("object_type", "material", "motifs") if before[key] != after[key]}
        if not delta:
            continue
        counts.update(delta.keys())
        changes.append({
            "item_id": item_id, "changes": delta, "official_catalog": new[item_id]["official_catalog"],
            "old_facets": {key: before[key] for key in ("object_form", "material_facets", "motif_facets")},
            "new_facets": {key: after[key] for key in
                           ("object_form", "material_facets", "motif_facets", "unattributed_mentions")},
        })
    (NEW.parent / "semantic_changes.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in changes), encoding="utf-8")
    report = {
        "changed_items": len(changes), "fields": dict(counts),
        "old_sha256": hashlib.sha256(OLD.read_bytes()).hexdigest(),
        "new_sha256": hashlib.sha256(NEW.read_bytes()).hexdigest(),
    }
    (NEW.parent / "semantic_changes_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
