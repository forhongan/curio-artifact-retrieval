"""Freeze relation-connected object families without moving a held-out family into train."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from curio.represent.validation import validate_object_splits


def freeze(source, relations, output, report):
    source, relations, output, report = map(Path, (source, relations, output, report))
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    relation = json.loads(relations.read_text(encoding="utf-8"))
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if relation["data_sha256"] != digest:
        raise ValueError("Relation report does not describe this snapshot")
    mapping = relation["item_to_group"]
    if set(mapping) != {row["item_id"] for row in rows}:
        raise ValueError("Relation mapping must cover every item exactly")
    grouped = defaultdict(list)
    for row in rows:
        grouped[mapping[row["item_id"]]].append(row)
    priority = {"train": 0, "dev": 1, "test": 2}
    group_splits = {group: max((row["split"] for row in members), key=priority.get)
                    for group, members in grouped.items()}
    # Preserve the independently authored blind dev/test galleries. A former
    # full-catalog test object may move to dev, but no held-out family enters train.
    anchors = {}
    for split in ("dev", "test"):
        path = source.parent.parent.parent / f"eval/visual_{split}/gallery_ids.json"
        ids = json.loads(path.read_text(encoding="utf-8"))
        for item_id in ids:
            group = mapping[item_id]
            if group in anchors and anchors[group] != split:
                raise ValueError("Blind dev/test share an object family; rebuild galleries first")
            anchors[group] = split
    group_splits.update(anchors)
    changes = []
    for row in rows:
        old = row["split"]
        group = mapping[row["item_id"]]
        row.update(previous_split=old, object_group=group, split=group_splits[group],
                   split_policy="official-relations-components-v2-heldout-priority")
        if row["split"] != old:
            changes.append({"item_id": row["item_id"], "from": old, "to": row["split"], "group": group})
    validate_object_splits(rows)
    if output.exists():
        raise FileExistsError(f"Immutable output exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    result = {"source": str(source), "source_sha256": digest, "output": str(output),
              "data_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
              "relations_sha256": hashlib.sha256(relations.read_bytes()).hexdigest(),
              "items": len(rows), "groups": len(grouped), "splits": dict(Counter(row["split"] for row in rows)),
              "policy": "Whole connected group: preserve disjoint blind dev/test anchor galleries; other groups use original test > dev > train priority. Never move a held-out family into train. All formal local training restarts from pretrained weights; v1 results invalidated.",
              "changed_items": changes}
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="data/snapshots/collection_v1.jsonl")
    parser.add_argument("--relations", default="data/processed/object_relations_v1.json")
    parser.add_argument("--output", default="data/snapshots/collection_v2.jsonl")
    parser.add_argument("--report", default="data/processed/grouped_freeze_v2.json")
    args = parser.parse_args()
    result = freeze(args.source, args.relations, args.output, args.report)
    print(json.dumps({key: value for key, value in result.items() if key != "changed_items"}, indent=2))


if __name__ == "__main__":
    main()
