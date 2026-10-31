"""Read-only object/part relationship audit, including blind-set overlap."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def accession_root(accession):
    # Strip lettered components/ranges, preserving numeric accession subitems.
    # Also handles Met's cached mojibake range separator (e.g. 4a�Ct).
    return re.sub(r"(?<=\d)\.?[a-z][^0-9]*$", "", str(accession).strip(), flags=re.I)


def build_relation_report(rows, root=ROOT, raw_metadata=None):
    root = Path(root)
    by_id = {row["item_id"]: row for row in rows}
    parent, evidence = {item_id: item_id for item_id in by_id}, []

    def find(item_id):
        while parent[item_id] != item_id:
            parent[item_id] = parent[parent[item_id]]
            item_id = parent[item_id]
        return item_id

    def link(members, basis, **details):
        members = sorted(set(members))
        if len(members) < 2:
            return
        for item_id in members[1:]:
            parent[find(item_id)] = find(members[0])
        evidence.append({"basis": basis, "item_ids": members, **details})

    raw, raw_errors = {}, []
    for row in rows:
        item_id = row["item_id"]
        path = root / "data/raw" / row["source"] / f"{row['source_id']}.json"
        try:
            metadata = raw_metadata[item_id] if raw_metadata is not None else json.loads(path.read_bytes())
            if not isinstance(metadata, dict):
                raise ValueError("raw JSON is not an object")
            raw[item_id] = metadata
        except Exception as exc:
            raw_errors.append({"item_id": item_id, "path": str(path), "error": str(exc)})
    duplicate_groups = {}
    for field in ("image_path", "sha256", "image_sha256", "object_group", "duplicate_group", "phash_group", "phash"):
        groups = defaultdict(list)
        for row in rows:
            if row.get(field):
                groups[str(row[field])].append(row["item_id"])
        duplicate_groups[field] = sum(len(members) > 1 for members in groups.values())
        for value, members in sorted(groups.items()):
            link(members, f"catalog_{field}", value=value)
    dangling = []
    for row in rows:
        for field in ("duplicate_sha256_of", "near_duplicate_of"):
            target = row.get(field)
            if target in by_id:
                link([row["item_id"], target], field)
            elif target:
                dangling.append({"item_id": row["item_id"], "field": field, "target": target})
    accessions, roots, accession_values = defaultdict(list), defaultdict(list), {}
    for item_id, metadata in raw.items():
        source = by_id[item_id]["source"]
        accession = str(metadata.get("accessionNumber" if source == "met" else "accession_number") or "").strip()
        accession_values[item_id] = accession
        if accession:
            accessions[(source, accession)].append(item_id)
            roots[(source, accession_root(accession))].append(item_id)
    for (source, accession), members in sorted(accessions.items()):
        link(members, "same_official_accession", source=source, accession=accession)
    for (source, accession), members in sorted(roots.items()):
        link(members, "official_accession_lettered_parts", source=source, accession_root=accession)
    for item_id, metadata in raw.items():
        if by_id[item_id]["source"] != "cleveland":
            continue
        accession = accession_values[item_id]
        parts = accession.split(".")
        # A parent record must actually exist; do not collapse unrelated
        # acquisitions sharing only a museum/year accession prefix.
        for end in range(2, len(parts)):
            ancestor = ".".join(parts[:end])
            other = accessions.get(("cleveland", ancestor), [])
            if other:
                link([item_id, *other], "cma_accession_parent_record", accession_root=ancestor)
    related, seen_related = [], set()
    for item_id, metadata in raw.items():
        if by_id[item_id]["source"] != "cleveland":
            continue
        for entry in metadata.get("related_works") or []:
            other = f"cma_{entry.get('id')}"
            pair = tuple(sorted((item_id, other)))
            if other in by_id and pair not in seen_related:
                seen_related.add(pair)
                details = {"relationship": entry.get("relationship"), "description": entry.get("description"),
                           "accessions": [accession_values.get(member, "") for member in pair]}
                related.append({"item_ids": list(pair), **details})
                # Conservatively keep all official related works together,
                # including a complete set and separately photographed parts.
                link(pair, "cma_official_related_works_conservative", **details)
    components = defaultdict(list)
    for item_id in by_id:
        components[find(item_id)].append(item_id)
    groups, item_to_group = [], {}
    for members in sorted((sorted(members) for members in components.values())):
        group_id = members[0]
        for item_id in members:
            item_to_group[item_id] = group_id
        member_set = set(members)
        groups.append({"group_id": group_id, "item_ids": members,
                       "splits": sorted({by_id[item_id]["split"] for item_id in members}),
                       "members": [{"item_id": item_id, "source": by_id[item_id]["source"],
                                    "split": by_id[item_id]["split"], "accession": accession_values.get(item_id),
                                    "accession_root": accession_root(accession_values.get(item_id, "")),
                                    "title": raw.get(item_id, {}).get("title")} for item_id in members],
                       "evidence": [entry for entry in evidence if set(entry["item_ids"]) <= member_set]})
    blind, blind_sources = {}, {}
    for split in ("train", "dev", "test"):
        path = root / f"eval/visual_{split}/blind_map.json"
        if path.exists():
            entries = json.loads(path.read_bytes())
            if isinstance(entries, dict):
                entries = list(entries.values())
            blind[split] = entries
            blind_sources[split] = {"path": str(path.relative_to(root)).replace("\\", "/"), "sha256": digest(path), "count": len(entries),
                                    "missing_item_ids": [entry["item_id"] for entry in entries if entry["item_id"] not in by_id]}
    blind_overlaps, active_blind_overlaps, inactive_blind = [], [], []
    active_blind = {}
    for split, entries in blind.items():
        active_blind[split] = []
        for entry in entries:
            if entry["item_id"] in by_id and by_id[entry["item_id"]]["split"] == split:
                active_blind[split].append(entry)
            else:
                inactive_blind.append({"blind_split": split, **entry, "actual_split": by_id.get(entry["item_id"], {}).get("split")})
    for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
        left_groups = defaultdict(list)
        right_groups = defaultdict(list)
        for split, target in ((left, left_groups), (right, right_groups)):
            for entry in blind.get(split, []):
                if entry["item_id"] in item_to_group:
                    target[item_to_group[entry["item_id"]]].append({"alias": entry.get("alias"), "item_id": entry["item_id"]})
        shared = sorted(left_groups.keys() & right_groups.keys())
        blind_overlaps.append({"left": left, "right": right, "overlap_groups": len(shared),
                               "groups": [{"group_id": group_id, "left": left_groups[group_id], "right": right_groups[group_id]} for group_id in shared]})
        active_shared = []
        for group_id in shared:
            active_left = [entry for entry in left_groups[group_id] if by_id[entry["item_id"]]["split"] == left]
            active_right = [entry for entry in right_groups[group_id] if by_id[entry["item_id"]]["split"] == right]
            if active_left and active_right:
                active_shared.append({"group_id": group_id, "left": active_left, "right": active_right})
        active_blind_overlaps.append({"left": left, "right": right, "overlap_groups": len(active_shared), "groups": active_shared})
    blind_vs_full = []
    group_by_id = {group["group_id"]: group for group in groups}
    for split, entries in blind.items():
        for entry in entries:
            if entry["item_id"] not in item_to_group:
                continue
            group = group_by_id[item_to_group[entry["item_id"]]]
            external = [member for member in group["members"] if member["split"] != split]
            if external:
                blind_vs_full.append({"blind_split": split, "alias": entry.get("alias"), "item_id": entry["item_id"],
                                      "group_id": group["group_id"], "other_split_members": external})
    crossing = [group for group in groups if len(group["splits"]) > 1]
    return {"protocol": "curio-object-relations-v2", "rows": len(rows), "group_count": len(groups),
            "non_singleton_groups": sum(len(group["item_ids"]) > 1 for group in groups),
            "cross_split_group_count": len(crossing), "cross_split_item_count": sum(len(group["item_ids"]) for group in crossing),
            "cross_split_group_ids": [group["group_id"] for group in crossing], "groups": groups,
            "item_to_group": item_to_group, "catalog_duplicate_group_counts": duplicate_groups,
            "evidence_counts": dict(Counter(entry["basis"] for entry in evidence)), "official_related_works": related,
            "raw_metadata_errors": raw_errors, "dangling_duplicate_references": dangling,
            "blind_sources": blind_sources, "blind_pair_overlaps": blind_overlaps,
            "active_blind_counts": {split: len(entries) for split, entries in active_blind.items()},
            "active_blind_pair_overlaps": active_blind_overlaps, "inactive_blind_entries": inactive_blind,
            "blind_vs_full_other_splits": blind_vs_full,
            "interpretation": "Related works are conservatively grouped for split isolation; an official related-work link alone does not prove a single physical object."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/snapshots/collection_v1.jsonl")
    parser.add_argument("--output", default="data/processed/object_relations_v1.json")
    args = parser.parse_args()
    path = ROOT / args.data
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    report = build_relation_report(rows)
    report.update(data=args.data, data_sha256=digest(path))
    destination = ROOT / args.output
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("rows", "group_count", "non_singleton_groups", "cross_split_group_count", "cross_split_item_count", "evidence_counts", "blind_sources", "blind_pair_overlaps")}, ensure_ascii=False, indent=2))
    return int(bool(report["cross_split_group_count"] or report["raw_metadata_errors"] or report["dangling_duplicate_references"]))


if __name__ == "__main__":
    raise SystemExit(main())
