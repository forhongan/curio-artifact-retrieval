"""Join blind image judgments to IDs, keeping authorship and complete qrels auditable."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def rows(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def merge(data, output):
    from curio.represent.detail_probes import load_annotations

    records = []
    catalog = {row["item_id"]: row for row in rows(data)}
    excluded = []
    for split in ("train", "dev", "test"):
        directory = ROOT / f"eval/visual_{split}"
        mapping = {
            row["alias"]: row
            for row in json.loads((directory / "blind_map.json").read_text(encoding="utf-8"))
        }
        seen = set()
        for path in sorted(directory.glob("labels_*.jsonl")):
            for row in rows(path):
                alias = row["alias"]
                if alias not in mapping or alias in seen:
                    raise ValueError(f"Unknown/duplicate blind alias: {split}/{alias}")
                if not row.get("annotator"):
                    raise ValueError(f"Missing disclosed annotator: {path}/{alias}")
                seen.add(alias)
                item_id = mapping[alias]["item_id"]
                actual_split = catalog[item_id]["split"]
                if actual_split != split:
                    if split != "train":
                        raise ValueError(f"Held-out blind gallery moved: {alias}/{actual_split}")
                    excluded.append({"alias": alias, "item_id": item_id, "pool": split,
                                     "actual_split": actual_split, "reason": "object_family_moved_to_holdout"})
                    continue
                records.append(
                    {
                        **row,
                        "item_id": mapping[alias]["item_id"],
                        "split": split,
                        "image_sha256": mapping[alias]["processed_sha256"],
                        "label_source": "ai_image_only",
                    }
                )
        if seen != mapping.keys():
            raise ValueError(f"Incomplete {split} image labels: {len(seen)}/{len(mapping)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8"
    )
    load_annotations(output, rows(data))
    output.with_name("excluded_training_labels.json").write_text(
        json.dumps(excluded, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return len(records)


def convert_queries(directory):
    directory = Path(directory)
    mapping = {
        row["alias"]: row
        for row in json.loads((directory / "blind_map.json").read_text(encoding="utf-8"))
    }
    queries, qrels = [], []
    seen_qids = set()
    for record in rows(directory / "query_judgments.jsonl"):
        qid = record["qid"]
        if qid in seen_qids:
            raise ValueError(f"Duplicate qid: {qid}")
        seen_qids.add(qid)
        grades, rationale = {}, record.get("rationale")
        if not rationale or not record.get("annotator"):
            raise ValueError(f"Missing visual rationale/annotator for {qid}")
        groups = record["grade_groups"]
        if set(groups) != {"0", "1", "2", "3"}:
            raise ValueError("Each query must explicitly list all four grade groups")
        for grade, aliases in groups.items():
            if isinstance(aliases, str):
                aliases = aliases.replace(",", " ").split()
            for alias in aliases:
                if alias not in mapping or mapping[alias]["item_id"] in grades:
                    raise ValueError(f"Unknown/repeated query judgment: {qid}/{alias}")
                grades[mapping[alias]["item_id"]] = int(grade)
        if len(grades) != len(mapping) or not any(grade >= 2 for grade in grades.values()):
            raise ValueError(f"Incomplete or no-positive visual query: {qid}")
        queries.append(
            {
                "qid": qid,
                "text_en": record["text_en"],
                "text_zh": record["text_zh"],
                "type": record.get("type", "visual_detail"),
                "label_source": "ai_visual_audit",
                "rationale": rationale,
                "annotator": record["annotator"],
            }
        )
        qrels.append(
            {
                "qid": qid,
                "grades": grades,
                "label_source": "ai_visual_audit",
                "annotator": record["annotator"],
                "rationale": rationale,
            }
        )
    if not queries:
        raise ValueError("No visual queries")
    for name, records in (("queries", queries), ("qrels", qrels)):
        (directory / f"{name}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8"
        )
    return len(queries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/snapshots/collection_v2.jsonl")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "eval/visual_labels/annotations.jsonl"
    )
    parser.add_argument(
        "--queries", type=Path, help="Convert an existing query_judgments.jsonl instead of labels"
    )
    args = parser.parse_args()
    print(convert_queries(args.queries) if args.queries else merge(args.data, args.output))


if __name__ == "__main__":
    main()
