"""Report catalogue support for the auditable query ontology.

The report separates concepts observed in catalogue text from concepts backed
by a trained visual probe. It is intended to guide vocabulary expansion and
does not turn catalogue mentions into visual ground truth.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re

from curio.query.ontology import VOCAB
from curio.query.parser import term_pattern
from curio.retrieve.engine import load_items


TEXT_FIELDS = ("title", "object_name", "classification", "medium", "culture", "period", "description", "tags")


def item_text(item: dict) -> str:
    values = []
    for key in TEXT_FIELDS:
        value = item.get(key)
        values.extend(str(entry) for entry in value) if isinstance(value, list) else values.append(str(value or ""))
    return " ".join(values).lower()


def coverage(items: list[dict], probe_report: dict | None = None) -> dict:
    result = {}
    texts = [item_text(item) for item in items]
    trained = {
        (slot, value)
        for slot, task in (probe_report or {}).get("tasks", {}).items()
        if task.get("status") == "trained"
        for value in task.get("classes", [])
    }
    for slot, values in VOCAB.items():
        rows = []
        for value, aliases in values.items():
            count = sum(any(re.search(term_pattern(alias), text, re.I) for alias in aliases) for text in texts)
            rows.append({"value": value, "catalogue_support": count, "probe_support": (slot, value) in trained})
        result[slot] = sorted(rows, key=lambda row: (-row["catalogue_support"], row["value"]))
    object_names = Counter(str(item.get("object_name") or "").strip() for item in items if str(item.get("object_name") or "").strip())
    mediums = Counter()
    for item in items:
        value = item.get("medium")
        entries = value if isinstance(value, list) else [value]
        mediums.update(str(entry).strip() for entry in entries if str(entry or "").strip())
    return {
        "items": len(items),
        "ontology_counts": {slot: len(values) for slot, values in VOCAB.items()},
        "concepts": result,
        "top_object_names": object_names.most_common(50),
        "top_medium_terms": mediums.most_common(50),
        "warning": "Catalogue support is weak text supervision; it does not prove image visibility. Probe support is false for newly added classes until independently trained.",
    }


def markdown(report: dict) -> str:
    lines = ["# Ontology coverage", "", f"Catalogue items: **{report['items']}**", "",
             "Counts are catalogue text matches, not visual labels. Newly added classes remain unknown to the current trained probes.", ""]
    for slot, rows in report["concepts"].items():
        lines.extend([f"## {slot}", "", "| Concept | Catalogue text support | Trained visual probe |", "|---|---:|---|"])
        lines.extend(f"| `{row['value']}` | {row['catalogue_support']} | {'yes' if row['probe_support'] else 'no'} |" for row in rows)
        lines.append("")
    lines.extend(["## Frequent unmapped catalogue vocabulary", "", "The following raw terms are retained as expansion candidates; they are not automatically canonicalized:", ""])
    lines.extend(f"- `{term}` ({count})" for term, count in report["top_object_names"][:25])
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/snapshots/collection_v2.jsonl")
    parser.add_argument("--probes", default="artifacts/probes/siglip_adapted")
    parser.add_argument("--json-output", default="docs/ontology_coverage.json")
    parser.add_argument("--markdown-output", default="docs/ONTOLOGY_COVERAGE.md")
    args = parser.parse_args()
    probe_report_path = Path(args.probes) / "report.json"
    probe_report = json.loads(probe_report_path.read_text(encoding="utf-8")) if probe_report_path.exists() else None
    report = coverage(load_items(args.data), probe_report)
    Path(args.json_output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.markdown_output).write_text(markdown(report), encoding="utf-8")
    print(json.dumps({"items": report["items"], "ontology_counts": report["ontology_counts"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
