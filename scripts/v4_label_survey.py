"""Inventory AI visual labels and rank unregistered query concepts.

This script is intentionally read-only with respect to ``eval/``. It reads raw
annotation and frozen evaluation files, joins blind aliases to the official
collection snapshot, and writes a separate v4 survey bundle.

Source terms are lexical evidence, not semantic ground truth. Catalogue support
and AI caption support are always reported separately.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from curio.query.ontology import SPECIALIST_GLOSSARY, VOCAB


ROOT = Path(__file__).resolve().parents[1]
CATALOGUE = ROOT / "data/snapshots/collection_v2.jsonl"
DEFAULT_OUT = ROOT / "artifacts/cv_v4_20261108/label_survey"

SURVEY_FILES = (
    ("visual_train_terra_a", "eval/visual_train/labels_terra_a.jsonl", "train", "visual_train"),
    ("visual_train_terra_b", "eval/visual_train/labels_terra_b.jsonl", "train", "visual_train"),
    (
        "visual_train_root_train",
        "eval/visual_train/labels_root_train.jsonl",
        "train",
        "visual_train",
    ),
    (
        "visual_train_root_supplement",
        "eval/visual_train/labels_root_supplement.jsonl",
        "train",
        "visual_train",
    ),
    (
        "visual_train_root_final",
        "eval/visual_train/labels_root_final.jsonl",
        "train",
        "visual_train",
    ),
    ("visual_dev_sol", "eval/visual_dev/labels_sol.jsonl", "dev", "visual_dev"),
    ("visual_test_root", "eval/visual_test/labels_root.jsonl", "test", "visual_test"),
    (
        "cv_research_a",
        "eval/cv_research_20261104/annotator_a/labels.jsonl",
        "cv_research",
        "cv_research_20261104",
    ),
    (
        "cv_research_b",
        "eval/cv_research_20261104/annotator_b/labels.jsonl",
        "cv_research",
        "cv_research_20261104",
    ),
    (
        "cv_research_qa",
        "eval/cv_research_20261104/annotator_qa/labels.jsonl",
        "cv_research",
        "cv_research_20261104",
    ),
)

CONCEPTS = {
    "shape_form": {
        "title": "shape / form",
        "description": "silhouette, body, neck, rim, foot, mouth, or geometric profile",
        "terms": [
            "shape",
            "shaped",
            "form",
            "formed",
            "silhouette",
            "profile",
            "body",
            "neck",
            "mouth",
            "rim",
            "foot",
            "base",
            "legs",
            "lobed",
            "polygonal",
            "octagonal",
            "rectangular",
            "square",
            "cylindrical",
            "conical",
            "ovoid",
            "spherical",
            "bulbous",
            "pear",
            "tray",
            "bottle-shaped",
            "bell-shaped",
            "flower-shaped",
            "animal-shaped",
            "ring-shaped",
        ],
    },
    "colors_appearance": {
        "title": "colors / appearance",
        "description": "visible color or appearance words not represented by official colors",
        "terms": [
            "pale",
            "dark",
            "golden",
            "multicolored",
            "multicolour",
            "orange",
            "gray",
            "grey",
            "cream",
            "purple",
            "mottled",
            "patterned",
            "speckled",
            "blue and white",
            "blue-and-white",
        ],
    },
    "material_appearance": {
        "title": "material / appearance",
        "description": "material-like visual naming, gloss, glaze, or translucency",
        "terms": [
            "porcelain",
            "ceramic",
            "stoneware",
            "earthenware",
            "pottery",
            "bronze",
            "copper",
            "metal",
            "gold",
            "silver",
            "jade",
            "silk",
            "textile",
            "wood",
            "wooden",
            "glass",
            "lacquer",
            "stone",
            "ivory",
            "glaze",
            "glazed",
            "glossy",
            "matte",
            "translucent",
            "opaque",
        ],
    },
    "part_completeness": {
        "title": "part / completeness",
        "description": "object part, attachment, detached state, or fragment",
        "terms": [
            "detached",
            "fragment",
            "partial",
            "complete",
            "full-length",
            "head",
            "face",
            "neck",
            "rim",
            "foot",
            "base",
            "cover",
            "lid",
            "knob",
            "finial",
            "spout",
            "handle",
            "shoulder",
            "panel",
            "border",
            "edge",
        ],
    },
    "multiplicity": {
        "title": "multiplicity",
        "description": "single or multiple objects, repetitions, pairs, or stacks",
        "terms": [
            "single",
            "one",
            "two",
            "three",
            "four",
            "pair",
            "paired",
            "multiple",
            "several",
            "group",
            "repeated",
            "repeating",
            "rows",
            "stacked",
        ],
    },
    "orientation": {
        "title": "orientation",
        "description": "figure pose or image/object orientation",
        "terms": [
            "standing",
            "seated",
            "reclining",
            "horizontal",
            "vertical",
            "upright",
            "leaning",
            "front view",
            "side view",
            "top view",
            "profile",
        ],
    },
    "surface_technique": {
        "title": "surface / technique",
        "description": "working method or visible surface treatment",
        "terms": [
            "carved",
            "incised",
            "engraved",
            "painted",
            "brushed",
            "crackle",
            "crackled",
            "raised",
            "relief",
            "openwork",
            "pierced",
            "embossed",
            "glaze",
            "glazed",
            "glossy",
            "matte",
            "molded",
            "moulded",
            "woven",
            "embroidered",
            "embroidery",
            "applique",
            "rouletted",
            "modelled",
            "modeled",
        ],
    },
    "motifs": {
        "title": "motifs",
        "description": "visible decoration not covered by the seven official motif fields",
        "terms": [
            "animal",
            "creature",
            "scene",
            "medallion",
            "medallions",
            "leaf",
            "leaves",
            "leafy",
            "peony",
            "chrysanthemum",
            "plum",
            "bamboo",
            "horse",
            "lion",
            "butterfly",
            "building",
            "tree",
            "waves",
            "scroll",
            "scrolling",
            "arabesque",
            "inscription",
            "character",
            "sun",
            "moon",
        ],
    },
    "vessel_subclass": {
        "title": "vessel subclass",
        "description": "specific visible form names not registered in official object_type",
        "terms": [
            "bottle",
            "flask",
            "ewer",
            "meiping",
            "stem bowl",
            "tea bowl",
            "water coupe",
            "censer",
            "incense burner",
            "double gourd",
            "double-gourd",
            "pear-shaped",
            "trumpet mouth",
            "tray",
            "dish",
            "covered",
            "lidded",
            "spout",
            "buckle",
            "stopper",
        ],
    },
}

REGISTERED = {
    "official_object_type": sorted(VOCAB["object_type"]),
    "official_material": sorted(VOCAB["material"]),
    "official_motifs": sorted(VOCAB["motifs"]),
    "official_colors": sorted(VOCAB["colors"]),
    "official_shape": sorted(VOCAB["shape"]),
    "specialist_glossary": sorted(SPECIALIST_GLOSSARY),
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def dump_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )


def flatten_values(value) -> list[str]:
    if isinstance(value, list):
        return [str(x) for x in value if x not in (None, "")]
    text = str(value or "").strip().lower()
    if not text:
        return []
    return [x for x in re.split(r"\s*[;,/]\s*", text) if x]


def official_slots(item: dict) -> dict[str, list[str]]:
    return {
        "object_type": flatten_values(item.get("object_type")),
        "material": flatten_values(item.get("material")),
        "motifs": flatten_values(item.get("motifs")),
        "colors": flatten_values(item.get("colors")),
    }


def official_text(item: dict) -> str:
    fields = [
        item.get("title", ""),
        item.get("object_name", ""),
        item.get("classification", ""),
        item.get("medium", ""),
        item.get("description", ""),
        " ".join(item.get("tags") or []),
    ]
    return " ".join(str(x) for x in fields).lower()


def aliases_to_ids(directory: str, rows: list[dict]) -> dict[str, dict]:
    mapping = read_json(ROOT / "eval" / directory / "blind_map.json")
    by_alias = {r["alias"]: r for r in mapping}
    unknown = sorted({r["alias"] for r in rows if r["alias"] not in by_alias})
    if unknown:
        raise ValueError(f"Unmapped blind aliases in {directory}: {unknown[:5]}")
    return by_alias


def survey_sources(catalogue: dict[str, dict]) -> tuple[list[dict], list[dict]]:
    source_manifest, joined = [], []
    for name, relpath, declared_split, mapping_dir in SURVEY_FILES:
        path = ROOT / relpath
        rows = read_jsonl(path)
        mapping = aliases_to_ids(mapping_dir, rows)
        aliases = [r["alias"] for r in rows]
        if len(aliases) != len(set(aliases)):
            raise ValueError(f"Duplicate aliases in {relpath}")
        mapped_ids = [mapping[a]["item_id"] for a in aliases]
        if len(mapped_ids) != len(set(mapped_ids)):
            raise ValueError(f"Repeated mapped items in {relpath}")
        if not set(mapped_ids) <= catalogue.keys():
            raise ValueError(f"Annotation items absent from collection_v2: {relpath}")
        official = [catalogue[i] for i in mapped_ids]
        source_manifest.append(
            {
                "source": name,
                "path": relpath,
                "rows": len(rows),
                "declared_split": declared_split,
                "actual_split_counts": dict(Counter(catalogue[i]["split"] for i in mapped_ids)),
                "annotators": dict(Counter(str(r.get("annotator")) for r in rows)),
                "label_source": "ai_derived",
                "sha256": sha(path),
                "has_caption": any(r.get("caption_en") for r in rows),
                "has_motifs": any("motifs" in r for r in rows),
                "official_object_type_known": sum(bool(x.get("object_type")) for x in official),
                "official_material_known": sum(bool(x.get("material")) for x in official),
                "official_medium_known": sum(bool(x.get("medium")) for x in official),
                "official_motifs_known": sum(bool(x.get("motifs")) for x in official),
                "official_colors_known": sum(bool(x.get("colors")) for x in official),
                "note": "Read-only raw AI annotation source",
            }
        )
        for row, item_id in zip(rows, mapped_ids):
            item = catalogue[item_id]
            joined.append(
                {
                    "source": name,
                    "source_path": relpath,
                    "alias": row["alias"],
                    "item_id": item_id,
                    "mapping_directory": mapping_dir,
                    "declared_split": declared_split,
                    "actual_split": item["split"],
                    "official_source": item.get("source"),
                    "official_object_group": item.get("object_group"),
                    "annotator": row.get("annotator"),
                    "label_source": "ai_derived",
                    "caption_en": row.get("caption_en", ""),
                    "caption_zh": row.get("caption_zh", ""),
                    "uncertainty": row.get("uncertainty", ""),
                    "confidence": row.get("confidence"),
                    "object_type": row.get("object_type"),
                    "colors": row.get("colors"),
                    "shape": row.get("shape"),
                    "motifs": row.get("motifs"),
                    "official_object_type": item.get("object_type"),
                    "official_object_name": item.get("object_name"),
                    "official_classification": item.get("classification"),
                    "official_material": item.get("material"),
                    "official_medium": item.get("medium"),
                    "official_motifs": item.get("motifs"),
                    "official_colors": item.get("colors"),
                    "official_title": item.get("title"),
                    "official_tags": item.get("tags"),
                    "official_description": item.get("description"),
                    "image_sha256": mapping[row["alias"]].get(
                        "processed_sha256", mapping[row["alias"]].get("image_sha256")
                    ),
                }
            )
    return source_manifest, joined


def inventory_all_jsonl(catalogue: dict[str, dict]) -> list[dict]:
    """Inventory every JSONL under eval without interpreting derived files as new labels."""
    inventory = []
    for path in sorted((ROOT / "eval").rglob("*.jsonl")):
        rows = read_jsonl(path)
        keys = sorted({k for row in rows for k in row})
        label_sources = sorted(
            {str(row.get("label_source")) for row in rows if "label_source" in row}
        )
        annotators = sorted({str(row.get("annotator")) for row in rows if "annotator" in row})
        item_ids = {row["item_id"] for row in rows if "item_id" in row}
        graded_items = {item_id for row in rows for item_id in (row.get("grades") or {})}
        alias_items = set()
        if any("alias" in row for row in rows):
            mapping_dir = path.parent
            while mapping_dir != ROOT / "eval" and not (mapping_dir / "blind_map.json").exists():
                mapping_dir = mapping_dir.parent
            mapping_path = mapping_dir / "blind_map.json"
            if mapping_path.exists():
                mapping = {row["alias"]: row["item_id"] for row in read_json(mapping_path)}
                alias_items = {mapping[row["alias"]] for row in rows if row.get("alias") in mapping}
        referenced = item_ids | graded_items | alias_items
        inventory.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "rows": len(rows),
                "bytes": path.stat().st_size,
                "sha256": sha(path),
                "keys": keys,
                "declared_label_sources": label_sources,
                "annotators": annotators,
                "direct_item_ids": len(item_ids),
                "graded_item_ids": len(graded_items),
                "mapped_alias_item_ids": len(alias_items),
                "official_items_resolvable": len(referenced & catalogue.keys()),
                "official_items_missing": len(referenced - catalogue.keys()),
            }
        )
    return inventory


def source_field_counts(source_manifest: list[dict], joined: list[dict]) -> list[dict]:
    """Expose exact source-row known and category counts before item deduplication."""
    result = []
    for source in source_manifest:
        rows = [r for r in joined if r["source"] == source["source"]]
        result.append(
            {
                "source": source["source"],
                "rows": len(rows),
                "caption_en_known": sum(bool(r.get("caption_en")) for r in rows),
                "caption_zh_known": sum(bool(r.get("caption_zh")) for r in rows),
                "object_type_known": sum(r.get("object_type") is not None for r in rows),
                "object_type_counts": dict(Counter(str(r.get("object_type")) for r in rows)),
                "uncertainty_nonempty": sum(bool(r.get("uncertainty")) for r in rows),
                "confidence_counts": dict(Counter(str(r.get("confidence")) for r in rows)),
            }
        )
    return result


def field_value_counts(source_manifest: list[dict], joined: list[dict]) -> list[dict]:
    result = []
    for source in source_manifest:
        rows = [r for r in joined if r["source"] == source["source"]]
        for task in ("colors", "shape", "motifs"):
            keys = sorted({key for r in rows for key in (r.get(task) or {})})
            for key in keys:
                values = [(r.get(task) or {}).get(key) for r in rows]
                result.append(
                    {
                        "source": source["source"],
                        "task": task,
                        "field": key,
                        "rows": len(rows),
                        "known": sum(value is not None for value in values),
                        "one": values.count(1),
                        "zero": values.count(0),
                        "null": values.count(None),
                    }
                )
    return result


def query_judgment_counts() -> list[dict]:
    result = []
    for split in ("dev", "test"):
        for row in read_jsonl(ROOT / f"eval/visual_{split}/query_judgments.jsonl"):
            groups = row["grade_groups"]
            counts = {grade: len(groups[grade]) for grade in ("0", "1", "2", "3")}
            result.append(
                {
                    "split": split,
                    "qid": row["qid"],
                    "type": row["type"],
                    "text_en": row["text_en"],
                    "text_zh": row["text_zh"],
                    "annotator": row.get("annotator"),
                    "label_source": "ai_visual_audit",
                    **{f"grade_{grade}": count for grade, count in counts.items()},
                    "images": sum(counts.values()),
                    "positive_grade_2_or_3": counts["2"] + counts["3"],
                }
            )
    return result


def metadata_qrel_counts(catalogue: dict[str, dict]) -> list[dict]:
    result = []
    for split in ("dev", "test"):
        for row in read_jsonl(ROOT / f"eval/qrels_{split}.jsonl"):
            grades = Counter(row["grades"].values())
            result.append(
                {
                    "split": split,
                    "qid": row["qid"],
                    "label_source": row.get("label_source"),
                    "graded_items": len(row["grades"]),
                    "official_items": sum(item_id in catalogue for item_id in row["grades"]),
                    "grade_0": grades[0],
                    "grade_1": grades[1],
                    "grade_2": grades[2],
                    "grade_3": grades[3],
                }
            )
    return result


def official_pool_support(catalogue: dict[str, dict], joined: list[dict]) -> dict:
    result = {}
    for split in ("train", "dev", "test"):
        pool_ids = {
            mapping["item_id"]
            for mapping in read_json(ROOT / f"eval/visual_{split}/blind_map.json")
        }
        rows = [catalogue[item_id] for item_id in pool_ids]
        result[split] = {
            "pool_items": len(pool_ids),
            **{
                field: sum(bool(row.get(field)) for row in rows)
                for field in (
                    "object_type",
                    "object_name",
                    "classification",
                    "material",
                    "medium",
                    "motifs",
                    "colors",
                )
            },
        }
    result["all_surveyed_items"] = {
        "unique_items": len({row["item_id"] for row in joined}),
        **{
            field: sum(
                bool(catalogue[item_id].get(field))
                for item_id in {row["item_id"] for row in joined}
            )
            for field in (
                "object_type",
                "object_name",
                "classification",
                "material",
                "medium",
                "motifs",
                "colors",
            )
        },
    }
    return result


def raw_schema_registry(source_manifest: list[dict], joined: list[dict]) -> dict:
    sources = {s["source"] for s in source_manifest}
    object_types = sorted(
        {
            str(r["object_type"])
            for r in joined
            if r["source"] in sources and r.get("object_type") is not None
        }
    )
    values = {}
    for task in ("colors", "shape", "motifs"):
        values[task] = sorted(
            {key for r in joined if r["source"] in sources for key in (r.get(task) or {})}
        )
    return {
        "labels_fields": [
            "object_type",
            "colors",
            "shape",
            "motifs",
            "caption_en",
            "caption_zh",
            "confidence",
            "uncertainty",
        ],
        "object_type_values_observed": object_types,
        "structured_task_values": values,
        "absent_structured_tasks": [
            "material",
            "part_completeness",
            "multiplicity",
            "orientation",
            "surface_technique",
            "vessel_subclass",
        ],
        "note": "Observed raw schema, not a template for required vocabulary.",
    }


def registered_support(catalogue: list[dict]) -> dict:
    result = {}
    for slot, values in VOCAB.items():
        result[slot] = []
        for value in values:
            if slot == "motifs":
                count = sum(value in (r.get("motifs") or []) for r in catalogue)
            elif slot in {"colors", "shape"}:
                count = 0
            else:
                count = sum(str(r.get(slot) or "").lower() == value for r in catalogue)
            result[slot].append(
                {
                    "concept": value,
                    "official_rows": count,
                    "official_prevalence": count / len(catalogue),
                }
            )
        result[slot].sort(key=lambda r: (-r["official_rows"], r["concept"]))
    return {
        "note": "Exact equality for object_type/material, membership for motifs, exact official colors/shape fields.",
        "items": len(catalogue),
        "slots": result,
    }


def annotation_counts(catalogue: dict[str, dict]) -> tuple[dict, list[dict]]:
    # Unique official items, not source rows, are the correct denominator when
    # combining duplicate raw source files.
    items: dict[str, dict] = {}
    for name, relpath, _, mapping_dir in SURVEY_FILES:
        rows = read_jsonl(ROOT / relpath)
        mapping = aliases_to_ids(mapping_dir, rows)
        for row in rows:
            item_id = mapping[row["alias"]]["item_id"]
            items.setdefault(item_id, []).append((name, row))

    task_counts = {}
    for task in ("object_type", "colors", "shape", "motifs"):
        task_counts[task] = {}
        if task == "object_type":
            for value in sorted({r[1].get("object_type") for r in items.values() for r in r}):
                task_counts[task][str(value)] = {"known": 0, "positive": 0}
            for entries in items.values():
                vals = [e[1].get("object_type") for e in entries]
                if any(v is not None for v in vals):
                    # Separate raw-source repeats are retained in source_rows.jsonl.
                    # This unique view uses the last source's explicit value.
                    value = next((v for v in reversed(vals) if v is not None), None)
                    task_counts[task][str(value)]["known"] += 1
                    task_counts[task][str(value)]["positive"] += int(value is not None)
            continue
        keys = sorted(
            {k for entries in items.values() for _, row in entries for k in (row.get(task) or {})}
        )
        for key in keys:
            task_counts[task][key] = {"known": 0, "positive": 0, "unknown": 0}
        for entries in items.values():
            for key in keys:
                vals = [(row.get(task) or {}).get(key) for _, row in entries]
                known = [v for v in vals if v is not None]
                if known:
                    task_counts[task][key]["known"] += 1
                    task_counts[task][key]["positive"] += int(any(v == 1 for v in known))
                else:
                    task_counts[task][key]["unknown"] += 1
    return {
        "unique_official_items": len(items),
        "source_rows": sum(len(v) for v in items.values()),
        "tasks": task_counts,
        "note": "Unique-item rollup: repeated source rows are not independent items; last non-null source value is used for object_type.",
    }, [
        {"item_id": item_id, "source_row_count": len(entries), "sources": [e[0] for e in entries]}
        for item_id, entries in sorted(items.items())
    ]


def term_rows(value: str) -> tuple[str, re.Pattern[str]]:
    escaped = re.escape(value.lower())
    if re.fullmatch(r"[a-z0-9]+", value.lower()):
        escaped = rf"\b{escaped}(?:s|ed|ing)?\b"
    return value, re.compile(escaped, re.I)


def concept_survey(
    joined: list[dict], catalogue: dict[str, dict]
) -> tuple[list[dict], list[dict], list[dict]]:
    by_concept = []
    concept_rows = []
    examples = []
    official_texts = {item_id: official_text(item) for item_id, item in catalogue.items()}
    raw_schema_fields = {"object_type"}
    for row in joined:
        for task in ("colors", "shape", "motifs"):
            raw_schema_fields.update((row.get(task) or {}).keys())
    visual_query_text = " ".join(
        query["text_en"]
        for path in (
            ROOT / "eval/visual_dev/query_judgments.jsonl",
            ROOT / "eval/visual_test/query_judgments.jsonl",
        )
        for query in read_jsonl(path)
    ).lower()
    for concept_key, spec in CONCEPTS.items():
        term_counts = Counter()
        term_rows_count = Counter()
        term_official_count = Counter()
        term_examples: dict[str, list[dict]] = defaultdict(list)
        for term in spec["terms"]:
            _, rx = term_rows(term)
            for row in joined:
                if not row["caption_en"]:
                    continue
                text = f"{row['caption_en']} {row['caption_zh']} {row['uncertainty']}"
                if rx.search(text):
                    term_counts[term] += 1
                    term_rows_count[term] += 1
                    if len(term_examples[term]) < 3:
                        term_examples[term].append(
                            {
                                "source": row["source"],
                                "alias": row["alias"],
                                "item_id": row["item_id"],
                                "annotator": row["annotator"],
                                "caption_en": row["caption_en"],
                                "label_source": "ai_derived",
                            }
                        )
            term_official_count[term] = sum(
                bool(rx.search(text)) for text in official_texts.values()
            )
        row_terms = {}
        for term in spec["terms"]:
            _, rx = term_rows(term)
            row_terms[term] = rx
        annotated_items = {r["item_id"] for r in joined if r["caption_en"]}
        caption_by_item = {}
        for row in joined:
            if row["caption_en"]:
                caption_by_item[row["item_id"]] = row["caption_en"]
        ai_items = {
            item_id
            for item_id in annotated_items
            if any(rx.search(caption_by_item[item_id]) for rx in row_terms.values())
        }
        official_items = {
            item_id
            for item_id, text in official_texts.items()
            if any(rx.search(text) for rx in row_terms.values())
        }
        for term in spec["terms"]:
            concept_rows.append(
                {
                    "concept": concept_key,
                    "term": term,
                    "ai_caption_rows": term_counts[term],
                    "ai_caption_item_rows": term_rows_count[term],
                    "catalogue_text_rows": term_official_count[term],
                    "catalogue_text_rate": term_official_count[term] / len(catalogue),
                    "registered": any(
                        term.lower() == value.lower()
                        for values in VOCAB.values()
                        for value in values
                    ),
                    "in_raw_structured_schema": term.lower()
                    in {field.lower() for field in raw_schema_fields},
                    "queried_by_visual_judgments": term.lower() in visual_query_text,
                    "source_rows": term_counts[term],
                    "roundtrips": len(term_examples[term]),
                }
            )
            if term_examples[term]:
                for ex in term_examples[term]:
                    examples.append({"concept": concept_key, "term": term, **ex})
        by_concept.append(
            {
                "concept": concept_key,
                "title": spec["title"],
                "description": spec["description"],
                "ai_caption_source_rows": sum(term_counts.values()),
                "ai_caption_unique_items": len(ai_items),
                "ai_caption_unique_item_rate": len(ai_items) / len(annotated_items),
                "catalogue_text_items": len(official_items),
                "catalogue_text_item_rate": len(official_items) / len(catalogue),
                "top_terms": [
                    {
                        "term": term,
                        "ai_caption_rows": count,
                        "catalogue_text_rows": term_official_count[term],
                    }
                    for term, count in term_counts.most_common(12)
                ],
            }
        )
    return by_concept, concept_rows, examples


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalogue", type=Path, default=CATALOGUE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    catalogue_rows = read_jsonl(args.catalogue)
    catalogue = {r["item_id"]: r for r in catalogue_rows}
    if len(catalogue) != len(catalogue_rows):
        raise ValueError("Duplicate item_id in official catalogue")

    source_manifest, joined = survey_sources(catalogue)
    jsonl_inventory = inventory_all_jsonl(catalogue)
    query_rows = query_judgment_counts()
    qrel_rows = metadata_qrel_counts(catalogue)
    counts, overlap_rows = annotation_counts(catalogue)
    concepts, concept_rows, examples = concept_survey(joined, catalogue)
    registered = registered_support(catalogue_rows)
    all_annotations = {r["item_id"] for r in joined}
    summary = {
        "schema": "curio-v4-label-survey-v1",
        "catalogue": str(args.catalogue.resolve()),
        "catalogue_rows": len(catalogue_rows),
        "catalogue_sha256": sha(args.catalogue),
        "raw_annotation_rows": len(joined),
        "raw_annotation_unique_items": len(all_annotations),
        "raw_annotation_source_files": len(source_manifest),
        "source_rows": sum(r["rows"] for r in source_manifest),
        "source_unique_items_within_file": sum(
            len({r["item_id"] for r in joined if r["source"] == r2["source"]})
            for r2 in source_manifest
        ),
        "sources": source_manifest,
        "eval_jsonl_inventory": jsonl_inventory,
        "query_judgment_summary": {
            "rows": len(query_rows),
            "images": sum(row["images"] for row in query_rows),
            "by_split": dict(Counter(row["split"] for row in query_rows)),
            "by_type": dict(Counter(row["type"] for row in query_rows)),
            "grade_counts": {
                grade: sum(row[f"grade_{grade}"] for row in query_rows)
                for grade in ("0", "1", "2", "3")
            },
        },
        "metadata_qrel_summary": {
            "rows": len(qrel_rows),
            "graded_items": sum(row["graded_items"] for row in qrel_rows),
            "official_items_resolved": sum(row["official_items"] for row in qrel_rows),
            "grade_counts": {
                grade: sum(row[f"grade_{grade}"] for row in qrel_rows)
                for grade in ("0", "1", "2", "3")
            },
        },
        "raw_schema": raw_schema_registry(source_manifest, joined),
        "registered_vocabulary": registered,
        "concept_categories": concepts,
        "high_value_candidates": [
            r for r in concept_rows if r["ai_caption_rows"] >= 3 or r["catalogue_text_rows"] >= 10
        ][:200],
        "official_colors_populated": sum(bool(r.get("colors")) for r in catalogue_rows),
        "official_motifs_populated": sum(bool(r.get("motifs")) for r in catalogue_rows),
        "official_shape_populated": sum(bool(r.get("shape")) for r in catalogue_rows),
        "official_support": official_pool_support(catalogue, joined),
        "annotation_rollup": counts,
        "interpretation": {
            "official_evidence": "Catalogue occurrences are text/category support, not independent visual ground truth.",
            "ai_evidence": "Caption occurrences are AI-derived visual descriptions; they are not human gold.",
            "ranking": "Term counts are lexical mentions weighted by source row frequency; normalized concepts require manual review.",
        },
    }
    dump_json(output / "summary.json", summary)
    dump_json(output / "source_manifest.json", source_manifest)
    dump_json(
        output / "field_counts.json",
        {
            "sources": source_field_counts(source_manifest, joined),
            "tasks": field_value_counts(source_manifest, joined),
        },
    )
    dump_jsonl(output / "joined_annotations.jsonl", joined)
    dump_jsonl(output / "overlap_by_item.jsonl", overlap_rows)
    dump_json(output / "concept_summary.json", concepts)
    dump_json(output / "concept_candidates.json", concept_rows)
    dump_json(output / "eval_jsonl_inventory.json", jsonl_inventory)
    dump_jsonl(output / "visual_query_judgment_counts.jsonl", query_rows)
    dump_jsonl(output / "metadata_qrel_counts.jsonl", qrel_rows)
    dump_jsonl(output / "evidence_examples.jsonl", examples)
    write_csv(output / "source_manifest.csv", source_manifest)
    write_csv(output / "concept_candidates.csv", concept_rows)
    print(
        json.dumps(
            {
                "catalogue": len(catalogue_rows),
                "sources": len(source_manifest),
                "raw_rows": len(joined),
                "unique_items": len(all_annotations),
                "output": str(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
