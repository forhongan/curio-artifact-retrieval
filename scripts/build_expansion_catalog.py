"""Promote independent painting/book references to a separate expansion shard.

This does not edit the frozen gallery. The promoted items leave the reference
bank before they can be searched as expansion-gallery images.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from curio.retrieve.references import image_fingerprints  # noqa: E402
from build_reference_bank import categories  # noqa: E402
from data_audit_relations import build_relation_report  # noqa: E402


def build(root: Path = ROOT, additional_references: list[dict] | None = None):
    bank_path = root / "data/references/bank.jsonl"
    refs = (additional_references if additional_references is not None else
            [json.loads(line) for line in bank_path.read_text(encoding="utf-8").splitlines()])
    frozen = root / "data/snapshots/collection_v2.jsonl"
    before = hashlib.sha256(frozen.read_bytes()).hexdigest()
    expansion_path = root / "data/snapshots/expansion_v3.jsonl"
    extension = ([json.loads(line) for line in expansion_path.read_text(encoding="utf-8").splitlines()]
                 if expansion_path.exists() else [])
    existing = {row["item_id"] for row in extension}
    remaining = []
    for ref in refs:
        record = json.loads((root / ref["metadata_path"]).read_text(encoding="utf-8"))
        ref["categories"] = categories(record, ref["source"])
        if additional_references is None and not set(ref["categories"]) & {"painting", "calligraphy"}:
            remaining.append(ref)
            continue
        rid = ref["reference_id"]
        if rid in existing:
            continue
        ip = root / "data/processed/images" / f"extension_{rid}.jpg"
        tp = root / "data/processed/thumbs" / f"extension_{rid}.jpg"
        with Image.open(root / ref["image_path"]) as image:
            image = image.convert("RGB")
            image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            image.save(ip, "JPEG", quality=94, optimize=True)
            fp = image_fingerprints(image)
            width, height = image.size
            image.thumbnail((320, 320), Image.Resampling.LANCZOS)
            image.save(tp, "JPEG", quality=88, optimize=True)
        medium = ref.get("medium", "")
        material = "paper" if "paper" in medium.casefold() else "textile" if "silk" in medium.casefold() else ""
        extension.append({
            "item_id": rid, "source": ref["source"], "source_id": ref["source_id"],
            "title": ref["title"], "object_name": "Calligraphy" if "calligraphy" in ref["categories"] else "Painting" if "painting" in ref["categories"] else "Textile or book",
            "classification": record.get("type" if ref["source"] == "cleveland" else "classification", ""),
            "object_type": "calligraphy" if "calligraphy" in ref["categories"] else "book" if "book" in ref["categories"] else "painting" if "painting" in ref["categories"] else "textile" if "textile" in ref["categories"] else "book",
            "medium": medium, "material": material, "culture": str(ref.get("culture", "")),
            "period": ref.get("date", ""), "year_from": record.get("creation_date_earliest", record.get("objectBeginDate")),
            "year_to": record.get("creation_date_latest", record.get("objectEndDate")),
            "description": ref.get("description", ""), "tags": ref["categories"], "motifs": [], "colors": [],
            "image_url": ref["image_url"], "source_url": ref["source_url"], "license": "CC0",
            "image_path": ip.relative_to(root).as_posix(), "thumb_path": tp.relative_to(root).as_posix(),
            "width": width, "height": height, "sha256": ref["original_image_sha256"],
            "image_sha256": hashlib.sha256(ip.read_bytes()).hexdigest(),
            "thumb_sha256": hashlib.sha256(tp.read_bytes()).hexdigest(), "phash": fp["ahash256"],
            "phash64": fp["phash64"], "object_group": rid, "split": "unassigned",
            "split_policy": "expansion_v3_independent_shard_not_a_new_benchmark_split",
            "origin_basis": "official museum metadata Chinese culture/country",
            "metadata_path": ref["metadata_path"], "metadata_api_url": ref["metadata_api_url"],
            "metadata_sha256": ref["metadata_sha256"], "verified_at": ref["verified_at"],
            "reference_exclusion": "removed from reference bank; references rebuilt against both catalogs",
        })
    def write(path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    # Refresh existing rows too: source classifications are evidence, while the
    # derived object categories are separately documented rules.
    raw_records = {}
    for row in extension:
        record = json.loads((root / row["metadata_path"]).read_bytes())
        raw_records[row["item_id"]] = record
        cats = categories(record, row["source"])
        row["tags"] = cats
        row["classification"] = record.get("type" if row["source"] == "cleveland" else "classification", "")
        row["object_type"] = ("calligraphy" if "calligraphy" in cats else "book" if "book" in cats else
                              "painting" if "painting" in cats else "textile" if "textile" in cats else "unknown")
        row["object_name"] = row["object_type"].capitalize()
        row["split"] = "unassigned"
    relations = build_relation_report(extension, root=root, raw_metadata=raw_records)
    # Accession series are conservatively grouped for this retrieval-only shard;
    # this grouping is not a claim that every series depicts one physical object.
    parents = {row["item_id"]: row["item_id"] for row in extension}
    def find(item_id):
        while parents[item_id] != item_id:
            parents[item_id] = parents[parents[item_id]]
            item_id = parents[item_id]
        return item_id
    series_groups = {}
    for row in extension:
        group = relations["item_to_group"][row["item_id"]]
        parents[find(row["item_id"])] = find(group)
        raw = raw_records[row["item_id"]]
        acc = raw.get("accessionNumber" if row["source"] == "met" else "accession_number", "")
        series = (row["source"], ".".join(acc.split(".")[:2]))
        if acc and series in series_groups:
            parents[find(row["item_id"])] = find(series_groups[series])
        elif acc:
            series_groups[series] = row["item_id"]
    for row in extension:
        row["object_group"] = find(row["item_id"])
        row["grouping_basis"] = "official relations and conservative source accession series"
    write(expansion_path, extension)
    if additional_references is None:
        write(bank_path, remaining)
    else:
        remaining = [json.loads(line) for line in bank_path.read_text(encoding="utf-8").splitlines()]
    report_path = root / "data/expansion_v3/report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {"count": len(extension), "category_counts": dict(Counter(tag for row in extension for tag in row["tags"])),
              "source_counts": dict(Counter(row["source"] for row in extension)),
              "object_type_counts": dict(Counter(row["object_type"] for row in extension)),
              "conservative_family_count": len({row["object_group"] for row in extension}),
              "family_policy": "Source accession series and official relations; families can include multiple distinct objects or pages, so image count is not independent physical-object count",
              "frozen_gallery_sha256": before,
              "frozen_gallery_unchanged": before == hashlib.sha256(frozen.read_bytes()).hexdigest(),
              "reference_ids_disjoint": not ({r["reference_id"] for r in remaining} & {r["item_id"] for r in extension}),
              "role": "Separate searchable category expansion candidate shard, not automatically merged into production or evaluation"}
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    build()
