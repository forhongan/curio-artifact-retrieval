"""Create blind, held-out image sheets without using object attributes or rankings."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import tempfile

from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ("met", "cma")
SOURCE_ALIASES = {"met": {"met"}, "cma": {"cma", "cleveland"}}


def source_bucket(value: str | None) -> str | None:
    value = str(value or "").lower()
    for bucket, aliases in SOURCE_ALIASES.items():
        if value in aliases:
            return bucket
    return None


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def raw_path_for(row: dict, data_path: Path) -> Path | None:
    source, source_id = str(row.get("source") or ""), str(row.get("source_id") or "")
    if not source_id and "_" in str(row.get("item_id")):
        source_id = str(row["item_id"]).split("_", 1)[1]
    candidates = [
        ROOT / "data" / "raw" / source / f"{source_id}.jpg",
        data_path.parent / "raw" / source / f"{source_id}.jpg",
    ]
    return next((path for path in candidates if path.is_file()), None)


def select(rows: list[dict], seed: str, count: int = 60, split: str = "test") -> list[dict]:
    """Deduplicate transitively across the full catalog, then sample held-out items."""
    if split not in {"train", "dev", "test"}:
        raise ValueError("split must be train, dev, or test")
    if count < 1:
        raise ValueError("count must be positive")
    ids = [str(row["item_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate item_id in catalog")
    parent = {item_id: item_id for item_id in ids}

    def find(item_id):
        while parent[item_id] != item_id:
            parent[item_id] = parent[parent[item_id]]
            item_id = parent[item_id]
        return item_id

    def union(first, second):
        a, b = find(first), find(second)
        if a != b:
            parent[max(a, b)] = min(a, b)

    for field in ("sha256", "phash", "phash_group", "duplicate_group"):
        seen = {}
        for row in rows:
            value = row.get(field)
            if value:
                key = str(value)
                item_id = str(row["item_id"])
                if key in seen:
                    union(item_id, seen[key])
                else:
                    seen[key] = item_id
    for row in rows:
        for field in ("duplicate_sha256_of", "near_duplicate_of"):
            linked = str(row.get(field) or "")
            if linked in parent:
                union(str(row["item_id"]), linked)

    def order(row):
        return hashlib.sha256(f"{seed}{row['item_id']}".encode("utf-8")).hexdigest()

    candidates = sorted(
        (row for row in rows if row.get("split") == split),
        key=lambda row: (order(row), str(row["item_id"])),
    )
    selected, groups = [], set()
    targets = {"met": count // 2, "cma": count - count // 2}
    for source in SOURCES:
        for row in candidates:
            group = find(str(row["item_id"]))
            if (
                source_bucket(row.get("source")) == source
                and group not in groups
                and targets[source] > 0
            ):
                selected.append(row)
                groups.add(group)
                targets[source] -= 1
    for row in candidates:
        group = find(str(row["item_id"]))
        if len(selected) >= count:
            break
        if group not in groups:
            selected.append(row)
            groups.add(group)
    if len(selected) != count:
        raise ValueError(f"Only {len(selected)} unique test objects available; need {count}")
    return sorted(selected, key=lambda row: (order(row), str(row["item_id"])))


def prepare(
    data: Path,
    output: Path,
    sheets: Path,
    seed: str,
    count: int = 60,
    force=False,
    split: str = "test",
):
    data = data.resolve()
    raw = data.read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    selected = select(rows, seed, count, split)
    output, sheets = output.resolve(), sheets.resolve()
    # Never silently replace a prepared or annotated audit. Preserve all files on --force.
    existing = any(path.exists() and any(path.iterdir()) for path in (output, sheets))
    if existing and not force:
        raise FileExistsError("Audit output already exists; use --force to rebuild generated files")
    if force:
        for name in ("gallery_ids.json", "blind_map.json", "manifest.json"):
            (output / name).unlink(missing_ok=True)
        for old_sheet in sheets.glob("sheet_*.png") if sheets.exists() else ():
            old_sheet.unlink()
    image_records = []
    for row in selected:
        path = Path(row["image_path"])
        if not path.is_absolute():
            path = ROOT / path
            if not path.is_file():
                path = data.parent / row["image_path"]
        if not path.is_file():
            raise ValueError(f"Missing processed image for {row['item_id']}")
        processed_hash = digest(path)
        expected_raw_hash = row.get("sha256")
        expected_processed_hash = row.get("image_sha256")
        raw_path = raw_path_for(row, data)
        if expected_processed_hash and processed_hash != expected_processed_hash:
            raise ValueError(f"Processed image SHA256 mismatch for {row['item_id']}")
        if not expected_processed_hash and raw_path is None:
            raise ValueError(f"{row['item_id']} needs image_sha256 or a resolvable raw image")
        if raw_path is not None and expected_raw_hash and digest(raw_path) != expected_raw_hash:
            raise ValueError(f"Raw image SHA256 mismatch for {row['item_id']}")
        image_records.append((path, expected_raw_hash, processed_hash))
    prefix = {"train": "T", "dev": "D", "test": "V"}[split]
    mapping = [
        {
            "alias": f"{prefix}{i:03d}",
            "item_id": str(row["item_id"]),
            "raw_sha256": raw_hash,
            "processed_sha256": processed_hash,
        }
        for i, (row, (_, raw_hash, processed_hash)) in enumerate(zip(selected, image_records), 1)
    ]
    manifest = {
        "data_sha256": hashlib.sha256(raw).hexdigest(),
        "split": split,
        "seed": seed,
        "count": count,
        "source_counts": dict(Counter(row["source"] for row in selected)),
        "selection": "SHA256(seed + item_id), source-balanced then backfilled; transitive duplicate groups",
        "image_hashes": "blind_map.raw_sha256 is the original/raw image hash (catalog sha256); blind_map.processed_sha256 is the exact rendered source file hash (catalog image_sha256 when present)",
        "deduplication_fields": [
            "sha256",
            "phash",
            "phash_group",
            "duplicate_group",
            "duplicate_sha256_of",
            "near_duplicate_of",
        ],
        "label_source": "unlabeled; AI-generated judgments must be labeled AI, not human",
    }
    # Render and validate every image before replacing any output files.
    with tempfile.TemporaryDirectory() as temp:
        stage = Path(temp)
        (stage / "gallery_ids.json").write_text(
            json.dumps([x["item_id"] for x in mapping], indent=2), encoding="utf-8"
        )
        (stage / "blind_map.json").write_text(json.dumps(mapping, indent=2), encoding="utf-8")
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        for start in range(0, count, 12):
            canvas = Image.new("RGB", (960, 1200), "white")
            draw = ImageDraw.Draw(canvas)
            for offset, (path, _, _) in enumerate(image_records[start : start + 12]):
                x, y = (offset % 3) * 320, (offset // 3) * 300
                with Image.open(path) as image:
                    image = ImageOps.exif_transpose(image).convert("RGB")
                    image.thumbnail((304, 264), Image.Resampling.LANCZOS)
                    canvas.paste(
                        image, (x + (320 - image.width) // 2, y + (264 - image.height) // 2)
                    )
                draw.text((x + 10, y + 273), mapping[start + offset]["alias"], fill="black")
            canvas.save(stage / f"sheet_{start // 12 + 1:03d}.png")
        output.mkdir(parents=True, exist_ok=True)
        sheets.mkdir(parents=True, exist_ok=True)
        for name in ("gallery_ids.json", "blind_map.json", "manifest.json"):
            (output / name).write_bytes((stage / name).read_bytes())
        for path in stage.glob("sheet_*.png"):
            (sheets / path.name).write_bytes(path.read_bytes())
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Frozen catalog JSONL snapshot")
    parser.add_argument("--output", default="eval/visual_audit")
    parser.add_argument("--sheets", default="docs/qa/visual_audit")
    parser.add_argument("--seed", default="5523")
    parser.add_argument("--split", choices=("train", "dev", "test"), default="test")
    parser.add_argument("--count", type=int, default=60)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                Path(args.data),
                Path(args.output),
                Path(args.sheets),
                args.seed,
                args.count,
                args.force,
                args.split,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
