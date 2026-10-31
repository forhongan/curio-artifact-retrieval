"""Acquire a separate expansion gallery while preserving the frozen reference bank."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil

from PIL import Image

from build_reference_bank import Builder, ROOT, categories, chinese
from build_expansion_catalog import build
from curio.retrieve.references import GalleryExclusions, image_fingerprints


def acquire(root: Path, target: int):
    output = root / "data/expansion_v3/acquisition"
    output.mkdir(parents=True, exist_ok=True)
    # Reuse only this read-only gallery fingerprint computation, never responses
    # or bank rows. Official expansion responses receive their own cache.
    original_fp = root / "data/references/gallery_fingerprints.json"
    if original_fp.exists():
        shutil.copyfile(original_fp, output / "gallery_fingerprints.json")
    builder = Builder(root, output=output)
    bank_path = root / "data/references/bank.jsonl"
    bank_before = hashlib.sha256(bank_path.read_bytes()).hexdigest()
    refs = [json.loads(line) for line in bank_path.read_text(encoding="utf-8").splitlines()]
    frozen_rows = [json.loads(line) for line in (root / "data/snapshots/collection_v2.jsonl").read_text(encoding="utf-8").splitlines()]
    expansion_path = root / "data/snapshots/expansion_v3.jsonl"
    existing = [json.loads(line) for line in expansion_path.read_text(encoding="utf-8").splitlines()] if expansion_path.exists() else []
    exclusions = frozen_rows + existing + [{**r, "item_id": r["reference_id"], "object_group": r["reference_id"], "phash": r.get("ahash256", "")} for r in refs]
    raw_records = {}
    for row in exclusions:
        raw_path = root / row.get("metadata_path", f"data/raw/{row['source']}/{row['source_id']}.json")
        if raw_path.exists():
            raw_records[row["item_id"]] = json.loads(raw_path.read_bytes())
    guard = GalleryExclusions(exclusions, raw_records)
    for item_id, value in builder.exclusions.phashes:
        guard.phashes.append((item_id, value))
    for row in existing + refs:
        with Image.open(root / row["image_path"]) as image:
            guard.add_gallery_phash(row.get("item_id", row.get("reference_id")), image_fingerprints(image)["phash64"])
    builder.exclusions = guard
    candidates = []
    for source in ("cleveland", "met"):
        for path in sorted((root / "data/raw" / source).glob("*.json")):
            record = json.loads(path.read_bytes())
            cats = categories(record, source)
            rid = f"{'cma' if source == 'cleveland' else 'met'}_{record.get('id' if source == 'cleveland' else 'objectID')}"
            if rid in guard.ids or not chinese(record) or not set(cats) & {"painting", "calligraphy", "book", "textile"}:
                continue
            candidates.append((source, record, cats))
    # Mix independent classes before filling with albums and scrolls.
    candidates.sort(key=lambda entry: ("textile" not in entry[2], "calligraphy" not in entry[2], "book" not in entry[2], str(entry[1].get("id", entry[1].get("objectID")))))
    need = max(0, target - len(existing))
    for start in range(0, len(candidates), 4):
        if len(builder.bank) >= need:
            break
        batch = candidates[start:start + 4]
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(builder.attempt, source, record, ["Chinese " + c for c in cats], 100000)
                       for source, record, cats in batch]
            for future in futures:
                future.result()
        if builder.bank:
            # Checkpoint acquisitions without touching reference bank.
            builder.save(100000)
    build(root, additional_references=builder.bank)
    report_path = root / "data/expansion_v3/report.json"
    report = json.loads(report_path.read_bytes())
    report.update(reference_bank_sha256=bank_before,
                  reference_bank_unchanged=bank_before == hashlib.sha256(bank_path.read_bytes()).hexdigest(),
                  acquisition_count=len(builder.bank), acquisition_failures=len(builder.failures),
                  exclusion_count=len(builder.ledger), target_count=target,
                  whole_gallery_and_reference_exclusion=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--target", type=int, default=100)
    args = parser.parse_args()
    acquire(args.root, args.target)
