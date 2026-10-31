"""Audit frozen references and the separate extension against both catalogs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from curio.retrieve.references import GalleryExclusions, normalize_record  # noqa: E402


def audit(root=ROOT):
    def rows(path):
        return [json.loads(line) for line in (root / path).read_text(encoding="utf-8").splitlines()]
    def digest(path):
        return hashlib.sha256((root / path).read_bytes()).hexdigest()
    frozen_path = "data/snapshots/collection_v2.jsonl"
    bank_path = "data/references/bank.jsonl"
    expansion_path = "data/snapshots/expansion_v3.jsonl"
    gallery, refs, expansion = rows(frozen_path), rows(bank_path), rows(expansion_path)
    supplemental_path = "data/references/supplemental.jsonl"
    supplemental = rows(supplemental_path) if (root / supplemental_path).exists() else []
    ref_rows = [{**r, "item_id": r["reference_id"], "object_group": r["reference_id"], "phash": r["ahash256"]} for r in refs]
    supplemental_rows = [{**r, "item_id": r["reference_id"], "object_group": r["reference_id"], "phash": r["ahash256"]} for r in supplemental]
    raw = {}
    for row in gallery + expansion + ref_rows + supplemental_rows:
        path = row.get("metadata_path")
        if path is None:
            path = f"data/raw/{row['source']}/{row['source_id']}.json"
        raw[row["item_id"]] = json.loads((root / path).read_bytes())
    fingerprint_cache = json.loads((root / "data/references/gallery_fingerprints.json").read_bytes())["phash64"]
    issues = []
    for target, blockers in ((refs, gallery + expansion), (expansion, gallery + ref_rows + supplemental_rows),
                             (supplemental, gallery + expansion + ref_rows)):
        guard = GalleryExclusions(blockers, raw)
        for item_id, value in fingerprint_cache.items():
            guard.add_gallery_phash(item_id, value)
        for row in blockers:
            if row.get("phash64"):
                guard.add_gallery_phash(row["item_id"], row["phash64"])
        for row in target:
            rid = row.get("reference_id", row.get("item_id"))
            candidate = normalize_record(row["source"], raw[rid])
            candidate.update(image_sha256=row["image_sha256"],
                             original_image_sha256=row.get("original_image_sha256", row.get("sha256")),
                             phash64=row["phash64"], ahash256=row.get("ahash256", row.get("phash")))
            reason = guard.reason(candidate)
            if reason:
                issues.append({"item_id": rid, **reason})
            if digest(row["image_path"]) != row["image_sha256"]:
                issues.append({"item_id": rid, "reason": "image_hash_mismatch"})
            if candidate["metadata_sha256"] != row["metadata_sha256"]:
                issues.append({"item_id": rid, "reason": "metadata_hash_mismatch"})
            if row.get("thumb_path") and digest(row["thumb_path"]) != row["thumb_sha256"]:
                issues.append({"item_id": rid, "reason": "thumbnail_hash_mismatch"})
    report = {"passed": not issues, "frozen_gallery_count": len(gallery),
              "reference_count": len(refs), "expansion_image_count": len(expansion),
              "supplemental_reference_count": len(supplemental),
              "supplemental_sha256": digest(supplemental_path) if supplemental else None,
              "frozen_gallery_sha256": digest(frozen_path), "reference_bank_sha256": digest(bank_path),
              "expansion_sha256": digest(expansion_path), "issues": issues,
              "checks": ["official CC0/public-domain metadata and official HTTPS image host",
                         "metadata canonical SHA256 and image/thumbnail byte SHA256",
                         "whole frozen gallery IDs and object groups",
                         "official source accession lettered parts and parent records",
                         "Cleveland official related works and inseparable parts",
                         "exact raw/processed/thumbnail SHA256",
                         "256-bit average-hash Hamming distance <=4",
                         "64-bit perceptual-hash Hamming distance <=6",
                         "references isolated from expansion, expansion isolated from references"],
              "limitation": "Conservative automatic checks reduce leakage; they do not prove identity completeness when museum relationships or alternate views are undocumented."}
    path = root / "data/references/isolation_audit.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)
    return not issues


if __name__ == "__main__":
    raise SystemExit(0 if audit() else 1)
