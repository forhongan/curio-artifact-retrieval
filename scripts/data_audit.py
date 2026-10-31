"""Audit frozen files without repair; write only audit.json and licenses.csv."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import time

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from curio.represent.validation import validate_object_splits  # noqa: E402
from data_audit_relations import build_relation_report  # noqa: E402


def digest(content):
    return hashlib.sha256(content).hexdigest()


def split_for(item_id):
    bucket = int(digest(str(item_id).encode())[:8], 16) % 10
    return "train" if bucket < 8 else "dev" if bucket == 8 else "test"


def text(value):
    if value is None:
        return ""
    return "; ".join(map(str, value)) if isinstance(value, list) else str(value).strip()


def audit_item(row, root):
    """Hash actual bytes and force full PIL pixel decode of all three files."""
    item_id = row["item_id"]
    issues, counts, sizes, hashes = [], Counter(), {}, {}

    def issue(code, **details):
        issues.append({"item_id": item_id, "code": code, **details})

    source, sid = row.get("source"), str(row.get("source_id", ""))
    if source not in {"met", "cleveland"}:
        issue("source", actual=source)
    prefix = "met" if source == "met" else "cma"
    if item_id != f"{prefix}_{sid}":
        issue("source_id", source=source, source_id=sid)
    metadata_path = root / "data/raw" / str(source) / f"{sid}.json"
    raw, metadata_sha = {}, ""
    try:
        content = metadata_path.read_bytes()
        metadata_sha, raw = digest(content), json.loads(content)
        if not isinstance(raw, dict):
            raise ValueError("metadata must be an object")
        counts["raw_metadata_loaded"] += 1
    except Exception as exc:
        raw = {}
        issue("raw_metadata", error=f"{type(exc).__name__}: {exc}")
    raw_id = raw.get("objectID" if source == "met" else "id")
    if str(raw_id) != sid:
        issue("raw_source_id", expected=sid, actual=raw_id)
    rights_field = "isPublicDomain" if source == "met" else "share_license_status"
    rights = raw.get(rights_field)
    rights_ok = rights is True if source == "met" else rights == "CC0"
    counts["official_rights_verified"] += int(rights_ok)
    if not rights_ok or row.get("license") != "CC0":
        issue("license", raw_field=rights_field, raw_value=rights, catalog_license=row.get("license"))
    origin = " ".join(text(raw.get(key)) for key in ("country", "culture", "geographyType", "find_spot"))
    china_ok = bool(re.search(r"\b(?:china|chinese)\b", origin, re.I))
    country = text(raw.get("country")).lower()
    if source == "met" and country and country not in {"china", "china (?)"}:
        china_ok = False
    counts["china_origin_verified"] += int(china_ok)
    if not china_ok:
        issue("origin", raw_evidence=origin)
    y0, y1 = row.get("year_from"), row.get("year_to")
    valid = type(y0) is int and type(y1) is int and y0 <= y1
    counts["pre1912_start_verified"] += int(valid and y0 < 1912)
    counts["pre1912_entire_range_verified"] += int(valid and y1 < 1912)
    if not valid:
        issue("date_range", year_from=y0, year_to=y1)
    elif y0 >= 1912:
        issue("date_start_not_pre1912", year_from=y0, year_to=y1)
    elif y1 >= 1912:
        issue("date_range_crosses_1912", severity="warning", year_from=y0, year_to=y1)
    raw_dates = ((raw.get("objectBeginDate"), raw.get("objectEndDate")) if source == "met"
                 else (raw.get("creation_date_earliest"), raw.get("creation_date_latest")))
    if all(type(year) is int and year != 0 for year in raw_dates):
        if (y0, y1) != tuple(sorted(raw_dates)):
            issue("raw_date_mismatch", catalog=[y0, y1], raw=list(raw_dates))
        else:
            counts["raw_numeric_dates_match"] += 1
    paths = {"raw": root / "data/raw" / str(source) / f"{sid}.jpg",
             "processed": root / str(row.get("image_path", "")),
             "thumb": root / str(row.get("thumb_path", ""))}
    fields = {"raw": "sha256", "processed": "image_sha256", "thumb": "thumb_sha256"}
    for kind, path in paths.items():
        try:
            content = path.read_bytes()
            hashes[kind] = digest(content)
            counts[f"{kind}_files_hashed"] += 1
            if hashes[kind] != row.get(fields[kind]):
                issue(f"{kind}_sha256", expected=row.get(fields[kind]), actual=hashes[kind])
            else:
                counts[f"{kind}_sha256_verified"] += 1
            with Image.open(io.BytesIO(content)) as image:
                image.load()
                sizes[kind] = image.size
                counts[f"{kind}_decoded"] += 1
                if kind != "raw" and (image.mode != "RGB" or image.format != "JPEG"):
                    issue(f"{kind}_format", mode=image.mode, format=image.format)
        except Exception as exc:
            issue(f"{kind}_decode", path=str(path), error=f"{type(exc).__name__}: {exc}")
    if "raw" in sizes and (min(sizes["raw"]) < 256 or max(sizes["raw"]) < 512):
        issue("raw_quality_floor", size=list(sizes["raw"]))
    if "processed" in sizes:
        if sizes["processed"] != (row.get("width"), row.get("height")):
            issue("processed_catalog_dimensions", actual=list(sizes["processed"]), expected=[row.get("width"), row.get("height")])
        if max(sizes["processed"]) > 1024:
            issue("processed_size_limit", size=list(sizes["processed"]))
    if "thumb" in sizes and max(sizes["thumb"]) > 320:
        issue("thumb_size_limit", size=list(sizes["thumb"]))
    for parent, derived, edge in (("raw", "processed", 1024), ("processed", "thumb", 320)):
        if parent in sizes and derived in sizes:
            with Image.new("1", sizes[parent]) as expected:
                expected.thumbnail((edge, edge), Image.Resampling.LANCZOS)
                if sizes[derived] != expected.size:
                    issue(f"{derived}_resize_dimensions", actual=list(sizes[derived]), expected=list(expected.size))
    license_row = {"item_id": item_id, "source": source, "source_id": sid, "license": row.get("license"),
                   "source_url": row.get("source_url"), "image_url": row.get("image_url"), "credit_line": row.get("credit_line"),
                   "rights_field": rights_field, "rights_value": json.dumps(rights), "rights_verified": rights_ok,
                   "raw_json_path": str(metadata_path.relative_to(root)).replace("\\", "/"), "raw_json_sha256": metadata_sha,
                   "raw_sha256": hashes.get("raw", ""), "image_sha256": hashes.get("processed", ""), "thumb_sha256": hashes.get("thumb", ""),
                   "china_origin_verified": china_ok, "origin_evidence": origin, "year_from": y0, "year_to": y1, "split": row.get("split")}
    return {"item_id": item_id, "issues": issues, "counts": counts, "raw": raw, "hashes": hashes, "license": license_row}


def validate_grouped_policy(rows, root, raw_metadata):
    """Independently reconstruct v2 assignments from immutable v1 and anchors."""
    issues = []
    source_path = root / "data/snapshots/collection_v1.jsonl"
    try:
        source_bytes = source_path.read_bytes()
        source = [json.loads(line) for line in source_bytes.decode("utf-8").splitlines() if line.strip()]
        original = {row["item_id"]: row for row in source}
        if set(original) != {row["item_id"] for row in rows}:
            raise ValueError("grouped snapshot item IDs differ from immutable v1")
        relations = build_relation_report(source, root, raw_metadata=raw_metadata)
        mapping = relations["item_to_group"]
        priority = {"train": 0, "dev": 1, "test": 2}
        expected = {group["group_id"]: max(group["splits"], key=priority.get) for group in relations["groups"]}
        anchors, anchor_evidence = {}, {}
        for split in ("dev", "test"):
            path = root / f"eval/visual_{split}/gallery_ids.json"
            content = path.read_bytes()
            ids = json.loads(content)
            anchor_evidence[split] = {"count": len(ids), "sha256": digest(content)}
            for item_id in ids:
                group_id = mapping[item_id]
                if group_id in anchors and anchors[group_id] != split:
                    raise ValueError("blind dev/test anchors share a connected object group")
                anchors[group_id] = split
        expected.update(anchors)
        ignored = {"split", "object_group", "previous_split", "split_policy"}
        for row in rows:
            item_id, old = row["item_id"], original[row["item_id"]]
            group_id = mapping[item_id]
            if (row.get("split_policy") != "official-relations-components-v2-heldout-priority"
                    or row.get("object_group") != group_id or row.get("previous_split") != old["split"]
                    or row["split"] != expected[group_id]):
                issues.append({"code": "grouped_split_policy", "item_id": item_id, "expected_group": group_id,
                               "expected_split": expected[group_id], "actual_split": row["split"]})
            if {key: value for key, value in row.items() if key not in ignored} != {key: value for key, value in old.items() if key not in ignored}:
                issues.append({"code": "grouped_non_split_data_changed", "item_id": item_id})
            if old["split"] in {"dev", "test"} and row["split"] == "train":
                issues.append({"code": "heldout_moved_to_train", "item_id": item_id})
        return {"passed": not issues, "policy": "official-relations-components-v2-heldout-priority",
                "source_sha256": digest(source_bytes), "groups": relations["group_count"],
                "anchor_groups": len(anchors), "anchor_galleries": anchor_evidence,
                "checked_items": len(rows)}, issues
    except (OSError, ValueError, KeyError) as exc:
        issues.append({"code": "grouped_policy_provenance", "error": str(exc)})
        return {"passed": False, "error": str(exc)}, issues


def audit(data, *, root=ROOT, workers=4):
    root, data = Path(root).resolve(), Path(data)
    if not data.is_absolute():
        data = root / data
    started, content = time.perf_counter(), data.read_bytes()
    rows, issues = [], []
    for line_number, line in enumerate(content.decode("utf-8-sig").splitlines(), 1):
        if line.strip():
            try:
                row = json.loads(line)
                if not isinstance(row, dict) or not row.get("item_id"):
                    raise ValueError("row requires item_id")
                rows.append(row)
            except (ValueError, TypeError) as exc:
                issues.append({"code": "catalog_json", "line": line_number, "error": str(exc)})
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(lambda row: audit_item(row, root), rows))
    counts = Counter()
    for result in results:
        counts.update(result["counts"])
        issues.extend(result["issues"])
    try:
        validate_object_splits(rows)
        validator = {"passed": True, "function": "curio.represent.validation.validate_object_splits"}
    except (ValueError, KeyError) as exc:
        validator = {"passed": False, "function": "curio.represent.validation.validate_object_splits", "error": str(exc)}
        issues.append({"code": "validate_object_splits", "error": str(exc)})
    relations = build_relation_report(rows, root, raw_metadata={result["item_id"]: result["raw"] for result in results})
    by_id = {row["item_id"]: row for row in rows}
    for item_id, count in Counter(row["item_id"] for row in rows).items():
        if count > 1:
            issues.append({"code": "duplicate_item_id", "item_id": item_id, "count": count})
    for entry in relations["dangling_duplicate_references"]:
        issues.append({"code": "dangling_duplicate_reference", **entry})
    for group in relations["groups"]:
        if len(group["splits"]) > 1:
            issues.append({"code": "object_group_cross_split", "group_id": group["group_id"],
                           "item_ids": group["item_ids"], "splits": group["splits"], "evidence": group["evidence"]})
    grouped = any(row.get("split_policy") == "official-relations-components-v2-heldout-priority" for row in rows)
    grouped_policy = None
    if grouped:
        grouped_policy, policy_issues = validate_grouped_policy(rows, root, {result["item_id"]: result["raw"] for result in results})
        issues.extend(policy_issues)
    for row in rows:
        seen, anchor = set(), row
        while anchor.get("near_duplicate_of") or anchor.get("duplicate_sha256_of"):
            target = anchor.get("near_duplicate_of") or anchor.get("duplicate_sha256_of")
            if target in seen:
                issues.append({"code": "duplicate_reference_cycle", "item_id": row["item_id"]})
                break
            seen.add(target)
            if target not in by_id:
                break
            anchor = by_id[target]
        expected = split_for(anchor.get("object_group") or anchor["item_id"])
        if not grouped and row.get("split") != expected:
            issues.append({"code": "deterministic_split", "item_id": row["item_id"], "actual": row.get("split"), "expected": expected})
    for kind in ("raw", "processed", "thumb"):
        hash_ids = {}
        for result in results:
            value = result["hashes"].get(kind)
            if not value:
                continue
            if value in hash_ids:
                first = hash_ids[value]
                if kind in {"raw", "processed"} or by_id[first]["split"] != by_id[result["item_id"]]["split"]:
                    cross_split = by_id[first]["split"] != by_id[result["item_id"]]["split"]
                    issues.append({"code": f"actual_{kind}_duplicate", "item_id": result["item_id"], "duplicate_of": first,
                                   "sha256": value, "cross_split": cross_split,
                                   "severity": "warning" if kind == "processed" and not cross_split else "error"})
            else:
                hash_ids[value] = result["item_id"]
        counts[f"{kind}_sha_unique"] = len(hash_ids)
    for issue in issues:
        issue.setdefault("severity", "error")
    error_count = sum(issue["severity"] == "error" for issue in issues)
    report = {"protocol": "curio-full-file-audit-v2", "audited_at": datetime.now(timezone.utc).isoformat(),
              "data": str(data.relative_to(root)).replace("\\", "/"), "data_sha256": digest(content), "rows": len(rows),
              "source_counts": dict(Counter(row.get("source") for row in rows)), "split_counts": dict(Counter(row.get("split") for row in rows)),
              "checks": dict(counts), "split_validator": validator, "grouped_split_policy": grouped_policy, "object_relations": relations,
              "rules": {"decode": "PIL.Image.open(...).load() on every raw/processed/thumb file", "raw_short_edge_min": 256,
                        "raw_long_edge_min": 512, "processed_long_edge_max": 1024, "thumb_long_edge_max": 320,
                        "resize": "PIL thumbnail dimensions, preserve aspect ratio, no upscaling",
                        "pre1912": "year_from<1912 inclusion; year_to>=1912 reported as range uncertainty without removing rows",
                        "rights": "Met isPublicDomain is true; Cleveland share_license_status equals CC0 in cached official JSON",
                        "origin": "explicit China/Chinese in raw origin fields; conflicting Met country rejected"},
              "issues": issues, "issue_counts": dict(Counter(issue["code"] for issue in issues)), "passed": error_count == 0,
              "error_count": error_count, "warning_count": len(issues) - error_count,
              "elapsed_seconds": round(time.perf_counter() - started, 3)}
    audit_path, license_path = root / "data/processed/audit.json", root / "data/licenses.csv"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fieldnames = list(results[0]["license"]) + ["audited_at", "data_sha256"] if results else ["item_id", "audited_at", "data_sha256"]
    with license_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow({**result["license"], "audited_at": report["audited_at"], "data_sha256": report["data_sha256"]})
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/snapshots/collection_v2.jsonl")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")
    report = audit(args.data, workers=args.workers)
    print(json.dumps({key: report[key] for key in ("data", "data_sha256", "rows", "source_counts", "split_counts", "checks", "issue_counts", "passed", "elapsed_seconds")}, ensure_ascii=False, indent=2))
    return int(not report["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
