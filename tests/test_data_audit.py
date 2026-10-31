import csv
import importlib.util
import json
from pathlib import Path

from PIL import Image
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
spec = importlib.util.spec_from_file_location("data_audit", SCRIPTS / "data_audit.py")
audit_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit_module)
relations = audit_module.build_relation_report


def fixture_item(root, sid="1", *, source="met", accession="1920.42", color="blue"):
    item_id = f"{'met' if source == 'met' else 'cma'}_{sid}"
    raw_dir = root / "data/raw" / source
    raw_dir.mkdir(parents=True, exist_ok=True)
    if source == "met":
        metadata = {"objectID": int(sid), "isPublicDomain": True, "country": "China", "culture": "Chinese",
                    "accessionNumber": accession, "objectBeginDate": 1700, "objectEndDate": 1799, "title": "Fixture"}
    else:
        metadata = {"id": int(sid), "share_license_status": "CC0", "culture": ["China"],
                    "accession_number": accession, "creation_date_earliest": 1700, "creation_date_latest": 1799,
                    "related_works": [], "title": "Fixture"}
    (raw_dir / f"{sid}.json").write_text(json.dumps(metadata), encoding="utf-8")
    row = {"item_id": item_id, "source": source, "source_id": sid, "license": "CC0", "year_from": 1700,
           "year_to": 1799, "width": 640, "height": 400, "split": audit_module.split_for(item_id), "object_group": item_id,
           "image_path": f"data/processed/images/{item_id}.jpg", "thumb_path": f"data/processed/thumbs/{item_id}.jpg"}
    for path, size, field in ((raw_dir / f"{sid}.jpg", (640, 400), "sha256"),
                              (root / row["image_path"], (640, 400), "image_sha256"),
                              (root / row["thumb_path"], (320, 200), "thumb_sha256")):
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", size, color).save(path, "JPEG")
        row[field] = audit_module.digest(path.read_bytes())
    return row, metadata


def write_catalog(root, rows):
    path = root / "data/snapshots/fixture.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_full_audit_decodes_hashes_and_writes_per_item_licenses_without_mutation(tmp_path):
    row, _ = fixture_item(tmp_path)
    path = write_catalog(tmp_path, [row])
    before = {str(p.relative_to(tmp_path)): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    report = audit_module.audit(path, root=tmp_path, workers=1)
    assert report["passed"] and report["issues"] == []
    assert report["split_validator"]["passed"]
    assert report["data_sha256"] == audit_module.digest(path.read_bytes())
    for kind in ("raw", "processed", "thumb"):
        assert report["checks"][f"{kind}_decoded"] == report["checks"][f"{kind}_sha256_verified"] == 1
    assert report["checks"]["official_rights_verified"] == report["checks"]["china_origin_verified"] == 1
    licenses = list(csv.DictReader((tmp_path / "data/licenses.csv").open(encoding="utf-8")))
    assert len(licenses) == 1 and licenses[0]["item_id"] == row["item_id"]
    assert licenses[0]["rights_field"] == "isPublicDomain" and licenses[0]["rights_value"] == "true"
    assert all((tmp_path / name).read_bytes() == content for name, content in before.items())
    new_files = {str(p.relative_to(tmp_path)).replace("\\", "/") for p in tmp_path.rglob("*") if p.is_file()} - {name.replace("\\", "/") for name in before}
    assert new_files == {"data/processed/audit.json", "data/licenses.csv"}


@pytest.mark.parametrize("failure,code", [("raw_sha", "raw_sha256"), ("derived_sha", "processed_sha256"),
                                          ("thumb_decode", "thumb_decode"), ("rights", "license"),
                                          ("origin", "origin"), ("dimensions", "processed_catalog_dimensions"),
                                          ("date", "date_range_crosses_1912")])
def test_audit_reports_actual_asset_and_raw_metadata_failures(tmp_path, failure, code):
    row, metadata = fixture_item(tmp_path, source="cleveland")
    if failure == "raw_sha":
        row["sha256"] = "wrong"
    elif failure == "derived_sha":
        row["image_sha256"] = "wrong"
    elif failure == "thumb_decode":
        (tmp_path / row["thumb_path"]).write_bytes(b"broken")
    elif failure == "rights":
        metadata["share_license_status"] = "Copyright"
    elif failure == "origin":
        metadata["culture"] = ["Japan"]
    elif failure == "dimensions":
        row["width"] = 999
    elif failure == "date":
        row["year_to"] = metadata["creation_date_latest"] = 1920
    (tmp_path / "data/raw/cleveland/1.json").write_text(json.dumps(metadata), encoding="utf-8")
    report = audit_module.audit(write_catalog(tmp_path, [row]), root=tmp_path, workers=1)
    assert report["passed"] is (failure == "date") and code in report["issue_counts"]
    if failure == "date":
        assert report["warning_count"] == 1 and report["error_count"] == 0


def test_accession_parts_related_work_and_existing_duplicate_edges_form_transitive_components(tmp_path):
    fixtures = [fixture_item(tmp_path, str(i), source="cleveland", accession=accession, color=color)
                for i, accession, color in ((1, "1920.42", "red"), (2, "1920.42.a-.d", "green"),
                                             (3, "1920.42.c", "blue"), (4, "1950.8", "yellow"),
                                             (5, "1960.3", "purple"))]
    rows, metadata = zip(*fixtures)
    rows = list(rows)
    rows[0]["split"], rows[1]["split"], rows[2]["split"] = "train", "dev", "test"
    metadata[2]["related_works"] = [{"id": 4, "relationship": None, "description": "Related whole set"}]
    rows[4]["phash"] = rows[3]["phash"] = "same"
    report = relations(rows, tmp_path, raw_metadata={row["item_id"]: raw for row, raw in zip(rows, metadata)})
    assert report["group_count"] == report["cross_split_group_count"] == 1
    assert set(report["groups"][0]["item_ids"]) == {row["item_id"] for row in rows}
    assert {entry["basis"] for entry in report["groups"][0]["evidence"]} >= {
        "official_accession_lettered_parts", "cma_accession_parent_record", "catalog_phash", "cma_official_related_works_conservative"}
    for split, row, alias in (("train", rows[0], "T001"), ("dev", rows[1], "D001"), ("test", rows[2], "V001")):
        path = tmp_path / f"eval/visual_{split}/blind_map.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps([{"item_id": row["item_id"], "alias": alias}]), encoding="utf-8")
    report = relations(rows, tmp_path, raw_metadata={row["item_id"]: raw for row, raw in zip(rows, metadata)})
    assert all(pair["overlap_groups"] == 1 for pair in report["blind_pair_overlaps"])
    assert len(report["blind_vs_full_other_splits"]) == 3


def test_accession_prefix_alone_does_not_collapse_separate_acquisitions_or_sources(tmp_path):
    fixtures = [fixture_item(tmp_path, "1", source="cleveland", accession="1920.42.1", color="red"),
                fixture_item(tmp_path, "2", source="cleveland", accession="1920.42.2", color="green"),
                fixture_item(tmp_path, "3", source="met", accession="1920.42.1", color="blue")]
    rows = [pair[0] for pair in fixtures]
    report = relations(rows, tmp_path, raw_metadata={row["item_id"]: raw for row, raw in fixtures})
    assert report["group_count"] == 3 and report["non_singleton_groups"] == 0


def test_cli_returns_nonzero_after_persisting_issues(monkeypatch):
    monkeypatch.setattr(audit_module, "audit", lambda *args, **kwargs: {key: False if key == "passed" else {} for key in
                        ("data", "data_sha256", "rows", "source_counts", "split_counts", "checks", "issue_counts", "passed", "elapsed_seconds")})
    assert audit_module.main(["--data", "fixture.jsonl"]) == 1


def test_grouped_v2_policy_preserves_anchor_and_rejects_mutated_metadata_or_heldout_train(tmp_path):
    fixtures = [fixture_item(tmp_path, str(i), source="cleveland", accession=accession, color=color)
                for i, accession, color in ((1, "1920.42", "red"), (2, "1920.42.a", "green"), (3, "1920.42.b", "blue"))]
    original = [entry[0] for entry in fixtures]
    for row, split in zip(original, ("train", "dev", "test")):
        row["split"] = split
    source = tmp_path / "data/snapshots/collection_v1.jsonl"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("".join(json.dumps(row) + "\n" for row in original), encoding="utf-8")
    for split, ids in (("dev", [original[1]["item_id"]]), ("test", [])):
        path = tmp_path / f"eval/visual_{split}/gallery_ids.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(ids), encoding="utf-8")
    grouped = [{**row, "previous_split": row["split"], "split": "dev", "object_group": "cma_1",
                "split_policy": "official-relations-components-v2-heldout-priority"} for row in original]
    raw = {row["item_id"]: metadata for row, metadata in fixtures}
    report, issues = audit_module.validate_grouped_policy(grouped, tmp_path, raw)
    assert report["passed"] and not issues
    assert report["groups"] == report["anchor_groups"] == 1
    grouped[0]["year_to"] = 1900
    grouped[2]["split"] = "train"
    report, issues = audit_module.validate_grouped_policy(grouped, tmp_path, raw)
    assert not report["passed"]
    assert {issue["code"] for issue in issues} >= {"grouped_non_split_data_changed", "grouped_split_policy", "heldout_moved_to_train"}
