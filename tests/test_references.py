import json

import pytest
from PIL import Image

from curio.retrieve.references import (
    GalleryExclusions, image_fingerprints, load_reference_bank, metadata_sha256,
    normalize_record, plan_reference_query, select_references,
)


def test_rule_planner_is_explicit_and_bilingual():
    plan = plan_reference_query("万历青花葫芦瓶")
    assert plan["planner"] == "bilingual_rules_v1"
    assert plan["is_llm"] is False
    assert plan["category_keys"] == ["blue_white", "gourd"]
    assert plan["required_eras"] == ["Wanli"]
    assert "Wanli double gourd vase" in plan["query_terms"]
    assert "Ming blue and white porcelain vase" in plan_reference_query("明代青花瓷瓶")["query_terms"]
    assert plan_reference_query("没有映射的词")['unmapped']


def test_rights_and_official_image_host_required():
    record = {"objectID": 1, "isPublicDomain": True,
              "primaryImage": "https://images.metmuseum.org/CRDImages/a.jpg"}
    assert normalize_record("met", record)["license"] == "CC0"
    for replacement in ({"isPublicDomain": False}, {"primaryImage": "https://evil.example/a.jpg"}):
        with pytest.raises(ValueError):
            normalize_record("met", {**record, **replacement})
    with pytest.raises(ValueError, match="CC0"):
        normalize_record("cleveland", {"id": 2, "share_license_status": "Copyright"})


def test_gallery_exclusion_all_splits_family_hash_and_perceptual():
    rows = [{"item_id": "cma_1", "source": "cleveland", "object_group": "cma_2",
             "split": "test", "sha256": "bytes", "phash": "0" * 256}]
    raw = {"cma_1": {"accession_number": "1900.7a", "related_works": [{"id": 3}]}}
    exclusion = GalleryExclusions(rows, raw)
    base = {"reference_id": "cma_99", "source": "cleveland", "accession": "2000.5"}
    cases = [({"reference_id": "cma_1"}, "gallery_item_id"),
             ({"reference_id": "cma_2"}, "gallery_object_family"),
             ({"reference_id": "cma_3"}, "official_related_work"),
             ({"related_ids": ["cma_1"]}, "official_related_work"),
             ({"accession": "1900.7b"}, "official_accession_family"),
             ({"accession": "1900.7.2"}, "official_accession_parent_record"),
             ({"image_sha256": "bytes"}, "exact_image_sha256"),
             ({"ahash256": "0" * 252 + "1111"}, "perceptual_ahash256")]
    for patch, expected in cases:
        assert exclusion.reason({**base, **patch})["reason"] == expected
    exclusion.add_gallery_phash("cma_1", "0000000000000000")
    assert exclusion.reason({**base, "phash64": "000000000000003f"})["reason"] == "perceptual_phash64"
    assert exclusion.reason(base) is None


def test_fingerprints_deterministic():
    image = Image.new("RGB", (512, 512), "red")
    assert image_fingerprints(image) == image_fingerprints(image.copy())
    assert len(image_fingerprints(image)["ahash256"]) == 256
    assert metadata_sha256({"a": 1, "b": 2}) == metadata_sha256({"b": 2, "a": 1})


def test_reference_selection_does_not_drop_era_constraint(tmp_path):
    folder = tmp_path / "data/references"
    folder.mkdir(parents=True)
    (folder / "example.jpg").write_bytes(b"image")
    row = {"reference_id": "met_99", "categories": ["blue_white"],
           "title": "Blue and white vase", "date": "Qianlong period (1736–95)",
           "image_path": "data/references/example.jpg"}
    (folder / "bank.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert load_reference_bank(folder) == load_reference_bank(tmp_path)
    assert select_references("乾隆青花瓶", tmp_path)["references"]
    missing = select_references("万历青花瓶", tmp_path)
    assert missing["references"] == []
    assert missing["constraint_status"] == "unverified_no_reference"
    assert not select_references("玉器", tmp_path)["references"]


def test_selection_requires_form_and_explicit_era_not_description(tmp_path):
    folder = tmp_path / "data/references"
    folder.mkdir(parents=True)
    (folder / "example.jpg").write_bytes(b"image")
    base = {"source": "met", "categories": ["blue_white"], "date": "Ming dynasty",
            "image_path": "data/references/example.jpg"}
    rows = [{**base, "reference_id": "met_1", "title": "Dish with flowers"},
            {**base, "reference_id": "met_2", "title": "Shard from a bottle"},
            {**base, "reference_id": "met_3", "title": "Bottle with peonies",
             "description": "Compared to Wanli porcelain"}]
    (folder / "bank.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    assert [r["reference_id"] for r in select_references("明代青花瓷瓶", tmp_path)["references"]] == ["met_3"]
    assert not select_references("万历青花瓷瓶", tmp_path)["references"]
    assert plan_reference_query("holding porcelain")['required_forms'] == []


def test_supplement_load_and_polychrome_qualification(tmp_path):
    folder = tmp_path / "data/references"
    folder.mkdir(parents=True)
    (folder / "image.jpg").write_bytes(b"image")
    rows = [{"reference_id": "cma_1", "source": "cleveland", "title": "Vase",
             "date": "Wanli reign", "categories": ["underglaze_polychrome", "porcelain"],
             "image_path": "data/references/image.jpg"},
            {"reference_id": "met_2", "source": "met", "title": "Gourd-shaped bottle",
             "period": "Ming dynasty, Jiajing period", "categories": ["blue_white", "gourd", "porcelain"],
             "image_path": "data/references/image.jpg"}]
    (folder / "bank.jsonl").write_text("", encoding="utf-8")
    (folder / "supplemental.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    assert len(load_reference_bank(folder)) == 2
    assert not select_references("万历青花瓶", tmp_path)["references"]
    qualified = select_references("万历青花五彩瓶", tmp_path)
    assert qualified["plan"]["category_keys"] == ["underglaze_polychrome"]
    assert qualified["references"][0]["reference_id"] == "cma_1"
    assert qualified["qualifications"]
    assert select_references("明代青花葫芦瓶", tmp_path)["references"][0]["reference_id"] == "met_2"
    assert select_references("嘉靖青花葫芦瓶", tmp_path)["references"][0]["reference_id"] == "met_2"
