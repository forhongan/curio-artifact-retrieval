import json

import numpy as np
from PIL import Image
import pytest

from curio.retrieve.dynamic_reference_runtime import (
    DynamicReferenceRuntime,
    reference_consensus,
    score_gallery,
    visual_reference_evidence,
)
from curio.retrieve.dynamic_references import DynamicMuseumAcquirer, plan_dynamic_query
from curio.retrieve.references import GalleryExclusions


def test_single_qualified_reference_scores_entire_gallery():
    gallery = np.array([[1, 0], [0, 1], [.8, .2]], dtype=np.float32)
    reference = np.array([[1, 0]], dtype=np.float32)
    result = score_gallery(["a", "b", "c"], gallery, gallery, reference, reference)
    assert result["consensus"]["state"] == "single_reference"
    assert result["orders"]["fusion"].tolist() == [0, 2, 1]
    assert set(result["orders"]["assisted"]) == {0, 1, 2}


def test_repeated_views_do_not_outvote_a_distinct_reference_mode():
    references = np.array([[1, 0], [1, 0], [0, 1]], dtype=np.float32)
    result = reference_consensus(references, references)
    assert result["visual_clusters"] == [[0, 1], [2]]
    assert result["reference_weights"] == [.25, .25, .5]
    assert result["state"] == "visually_diverse"


def test_auxiliary_unknown_keeps_slot_and_never_removes_gallery_items():
    gallery = np.array([[1, 0], [.9, .1], [.8, .2], [.7, .3]], dtype=np.float32)
    reference = np.array([[1, 0]], dtype=np.float32)
    result = score_gallery(["a", "b", "c", "d"], gallery, gallery, reference, reference,
                           auxiliary={"shape": np.array([0, np.nan, 1, .7])})
    assert result["orders"]["assisted"][1] == 1  # Unknown stays at its baseline slot.
    assert set(result["orders"]["assisted"]) == {0, 1, 2, 3}
    assert result["assistance"]["channels"]["shape"]["valid_gallery_count"] == 3
    assert result["assistance"]["status"] == "applied"


def test_unknown_mask_is_not_a_reference_contradiction(tmp_path):
    path = tmp_path / "blank.jpg"
    Image.new("RGB", (300, 400), "white").save(path)
    evidence, _ = visual_reference_evidence(path, plan_dynamic_query("Kangxi meiping"))
    assert evidence["status"] == "unknown"
    assert evidence["segmentation"]["status"] == "unknown"


def test_low_base_shadow_does_not_turn_tall_meiping_into_wide_object(tmp_path, monkeypatch):
    mask = np.zeros((200, 200), dtype=np.uint8)
    mask[15:40, 85:115] = 255
    mask[40:80, 60:140] = 255
    mask[80:170, 70:130] = 255
    mask[170:180, 10:190] = 255  # Attached low photographic shadow.
    monkeypatch.setattr("curio.retrieve.dynamic_reference_runtime.extract_visual_parts", lambda _: {
        "mask": mask, "diagnostics": {"status": "valid", "image_size": [200, 200]}})
    path = tmp_path / "object.jpg"
    Image.new("RGB", (200, 200), "gray").save(path)
    evidence, _ = visual_reference_evidence(path, plan_dynamic_query("Kangxi meiping"))
    assert evidence["status"] == "supported"
    assert evidence["checks"][0]["aspect"] > 1.15


def test_lateral_lighting_halo_causes_mask_abstention(tmp_path, monkeypatch):
    mask = np.zeros((200, 200), dtype=np.uint8)
    mask[15:40, 85:115] = 255
    mask[40:80, 60:140] = 255
    mask[80:130, 10:190] = 255  # The erroneous lateral mask wings observed in review.
    mask[130:180, 70:130] = 255
    monkeypatch.setattr("curio.retrieve.dynamic_reference_runtime.extract_visual_parts", lambda _: {
        "mask": mask, "diagnostics": {"status": "valid", "image_size": [200, 200]}})
    path = tmp_path / "object.jpg"
    Image.new("RGB", (200, 200), "gray").save(path)
    evidence, parts = visual_reference_evidence(path, plan_dynamic_query("Kangxi meiping"))
    assert evidence["status"] == "unknown"
    assert parts["validity"]["shape"] is False
    assert evidence["segmentation"]["runtime_reasons"] == ["lateral_mask_width_instability"]


def test_specific_famille_rose_does_not_accept_generic_enamel():
    from curio.retrieve.dynamic_references import reference_metadata_score
    plan = plan_dynamic_query("Qianlong famille rose porcelain")
    record = {"title": "Vase", "medium": "Porcelain with enamels", "culture": "China",
              "period": "Qianlong period", "objectDate": "1736-1795"}
    _, diagnostics = reference_metadata_score(record, "met", plan)
    assert not diagnostics["strict_match"]
    assert not diagnostics["concept_match"]["famille_rose"]


def test_blue_monochrome_is_not_blue_white_but_grayscale_is_unknown(tmp_path):
    path = tmp_path / "blue.jpg"
    Image.new("RGB", (300, 300), (0, 0, 180)).save(path)
    evidence, _ = visual_reference_evidence(path, plan_dynamic_query("Ming blue white porcelain"))
    assert evidence["status"] == "contradicted"
    assert evidence["checks"][0]["signal"] == "unsegmented_color_contradiction"


def test_plain_english_dynasty_and_unknown_query_do_not_crash():
    assert plan_dynamic_query("Ming blue white porcelain")["era_constraints"][0]["label"] == "Ming"
    assert plan_dynamic_query("a rare unknown object")["query_terms"] == ["a rare unknown object"]
    assert "leather_flask" not in plan_dynamic_query("Liao kendi")["concept_keys"]


def test_cached_request_provenance_records_replay(tmp_path):
    import hashlib
    url = "https://example.test/object"
    path = tmp_path / "http/object" / (hashlib.sha256(url.encode()).hexdigest() + ".json")
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"request": {"from_cache": False, "retrieved_at": "old",
                                           "request_url": url}, "response": {"id": 1}}))
    acquirer = DynamicMuseumAcquirer(tmp_path, tmp_path, GalleryExclusions([]), offline=True)
    assert acquirer.get_json(url, "object", "test") == {"id": 1}
    assert acquirer.request_log[0]["from_cache"] is True
    assert acquirer.request_log[0]["original_retrieved_at"] == "old"


def test_replay_description_cannot_silently_change(tmp_path, monkeypatch):
    runtime = DynamicReferenceRuntime(tmp_path)
    runtime._gallery = ([], GalleryExclusions([]))
    monkeypatch.setattr(runtime, "_load", lambda: None)
    path = tmp_path / "acquisition.json"
    path.write_text(json.dumps({"plan": {"input_query": "Ming blue white porcelain"}}))
    with pytest.raises(ValueError, match="exact input description"):
        runtime.run("Kangxi meiping", tmp_path / "out", mode="curated", acquisition=path)


def test_replayed_external_id_with_target_image_is_excluded_before_encoding(tmp_path, monkeypatch):
    import hashlib
    path = tmp_path / "reference.jpg"
    Image.new("RGB", (300, 400), "white").save(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata = {"id": 1, "title": "Meiping vase", "accession_number": "external.1",
                "share_license_status": "CC0", "technique": "Porcelain",
                "type": "Ceramic", "culture": ["China, Qing dynasty, Kangxi reign"],
                "creation_date": "1662-1722", "url": "https://example.test/1",
                "images": {"print": {"url": "https://openaccess-cdn.clevelandart.org/a.jpg"}}}
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps({"record": metadata}))
    target = {"item_id": "cma_2", "source": "cleveland", "source_id": "2",
              "sha256": digest, "image_path": "reference.jpg"}
    acquisition = tmp_path / "previous.json"
    acquisition.write_text(json.dumps({"plan": {"input_query": "Kangxi meiping"}, "accepted": [{
        "reference_id": "cma_1", "source": "cleveland", "metadata_path": "metadata.json",
        "image_path": "reference.jpg", "image_sha256": digest}]}))
    runtime = DynamicReferenceRuntime(tmp_path)
    runtime._gallery = ([target], GalleryExclusions([target]))
    monkeypatch.setattr(runtime, "_load", lambda: None)
    monkeypatch.setattr(runtime, "_encode", lambda *_: pytest.fail("excluded target reached encoder"))
    result = runtime.run("Kangxi meiping", tmp_path / "out", mode="curated", acquisition=acquisition)
    assert result["status"] == "no_qualified_reference"
    rejected = json.loads((tmp_path / "out/acquisition.json").read_text())["runtime_rejected"]
    assert rejected[0]["runtime_rejection"]["gallery_item_id"] == "cma_2"


def test_invalid_scoring_matrices_fail_before_ranking():
    reference = np.array([[1, 0]])
    with pytest.raises(ValueError, match="lengths/IDs"):
        score_gallery(["same", "same"], np.eye(2), np.eye(2), reference, reference)
    with pytest.raises(ValueError, match="zero"):
        score_gallery(["one"], np.array([[0, 0]]), reference, reference, reference)


def test_direct_exact_api_text_bypasses_taxonomy_and_masks(tmp_path, monkeypatch):
    import io
    binary = io.BytesIO()
    Image.new("RGB", (300, 400), "red").save(binary, format="PNG")
    image_path = tmp_path / "gallery.png"
    image_path.write_bytes(binary.getvalue())
    gallery = [{"item_id": f"gallery_{i}", "source": "met", "source_id": str(i),
                "image_path": "gallery.png"} for i in range(2)]
    runtime = DynamicReferenceRuntime(tmp_path)
    runtime._gallery = (gallery, GalleryExclusions(gallery))
    manifest = {"index_sha256": "test"}
    runtime._indices = ((np.eye(2), [row["item_id"] for row in gallery], manifest),) * 2
    monkeypatch.setattr(runtime, "_load", lambda: None)
    monkeypatch.setattr(runtime, "_encode", lambda *_: (np.array([[1., 0.]]), np.array([[1., 0.]]), {}))
    monkeypatch.setattr("curio.retrieve.dynamic_reference_runtime.extract_visual_parts",
                        lambda *_: pytest.fail("direct mode must not require masks"))
    calls = []

    def search(_self, url, *_args):
        calls.append(url)
        return {"data": [{"id": 99}, {"id": 100}]} if "cleveland" in url else {"objectIDs": []}

    # A deliberately off-topic API hit is retained: direct mode preserves API
    # relevance and reports source metadata, rather than enforcing our taxonomy.
    record = {"id": 99, "title": "Bronze ewer", "accession_number": "external.99",
              "share_license_status": "CC0", "technique": "Bronze", "type": "Metalwork",
              "culture": ["Japan"], "creation_date": "2020", "url": "https://example.test/99",
              "images": {"print": {"url": "https://openaccess-cdn.clevelandart.org/a.jpg"}}}
    monkeypatch.setattr(DynamicMuseumAcquirer, "get_json", search)
    monkeypatch.setattr(DynamicMuseumAcquirer, "object", lambda _self, _source, source_id: {
        **record, "id": int(source_id),
        "accession_number": "external.99" if source_id == "99" else "external.99.1"})
    monkeypatch.setattr(DynamicMuseumAcquirer, "get_image", lambda *_: (binary.getvalue(), {}))
    result = runtime.run("乾隆粉彩", tmp_path / "out", search_text="Qianlong famille rose porcelain")
    assert result["mode"] == "direct_api_text"
    assert result["reference_ids"] == ["cma_99"]
    assert result["original_description"] == "乾隆粉彩"
    assert result["search_text"] == "Qianlong famille rose porcelain"
    assert result["assistance"]["mask_prerequisite"] is False
    from urllib.parse import parse_qs, urlparse
    assert all(parse_qs(urlparse(url).query)["q"] == [result["search_text"]] for url in calls)
    trace = json.loads((tmp_path / "out/direct_acquisition.json").read_text())
    assert trace["raw_searches"][0]["returned_hit_ids_in_order"] == [99, 100]
    assert trace["candidate_decisions"][1]["exclusion"]["reason"] == "same_reference_accession_family"
    assert trace["metadata_relevance_filters"] == []
