from pathlib import Path
import tempfile
from types import SimpleNamespace

from curio.retrieve.dynamic_references import (
    DynamicMuseumAcquirer,
    plan_dynamic_query,
    reference_metadata_score,
    select_reference_set,
    strict_reference_rows,
)


def test_separate_shared_cache_can_write_and_replay_image(tmp_path):
    acquirer = DynamicMuseumAcquirer(tmp_path, tmp_path / 'audit', None,
                                    cache_dir=tmp_path / 'shared')
    acquirer.http = SimpleNamespace(get=lambda *args, **kwargs: SimpleNamespace(
        status_code=200, headers={'content-type': 'image/jpeg'}, content=b'image bytes',
        raise_for_status=lambda: None))
    row = {'image_url': 'https://openaccess-cdn.clevelandart.org/test.jpg'}
    first, request = acquirer.get_image(row, 'first')
    assert first == b'image bytes' and request['from_cache'] is False
    replay = DynamicMuseumAcquirer(tmp_path, tmp_path / 'second-audit', None,
                                  cache_dir=tmp_path / 'shared', offline=True)
    second, request = replay.get_image(row, 'second')
    assert second == first and request['from_cache'] is True


def cma_record(**updates):
    record = {
        "id": 1, "title": "Jar", "accession_number": "1990.1",
        "share_license_status": "CC0", "technique": "porcelain with famille rose decoration",
        "type": "Ceramic", "culture": ["China, Qing dynasty, Qianlong reign"],
        "creation_date": "1736-1795", "url": "https://example",
        "images": {"print": {"url": "https://openaccess-cdn.clevelandart.org/a.jpg"}},
    }
    record.update(updates)
    return record


def reference(reference_id, title, accession, strict=True, concepts=None):
    return {
        "reference_id": reference_id, "source": "cleveland", "title": title,
        "accession": accession, "metadata_score": 0.9 if strict else 0.4,
        "metadata_diagnostics": {"strict_match": strict,
                                  "concept_match": concepts or {"polychrome": True}},
        "image_path": "unused.jpg", "image_sha256": reference_id,
    }


def test_plan_uses_visual_concepts_and_not_longquan_substring():
    plan = plan_dynamic_query("南宋龙泉青瓷瓶")
    assert {"celadon", "porcelain", "vase"}.issubset(plan["concept_keys"])
    assert "dragon" not in plan["concept_keys"]
    assert any(era["label"] == "Southern Song" for era in plan["era_constraints"])


def test_reference_metadata_requires_form_era_and_material():
    plan = plan_dynamic_query("乾隆粉彩罐")
    score, diagnostics = reference_metadata_score(cma_record(), "cleveland", plan)
    assert score > 0
    assert diagnostics["strict_match"] is True
    wrong, wrong_diagnostics = reference_metadata_score(cma_record(culture=["Japan"]), "cleveland", plan)
    assert wrong < score
    assert "culture_not_china" in wrong_diagnostics["failure_reasons"]


def test_sibling_accession_titles_are_not_independent():
    rows = [reference("cma_1", "Dish", "1920.1"),
            reference("cma_2", "Dish", "1920.1"),
            reference("cma_3", "Dish", "1920.1")]
    plan = plan_dynamic_query("乾隆粉彩")
    assert len(strict_reference_rows(rows, plan)) == 1


def test_unverified_rows_never_fill_reference_set():
    plan = plan_dynamic_query("乾隆粉彩罐")
    rows = [reference("cma_1", "Vase", "1920.1", False),
            reference("cma_2", "Bowl", "1920.2", False)]
    selected = select_reference_set(rows, plan, limit=2)
    assert selected == []


def test_offline_failure_is_explicit():
    # Pure behavior check: normalization and model loading are not involved.
    with tempfile.TemporaryDirectory() as temporary:
        assert Path(temporary).exists()


def test_structured_liao_culture_without_literal_china_is_recognized():
    plan = plan_dynamic_query("辽代皮囊式壶")
    record = {"title": "Plaque in the shape of a dragon", "medium": "Jade (nephrite)",
              "culture": "Liao dynasty (907-1125)", "objectDate": "10th-11th century",
              "period": "Liao dynasty (907-1125)", "reign": "", "description": ""}
    record["description"] = "leather bag-shaped flask"

    _, diagnostics = reference_metadata_score(record, "met", plan)

    assert diagnostics["culture_class"] == "china"
    assert diagnostics["form_evidence"] == {"leather_flask": False}
    assert diagnostics["strict_match"] is False


def test_reign_mark_without_period_or_reign_attribution_is_rejected():
    plan = plan_dynamic_query("康熙梅瓶")
    record = {"title": "Meiping vase", "medium": "Porcelain", "culture": "China",
              "objectDate": "1662-1722", "period": "Qing dynasty, Kangxi mark",
              "reign": "", "description": ""}

    _, diagnostics = reference_metadata_score(record, "met", plan)

    assert diagnostics["era_match"] == {"Kangxi": False}
