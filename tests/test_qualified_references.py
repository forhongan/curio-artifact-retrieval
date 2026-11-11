import json
from pathlib import Path

from curio.retrieve.dynamic_references import plan_dynamic_query
from curio.retrieve.qualified_references import acquire_qualified, normalize_vam, source_support


def test_production_date_not_reign_mark_and_form_not_painted_subject():
    plan = plan_dynamic_query("Qianlong famille rose vase")
    row = {
        "title": "Vase",
        "medium": "Porcelain, famille rose enamels",
        "date_interval": [1736, 1795],
    }
    assert source_support(row, plan)["status"] == "source_supported"
    row["date_interval"] = [1850, 1900]
    row["description"] = "Qianlong reign mark on the base"
    assert source_support(row, plan)["status"] == "insufficient_source_support"
    row.update(title="Plate", description="A vase painted on the plate", date_interval=[1736, 1795])
    assert source_support(row, plan)["status"] == "insufficient_source_support"


def test_vam_contract_distinguishes_declared_image_and_actual_decode():
    path = Path(__file__).parent / "fixtures/vam_O165239.json"
    detail = json.loads(path.read_text(encoding="utf-8"))
    row = normalize_vam({"systemNumber": "O165239", "_primaryImageId": "2010EC3476"}, detail)
    assert row["api_reported_has_image"]
    assert row["date_interval"] == [1736, 1795]
    assert "download_verified" not in row
    assert (
        source_support(row, plan_dynamic_query("Qianlong famille rose vase"))["status"]
        == "source_supported"
    )


def test_source_failure_and_empty_results_never_fabricate_reference(tmp_path):
    class Acquirer:
        output = tmp_path
        root = tmp_path
        request_log = []

        def get_json(self, url, *args, **kwargs):
            if "artic.edu" in url:
                raise RuntimeError("HTTP 403")
            return {"records": []}

        def get_image(self, *args):
            raise AssertionError("No image should be requested")

    result = acquire_qualified(Acquirer(), "Qianlong vase", "Qianlong vase", sources=("aic", "vam"))
    assert result["status"] == "no_qualified_references"
    assert not result["accepted"]
    assert result["failures"][0]["source"] == "aic"


def test_specialist_form_in_description_requires_vessel_object_family():
    plan = plan_dynamic_query('Chinese meiping vase')
    row = {'title':'Vase', 'origin':'China', 'description':'A meiping with round shoulders and a narrow opening.'}
    assert source_support(row, plan)['status'] == 'source_supported'
    row.update(title='Painting', description='A painting depicting a meiping vase')
    assert source_support(row, plan)['status'] == 'insufficient_source_support'
