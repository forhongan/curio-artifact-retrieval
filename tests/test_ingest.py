import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "data_ingest", Path(__file__).resolve().parents[1] / "scripts" / "data_ingest.py"
)
data_ingest = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(data_ingest)
normalize_cle, normalize_met, years = (
    data_ingest.normalize_cle,
    data_ingest.normalize_met,
    data_ingest.years,
)


def test_years_prefers_numeric_fields_and_centuries():
    assert years(1785, 1790) == (1785, 1790)
    assert years("late 18th–early 19th century") == (1700, 1800)
    assert years(-200, -100) == (-200, -100)


def test_met_real_shape():
    row, image = normalize_met(
        {
            "objectID": 1000,
            "title": "Blue-and-white bowl",
            "objectName": "Bowl",
            "culture": "Chinese",
            "country": "China",
            "objectBeginDate": 1785,
            "objectEndDate": 1790,
            "medium": "Porcelain",
            "primaryImage": "https://x/a.jpg",
            "isPublicDomain": True,
            "tags": [{"term": "Flowers"}],
            "creditLine": "Credit",
        }
    )
    assert row["year_from"] == 1785 and row["year_to"] == 1790
    assert row["tags"] == ["Flowers"] and row["description"] == ""


def test_cleveland_real_shape():
    row, image = normalize_cle(
        {
            "id": 147576,
            "title": "Jade bird",
            "type": "Metalwork",
            "culture": ["China, Tang dynasty (618-907)"],
            "creation_date_earliest": 700,
            "creation_date_latest": 799,
            "technique": "bronze with silver inlaid lacquer",
            "description": "A description",
            "share_license_status": "CC0",
            "url": "https://clevelandart.org/art/1973.74",
            "images": {"print": {"url": "https://x/print.jpg"}},
        }
    )
    assert row["year_from"] == 700 and row["year_to"] == 799
    assert row["source_url"].endswith("1973.74")
    assert row["medium"].startswith("bronze")
