import json
from pathlib import Path

from curio.evaluate.assisted_review import judge_assisted, material_state


def test_obvious_wrong_object_is_no_even_if_color_cannot_be_seen():
    r = judge_assisted(
        {"forms": ["human_sculpture"], "note": "figure"}, {}, "v8_red_bowl", color_observable=False
    )
    assert r["grade"] == 0 and not r["decision"] and not r["needs_review"]
    assert any(e["condition"] == "form/bowl" and e["state"] == "no" for e in r["evidence"])


def test_catalogue_color_does_not_restore_gray_photo():
    r = judge_assisted(
        {"forms": ["bowl"], "red": True, "note": "gray photo"},
        {"medium": "Porcelain with red glaze"},
        "v8_red_bowl",
        color_observable=False,
    )
    assert r["grade"] == -1 and r["reason"] == "color_not_observable"
    assert not r["decision"] and not r["needs_review"]


def test_blue_white_also_requires_observable_color():
    r = judge_assisted(
        {"forms": ["vase"], "blue_white": True, "note": "gray photo"},
        {},
        "v8_blue_white_vase",
        color_observable=False,
    )
    assert r["grade"] == -1 and r["reason"] == "color_not_observable"


def test_material_is_explicitly_catalogue_evidence_and_ambiguity_is_preserved():
    r = judge_assisted(
        {"forms": ["bowl", "cup"], "note": "ambiguous form"},
        {"medium": "Porcelain", "source_url": "https://museum.example/object"},
        "v8_porcelain_cup",
        color_observable=True,
    )
    assert r["grade"] == -1 and r["needs_review"] and not r["decision"]
    assert r["evidence"][1]["basis"] == "catalogue_medium"
    assert r["evidence"][1]["catalogue_value"] == "Porcelain"
    assert material_state("porcelaneous stoneware", "porcelain") == "unknown"
    assert material_state("copper alloy", "bronze") == "unknown"
    assert material_state("", "porcelain") == "unknown"


def test_export_covers_queue_without_claiming_human_confirmation():
    root = Path(__file__).resolve().parents[1]
    base = root / "eval/boundary_assisted_v10"
    bundle = json.loads((base / "review_queue.json").read_text(encoding="utf-8"))
    assisted = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
    rows = list(map(json.loads, (base / "labels.jsonl").read_text(encoding="utf-8").splitlines()))
    assert assisted["bundle_id"] == bundle["bundle_id"]
    assert len(rows) == len({r["key"] for r in rows}) == len(bundle["rows"]) == 720
    assert {r["key"] for r in rows} == {r["key"] for r in bundle["rows"]}
    assert all(not r["human_confirmed"] and r["source"] == "ai_visual_plus_catalogue" for r in rows)
    assert all(type(r["decision"]) is bool and r["decision"] == (r["grade"] == 3) for r in rows)
    for row in rows:
        states = [e["state"] for e in row["evidence"]]
        assert row["grade"] == (0 if "no" in states else -1 if "unknown" in states else 3)
    assert not {r["object_group"] for r in rows if r["split"] == "dev"} & {
        r["object_group"] for r in rows if r["split"] == "test"
    }
