from types import SimpleNamespace
import hashlib
import importlib
import json
import threading

from PIL import Image
import pytest

from curio.retrieve.catalogue_query import catalogue_fact, compile_catalogue
from curio.retrieve.catalogue_visual import CatalogueVisualScorer, mouth_body_ratio
from curio.retrieve.joint_sets import JointSetJobs


@pytest.mark.parametrize(
    "zh,en",
    [
        ("梅瓶", "meiping vase"),
        ("葫芦瓶", "gourd-shaped vase"),
        ("双腹葫芦瓶", "a double-gourd vase"),
        ("三节葫芦瓶", "a triple-gourd vase"),
        ("绿色三节葫芦瓶", "green triple-gourd vase"),
        ("红色碗", "a red bowl"),
        ("青花瓷", "blue and white porcelain"),
    ],
)
def test_bilingual_plans_preserve_meaning(zh, en):
    assert compile_catalogue(zh).key == compile_catalogue(en).key


def test_gourd_subtypes_are_not_collapsed_or_unknown_modifiers_dropped():
    assert len({compile_catalogue(q).key for q in ["葫芦瓶", "双腹葫芦瓶", "三节葫芦瓶"]}) == 3
    assert compile_catalogue("not a triple gourd vase") is None
    assert compile_catalogue("a triple gourd vase made yesterday") is None
    for query in ["双腹葫芦瓶", "三节葫芦瓶"]:
        assert set(compile_catalogue("葫芦瓶").clauses) < set(compile_catalogue(query).clauses)


def test_category_uses_object_identity_not_decorative_subject():
    assert (
        catalogue_fact({"title": "Figure with a gourd vase"}, "form/gourd_generic")["status"]
        == "contradicted"
    )

    assert catalogue_fact({"title": "Bowl with cups"}, "form_family/bowl")["status"] == "supported"
    assert (
        catalogue_fact({"title": "Wine Cup", "medium": "red porcelain"}, "form_family/bowl")[
            "status"
        ]
        == "contradicted"
    )
    assert (
        catalogue_fact(
            {"title": "Vase", "description": "Meiping and gourds are common"}, "form/meiping"
        )["status"]
        == "unknown"
    )


@pytest.mark.parametrize(
    "title", ["Dish", "Dishes", "Plate", "Brush Washer", "Water coupe", "Box", "Figures"]
)
def test_non_bowl_museum_categories_cannot_be_reinterpreted_as_bowls(title):
    assert catalogue_fact({"title": title}, "form_family/bowl")["status"] == "contradicted"


def test_specific_counts_conflict_with_category_but_generic_includes_them():
    item = {"title": "Triple-gourd vase"}
    assert catalogue_fact(item, "form/gourd_generic")["status"] == "supported"
    assert catalogue_fact(item, "form/gourd_double")["status"] == "contradicted"
    assert catalogue_fact(item, "form/gourd_triple")["status"] == "supported"


class Visual:
    def __init__(self, root):
        self.root = root
        self.identity = {"fixture": True}
        self.calls = []
        self.probability = 0.99

    def score(self, path, plan, **kwargs):
        self.calls.append(plan.canonical)
        return {
            "score": self.probability,
            "yes_no_logit": 5.0 if self.probability > 0.5 else -5.0,
            "answer_mass": 1.0,
            "reason": "scored",
            "cached": False,
        }

    def plan(self, q):
        raise RuntimeError("Unknown syntax must be explicitly planned")

    def close(self):
        pass


def setup(tmp_path, title):
    image = tmp_path / "object.jpg"
    Image.new("RGB", (64, 64), (128, 128, 128)).save(image)
    item = {
        "item_id": "fixture_1",
        "image_path": "object.jpg",
        "title": title,
        "medium": "Porcelain",
    }
    service = SimpleNamespace(root=tmp_path, items=[item], by_id={item["item_id"]: item})
    visual = Visual(tmp_path)
    return service, visual, CatalogueVisualScorer(service, visual)


def test_museum_identity_still_checks_photo_consistency(tmp_path):
    _, visual, scorer = setup(tmp_path, "Meiping vase")
    p = scorer.plan("梅瓶")
    r = scorer.score("object.jpg", p)
    assert r["score"] == 1 and {e["source"] for e in r["condition_evidence"]} == {
        "catalogue",
        "image",
    }
    assert not any("short neck" in q for q in visual.calls)
    visual.probability = 0.01
    assert scorer.score("object.jpg", p)["reason"] == "image_category_conflict"


def test_named_gourd_still_requires_visible_lobes(tmp_path):
    _, visual, scorer = setup(tmp_path, "Gourd Flask")
    original = visual.score

    def score(path, plan, **kwargs):
        visual.probability = 0.01 if "body lobes" in plan.canonical else 0.99
        return original(path, plan, **kwargs)

    visual.score = score
    result = scorer.score("object.jpg", scorer.plan("葫芦瓶"))
    assert result["score"] == 0 and result["reason"] == "image_gourd_body_conflict"


def test_catalogue_red_cannot_turn_grayscale_photo_red(tmp_path):
    _, visual, scorer = setup(tmp_path, "Red bowl")
    r = scorer.score("object.jpg", scorer.plan("红色碗"))
    assert r["score"] == 0 and r["reason"] == "color_not_observable" and not visual.calls


def test_category_conflict_does_not_invoke_visual_model(tmp_path):
    _, visual, scorer = setup(tmp_path, "Wine Cup with Monochrome Red Glaze")
    r = scorer.score("object.jpg", scorer.plan("红色碗"))
    assert r["reason"] == "catalogue_conflict" and not visual.calls


def test_metadata_changes_invalidate_job_identity(tmp_path):
    service, _, scorer = setup(tmp_path, "Meiping vase")
    before = scorer.identity
    service.items[0]["title"] = "Wine Cup"
    assert scorer.identity["catalogue_sha256"] != before["catalogue_sha256"]


def test_material_only_query_still_requires_consistent_photo(tmp_path):
    _, visual, scorer = setup(tmp_path, "Porcelain object")
    visual.probability = 0.01
    result = scorer.score("object.jpg", scorer.plan("瓷器"))
    assert result["score"] == 0 and result["reason"] == "image_material_conflict"


def test_adding_form_does_not_bypass_material_photo_check(tmp_path):
    _, visual, scorer = setup(tmp_path, "Vase")
    original = visual.score

    def score(path, plan, **kwargs):
        visual.probability = 0.01 if "ceramic or porcelain" in plan.canonical else 0.99
        return original(path, plan, **kwargs)

    visual.score = score
    assert scorer.score("object.jpg", scorer.plan("瓷瓶"))["reason"] == "image_material_conflict"


def test_narrow_mouth_requires_negative_wide_or_hidden_evidence(tmp_path):
    _, visual, scorer = setup(tmp_path, "Vase")
    # The positive prompt alone cannot admit a wide/hidden opening.
    result = scorer.score("object.jpg", scorer.plan("窄口瓷瓶"))
    assert result["score"] == 0 and result["reason"] == "visible_narrow_opening"
    assert len(visual.calls) == 2


def test_mouth_measurement_distinguishes_flared_rim_from_narrow_neck():
    import numpy as np

    narrow = np.zeros((120, 120), bool)
    narrow[10:45, 50:70] = True
    narrow[45:110, 20:100] = True
    assert mouth_body_ratio(narrow) == 0.25
    wide = narrow.copy()
    wide[10:18, 30:90] = True
    assert mouth_body_ratio(wide) == 0.75
    assert mouth_body_ratio(wide, valid=False) is None
    contaminated = wide.copy()
    contaminated[40:100, 70:118] = True
    assert mouth_body_ratio(contaminated) == 0.75


def test_blue_white_head_cannot_override_negative_direct_evidence(tmp_path, monkeypatch):
    _, _, scorer = setup(tmp_path, "Porcelain dish")
    module = importlib.import_module("curio.retrieve.catalogue_visual")
    monkeypatch.setattr(module, "color_observability", lambda _: {"observable": True})
    scorer._color_heads = lambda: (
        {
            "blue_white": {
                "mean": [0],
                "scale": [1],
                "coefficient": [0],
                "intercept": 20,
                "threshold": 0.5,
            }
        },
        {},
    )
    scorer.factor.score = lambda *args: {
        "features": [0],
        "color_excluded": False,
        "positive": {"yes_no_logit": -5},
        "counter": {"yes_no_logit": -1},
    }
    assert not scorer._color("object.jpg", "tag/blue_white", scorer.plan("青花瓷"))[0]


def test_frozen_support_cannot_exclude_a_changed_photo(tmp_path):
    service, _, scorer = setup(tmp_path, "Vase")
    item = service.items[0]
    item["image_sha256"] = hashlib.sha256((tmp_path / "object.jpg").read_bytes()).hexdigest()
    snapshot = tmp_path / "data/snapshots/collection_v2.jsonl"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_text(json.dumps(item) + "\n")
    index = tmp_path / "artifacts/targeted_boundary_v10/support_followup/meiping_gallery.jsonl"
    index.parent.mkdir(parents=True)
    index.write_text(json.dumps({"item_id": item["item_id"], "decision": False}) + "\n")
    assert not scorer._candidate_supported(item, "form/meiping")
    Image.new("RGB", (64, 64), "green").save(tmp_path / "object.jpg")
    assert scorer._candidate_supported(item, "form/meiping")


def test_malformed_support_is_not_all_no(tmp_path):
    service, _, scorer = setup(tmp_path, "Vase")
    snapshot = tmp_path / "data/snapshots/collection_v2.jsonl"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_text(json.dumps(service.items[0]) + "\n")
    index = tmp_path / "artifacts/targeted_boundary_v10/support_followup/meiping_gallery.jsonl"
    index.parent.mkdir(parents=True)
    index.write_text(json.dumps({"item_id": "fixture_1", "decision": "false"}) + "\n")
    assert scorer._candidate_supported(service.items[0], "form/meiping")


def test_surface_cache_invalidates_replaced_image(tmp_path, monkeypatch):
    import numpy as np

    _, _, scorer = setup(tmp_path, "Vase")
    module = importlib.import_module("curio.retrieve.catalogue_visual")
    calls = []

    def extract(path):
        calls.append(path)
        return {
            "mask": np.ones((64, 64), bool),
            "rgb": np.array(Image.open(path)),
            "valid": True,
            "features": [0.0] * 36,
        }

    monkeypatch.setattr(module, "extract_surface", extract)
    assert scorer._surface("object.jpg")["fractions"]["green"] == 0
    scorer._surface("object.jpg")
    assert len(calls) == 1
    Image.new("RGB", (64, 64), "green").save(tmp_path / "object.jpg")
    assert scorer._surface("object.jpg")["fractions"]["green"] == 1
    assert len(calls) == 2


def test_catalogue_http_routes_shared_queue_cancel_failure_and_shutdown(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from curio.retrieve.engine import SearchService

    service, visual, _ = setup(tmp_path, "Meiping vase")
    module = importlib.import_module("curio.serve.app")
    release, entered = threading.Event(), threading.Event()
    original_score = visual.score
    closed = []

    def score(*args, **kwargs):
        entered.set()
        if not release.wait(5):
            raise RuntimeError("timeout in fixture")
        return original_score(*args, **kwargs)

    visual.score = score
    visual.close = lambda: closed.append(True)
    legacy = JointSetJobs(service, visual)
    monkeypatch.setattr(
        module,
        "JointSetJobs",
        lambda svc, *args, **kwargs: JointSetJobs(svc, *args, **kwargs) if args else legacy,
    )
    app = module.create_app(service=SearchService(service.items, root=tmp_path), root=tmp_path)
    hybrid = app.state.catalogue_set_jobs
    try:
        with TestClient(app) as client:
            assert hybrid.pool is legacy.pool
            start = client.post(
                "/api/search/set/jobs", json={"query": "梅瓶", "method": "catalogue_visual"}
            )
            assert start.status_code == 202
            ident = start.json()["job_id"]
            assert ident.startswith("cv15_") and entered.wait(2)
            assert (
                "decisions" not in client.get(f"/api/search/set/jobs/{ident}?decisions=true").json()
            )
            assert (
                client.post(f"/api/search/set/jobs/{ident}/cancel").json()["status"] == "cancelled"
            )
            release.set()
            hybrid.futures[ident].result(timeout=5)
            restarted = client.post(
                "/api/search/set/jobs", json={"query": "meiping vase", "method": "catalogue_visual"}
            ).json()
            assert restarted["job_id"] == ident
            hybrid.futures[ident].result(timeout=5)
            result = client.get(f"/api/search/set/jobs/{ident}?decisions=true").json()
            assert (
                result["complete"]
                and result["matched_count"] == 1
                and result["uncertain_count"] == 0
            )
            assert result["boundary"]["version"] == "catalogue-visual-v15"
            assert result["decisions"][0]["condition_evidence"][0]["source"] == "catalogue"
            assert client.get("/api/search/set/jobs/cv15_missing").status_code == 404

            def fail(*args, **kwargs):
                raise RuntimeError("simulated inference failure")

            visual.score = fail
            failed = client.post(
                "/api/search/set/jobs", json={"query": "瓷器", "method": "catalogue_visual"}
            ).json()
            hybrid.futures[failed["job_id"]].result(timeout=5)
            failed = client.get(f"/api/search/set/jobs/{failed['job_id']}?decisions=true").json()
            assert (
                failed["status"] == "failed" and not failed["complete"] and "results" not in failed
            )
    finally:
        release.set()
    assert closed == [True]


def test_job_returns_complete_binary_decisions_and_field_provenance(tmp_path):
    service, _, scorer = setup(tmp_path, "Meiping vase")
    manager = JointSetJobs(service, scorer, method="catalogue_visual")
    try:
        start = manager.create("梅瓶")
        ident = start["job_id"]
        assert ident.startswith("cv15_")
        manager.futures[ident].result(timeout=5)
        result = manager.get(ident, include_decisions=True)
        assert (
            result["complete"]
            and result["method"] == "catalogue_visual"
            and result["matched_count"] == 1
        )
        assert len(result["decisions"]) == 1 and result["uncertain_count"] == 0
        assert result["results"][0]["condition_evidence"][0]["value"] == "Meiping vase"
        assert manager.create("meiping vase")["job_id"] == ident
    finally:
        manager.close()
