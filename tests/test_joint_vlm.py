import json
import hashlib
import threading
from types import SimpleNamespace

from PIL import Image
import pytest

from curio.retrieve.joint_pixels import color_observability
from curio.retrieve.joint_query import compile_known, plan_free_query
from curio.retrieve.joint_sets import JointSetJobs, binary_decision
from curio.retrieve.joint_vlm import LocalVQA, validate_score


@pytest.mark.parametrize(
    "zh,en",
    [
        ("青花瓷", "blue and white porcelain"),
        ("梅瓶", "meiping vase"),
        ("葫芦瓶", "double-gourd vase"),
        ("绿色窄口瓷瓶", "green narrow-mouthed porcelain vase"),
        ("绿色梅瓶", "green meiping vase"),
        ("白色瓶子", "white vase"),
        ("玉琮", "jade cong"),
        ("抱月瓶", "moonflask"),
        ("敞口碗或水钵", "an open bowl or water bowl"),
    ],
)
def test_translation_pairs_share_exact_model_input_and_cache(zh, en):
    left, right = compile_known(zh), compile_known(en)
    assert left is not None and right is not None
    assert left.key == right.key
    assert left.question == right.question


def test_unknown_words_and_negation_reach_planner_without_being_discarded():
    calls = []

    def planner(text):
        calls.append(text)
        return {"canonical": "a red jar without handles", "requires_color": True, "visual": True}

    query = "红色没有把手的罐子"
    plan = plan_free_query(query, planner)
    assert calls == [query]
    assert "without handles" in plan.canonical
    assert plan.requires_color


def test_unverifiable_query_is_error_not_all_no():
    with pytest.raises(ValueError):
        plan_free_query("乾隆时期瓷器", lambda _: {"visual": False})


def test_new_phrasing_can_reuse_known_concept_after_translation():
    plan = plan_free_query(
        "一只带把手的罐子",
        lambda _: {
            "canonical": "a jar with handles",
            "requires_color": False,
            "visual": True,
        },
    )
    assert plan.key == compile_known("a jar with handles").key


def test_known_negative_feature_remains_visual_and_negation_is_preserved():
    plan = plan_free_query(
        "红色没有把手的罐子",
        lambda _: {
            "canonical": "A red jar without handles.",
            "requires_color": True,
            "visual": False,
        },
    )
    assert plan.canonical == "a red jar without handles"
    assert plan.requires_color


def test_free_query_cannot_suppress_explicit_color_guard():
    plan = plan_free_query(
        "白色灯笼",
        lambda _: {
            "canonical": "a white lantern",
            "requires_color": False,
            "visual": True,
        },
    )
    assert plan.requires_color


def test_visual_false_is_not_overridden_by_a_generic_translation():
    with pytest.raises(ValueError):
        plan_free_query(
            "乾隆时期瓷器",
            lambda _: {
                "canonical": "porcelain",
                "requires_color": False,
                "visual": False,
            },
        )


def test_all_color_conditions_include_black_white_and_blue_white():
    for q in ["white vase", "black vase", "青花瓷", "green vase"]:
        assert compile_known(q).requires_color
    assert not compile_known("梅瓶").requires_color


def test_gray_photo_cannot_be_certified_as_white_even_with_high_model_score(tmp_path):
    path = tmp_path / "gray.jpg"
    Image.new("RGB", (40, 40), (210, 210, 210)).save(path)
    assert not color_observability(path)["observable"]
    score = {"score": 0.999, "reason": "color_not_observable"}
    assert binary_decision(score) == (False, 0.0, "color_not_observable")


def test_grayscale_guard_does_not_require_model_process(tmp_path):
    directory = tmp_path / "artifacts/joint_vlm_v9"
    directory.mkdir(parents=True)
    (directory / "model_receipt.json").write_text(
        json.dumps({"repository": "test", "revision": "test"})
    )
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "joint_vlm_worker.py").write_text("# fixture")
    Image.new("RGB", (40, 40), "gray").save(tmp_path / "photo.png")
    scorer = LocalVQA(tmp_path)
    scorer._request = lambda _: pytest.fail("Gray photo must not enter color model")
    score = scorer.score("photo.png", compile_known("白色瓶子"))
    assert score["reason"] == "color_not_observable"
    assert scorer.score("photo.png", compile_known("white vase"))["cached"]


class FakeScorer:
    identity = {"revision": "fake"}

    def __init__(self):
        self.release = threading.Event()
        self.entered = threading.Event()
        self.calls = []
        self.fail = False
        self.closed = False

    def close(self):
        self.closed = True

    def plan(self, q):
        return compile_known(q)

    def score(self, path, query, *, references=()):
        self.calls.append((path, references))
        self.entered.set()
        self.release.wait(3)
        if self.fail:
            raise RuntimeError("simulated model failure")
        return {
            "score": 0.9 if path == "0.png" else 0.1,
            "yes_no_logit": 2 if path == "0.png" else -2,
            "reason": "model_score",
        }


@pytest.fixture
def jobs(tmp_path):
    items = []
    for i in range(3):
        Image.new("RGB", (20, 20), "red").save(tmp_path / f"{i}.png")
        items.append({"item_id": str(i), "image_path": f"{i}.png", "title": "not model input"})
    service = SimpleNamespace(root=tmp_path, items=items, by_id={r["item_id"]: r for r in items})
    scorer = FakeScorer()
    jobs = JointSetJobs(service, scorer)
    yield jobs, scorer
    scorer.release.set()
    jobs.pool.shutdown(wait=True)


def finish(jobs, ident):
    jobs.futures[ident].result(timeout=5)
    return jobs.get(ident, include_decisions=True)


def test_whole_gallery_binary_contract_and_same_job_for_languages(jobs):
    jobs, scorer = jobs
    start = jobs.create("梅瓶")
    scorer.entered.wait(2)
    assert not start["complete"] and "results" not in start
    assert jobs.create("meiping vase")["job_id"] == start["job_id"]
    scorer.release.set()
    end = finish(jobs, start["job_id"])
    assert end["complete"] and end["processed"] == end["total"] == 3
    assert end["matched_count"] == 1 and end["rejected_count"] == 2
    assert end["uncertain_count"] == 0
    assert [r["yes"] for r in end["decisions"]] == [True, False, False]
    assert all(
        set(r) <= {"item_id", "yes", "score", "reason", "raw_vqa_score", "cached"}
        for r in end["decisions"]
    )


def test_worker_error_is_not_empty_success_and_can_retry(jobs):
    jobs, scorer = jobs
    scorer.fail = True
    scorer.release.set()
    first = jobs.create("梅瓶")
    failed = finish(jobs, first["job_id"])
    assert failed["status"] == "failed" and not failed["complete"] and "results" not in failed
    scorer.fail = False
    again = jobs.create("梅瓶")
    assert finish(jobs, again["job_id"])["complete"]


def test_cancelled_partial_job_never_returns_complete_results(jobs):
    jobs, scorer = jobs
    start = jobs.create("梅瓶")
    scorer.entered.wait(2)
    jobs.cancel(start["job_id"])
    scorer.release.set()
    end = finish(jobs, start["job_id"])
    assert end["status"] == "cancelled" and not end["complete"] and "results" not in end


def test_reference_self_match_is_excluded(jobs):
    jobs, scorer = jobs
    jobs.service.items[0]["item_id"] = "met_48284"
    jobs.service.items[1]["item_id"] = "met_46727"
    jobs.service.by_id = {r["item_id"]: r for r in jobs.service.items}
    scorer.release.set()
    start = jobs.create("葫芦瓶", use_references=True)
    finish(jobs, start["job_id"])
    assert all(ref["image_path"] != path for path, refs in scorer.calls for ref in refs)
    assert len(scorer.calls[2][1]) == 2


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_score_fails_instead_of_becoming_no(score):
    with pytest.raises(ValueError):
        binary_decision({"score": score, "reason": "model_score"})
    with pytest.raises(RuntimeError):
        validate_score({"score": score, "reason": "model_score"})


def test_calibration_is_bound_to_model_and_exact_query(jobs):
    jobs, scorer = jobs
    plan = scorer.plan("梅瓶")
    path = jobs.directory.parent / "calibration.json"
    bundle = {
        "schema": "joint-calibration-v1",
        "model": scorer.identity,
        "query_keys": [plan.key],
        "coefficient": 0.5,
        "intercept": -1,
        "threshold": 0.7,
    }
    path.write_text(json.dumps(bundle))
    assert jobs._calibration(plan, []) == bundle
    assert jobs._calibration(scorer.plan("青花瓷"), []) is None
    assert jobs._calibration(plan, [{"item_id": "new-reference"}]) is None
    bundle["model"] = {"revision": "other"}
    path.write_text(json.dumps(bundle))
    assert jobs._calibration(plan, []) is None


def test_http_job_lifecycle_and_app_shutdown(jobs, monkeypatch):
    from fastapi.testclient import TestClient
    from curio.retrieve.engine import SearchService
    import importlib

    module = importlib.import_module("curio.serve.app")
    jobs, scorer = jobs
    real_manager = module.JointSetJobs
    monkeypatch.setattr(
        module,
        "JointSetJobs",
        lambda service, *args, **kwargs: real_manager(service, *args, **kwargs) if args else jobs,
    )
    service = SearchService(jobs.service.items, root=jobs.service.root)
    app = module.create_app(service=service, root=jobs.service.root)
    with TestClient(app) as client:
        response = client.post("/api/search/set/jobs", json={"query": "梅瓶"})
        assert response.status_code == 202
        ident = response.json()["job_id"]
        assert client.get("/api/search/set/jobs/unknown").status_code == 404
        partial = client.get(f"/api/search/set/jobs/{ident}?decisions=true").json()
        assert not partial["complete"] and "results" not in partial and "decisions" not in partial
        assert client.post("/api/search/set/jobs", json={"query": "  "}).status_code == 422
        scorer.release.set()
        finish(jobs, ident)
        complete = client.get(f"/api/search/set/jobs/{ident}?decisions=true").json()
        assert len(complete["decisions"]) == 3
        assert complete["matched_count"] + complete["rejected_count"] == complete["total"]
        assert client.post(f"/api/search/set/jobs/{ident}/cancel").json()["complete"]
        assert (
            client.post("/api/search/set/jobs", json={"query": "meiping vase"}).json()["job_id"]
            == ident
        )
    assert scorer.closed


def test_learned_reference_head_requires_exact_bank_hash(jobs):
    jobs, scorer = jobs
    plan = scorer.plan("梅瓶")
    path = jobs.directory.parent / "reference_bank.json"
    refs = [
        {"item_id": "0", "image_path": "0.png", "yes": True},
        {"item_id": "1", "image_path": "1.png", "yes": False},
    ]
    path.write_text(json.dumps({"references": {plan.key: refs}}))
    bundle = {
        "schema": "joint-calibration-v1",
        "model": scorer.identity,
        "query_keys": [plan.key],
        "coefficient": 0.7,
        "intercept": -1,
        "threshold": 0.6,
        "reference_bank_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "release_approved": False,
        "calibration_goal_met": False,
    }
    (path.parent / "reference_calibration.json").write_text(json.dumps(bundle))
    assert jobs._reference_adaptation(plan) is None
    bundle.update(release_approved=True, calibration_goal_met=True)
    (path.parent / "reference_calibration.json").write_text(json.dumps(bundle))
    assert jobs._reference_adaptation(plan) == (refs, bundle)
    assert jobs._reference_adaptation(scorer.plan("葫芦瓶")) is None
    scorer.release.set()
    start = jobs.create("梅瓶", use_references=True)
    end = finish(jobs, start["job_id"])
    assert end["adaptation"] == "fit_reference_bank" and end["calibration_applied"]
    assert "references" not in end  # not the unrelated dynamic-reference UI schema
    assert end["reference_item_ids"] == ["0", "1"]
    assert all(
        ref["image_path"] != image for image, references in scorer.calls for ref in references
    )
    path.write_text(path.read_text() + "\n")
    assert jobs._reference_adaptation(plan) is None
