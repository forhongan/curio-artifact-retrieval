import json
from pathlib import Path

import pytest

from curio.evaluate.targeted_boundary import validate_partitions
from curio.retrieve.factor_boundary import FactorEvidence, and_decision, evidence_features, query_id


def test_red_bowl_requires_both_independent_conditions():
    thresholds = {"red": 0.7, "bowl": 0.8}
    assert not and_decision(
        "red_bowl", {"red": {"score": 0.99}, "bowl": {"score": 0.2}}, thresholds
    )["yes"]
    assert not and_decision(
        "red_bowl", {"red": {"score": 0.2}, "bowl": {"score": 0.99}}, thresholds
    )["yes"]
    assert and_decision("red_bowl", {"red": {"score": 0.9}, "bowl": {"score": 0.9}}, thresholds)[
        "yes"
    ]


def test_gray_photo_never_authorizes_color_even_with_high_model_score():
    result = and_decision(
        "red_bowl",
        {"red": {"score": 1.0, "color_excluded": True}, "bowl": {"score": 1.0}},
        {"red": 0.5, "bowl": 0.5},
    )
    assert not result["yes"]
    assert result["evidence"][0]["reason"] == "color_not_observable"


def test_inference_failure_is_not_silently_no():
    with pytest.raises(KeyError):
        and_decision("red_bowl", {"red": {"score": 0.8}}, {"red": 0.5, "bowl": 0.5})
    with pytest.raises(ValueError, match="Invalid"):
        and_decision("gourd", {"gourd": {"score": float("nan")}}, {"gourd": 0.5})


def test_failed_foreground_mask_preserves_learned_visual_evidence():
    features = evidence_features(
        {"reason": "model_score", "yes_no_logit": 4.0},
        {"reason": "model_score", "yes_no_logit": -3.0},
        {"mask_valid": False, "color_fractions": {}, "chromatic_fraction": 0.0},
        "red",
    )
    assert features[:4] == [4.0, -3.0, 7.0, 0.0]


@pytest.mark.parametrize(
    "zh,en,qid",
    [
        ("红色碗", "a red bowl", "red_bowl"),
        ("青花瓷", "blue and white porcelain", "blue_white"),
        ("梅瓶", "meiping vase", "meiping"),
        ("葫芦瓶", "double gourd vase", "gourd"),
    ],
)
def test_bilingual_queries_share_factor_plan(zh, en, qid):
    assert query_id(zh) == query_id(en) == qid


def test_unsupported_query_is_explicit_not_empty_result():
    with pytest.raises(ValueError, match="does not cover"):
        query_id("唐代马车")


def test_image_path_is_checked_before_pixel_read(tmp_path):
    from types import SimpleNamespace

    scorer = FactorEvidence(SimpleNamespace(root=tmp_path))
    with pytest.raises(ValueError, match="inside the project"):
        scorer.score("../outside.jpg", "red")


def test_fresh_holdouts_and_fitting_are_disjoint():
    root = Path(__file__).resolve().parents[1]
    base = root / "eval/boundary_factors_v11"
    components = [json.loads(s) for s in (base / "components.jsonl").read_text().splitlines()]
    validate_partitions(components)
    previous = []
    for name in [
        "labels.jsonl",
        "gallery_diagnostic_labels.jsonl",
        "support_diagnostic_labels.jsonl",
    ]:
        previous.extend(
            json.loads(s)
            for s in (root / "eval/boundary_targeted_v10" / name).read_text().splitlines()
        )
    groups = {r["object_group"] for r in previous}
    fresh = [json.loads(s) for s in (base / "labels.jsonl").read_text().splitlines()]
    assert len(fresh) == len({r["object_group"] for r in fresh}) == 96
    assert not groups & {r["object_group"] for r in fresh}
    assert all(r["source_split"] == "train" for r in components if r["split"] == "train")
    assert all(not r["human_confirmed"] for r in components)
