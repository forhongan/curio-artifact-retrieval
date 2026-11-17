import json
from pathlib import Path

import pytest

from curio.evaluate.targeted_boundary import (
    calibrate,
    policy_metrics,
    validate_partitions,
    positive_support_threshold,
)
from curio.retrieve.joint_query import compile_known


def row(key="q:1", split="dev", group="1", image_hash="h1", grade=3):
    return {
        "key": key,
        "split": split,
        "object_group": group,
        "image_sha256": image_hash,
        "grade": grade,
        "decision": grade == 3,
        "reason": "supported" if grade == 3 else "explicit_mismatch",
    }


@pytest.mark.parametrize(
    "other",
    [
        row("q2:2", "test", "1", "h2"),
        row("q2:2", "test", "2", "h1"),
    ],
)
def test_related_images_cannot_leak_between_queries_and_splits(other):
    with pytest.raises(ValueError, match="leaks"):
        validate_partitions([row(), other])


def test_calibration_rejects_test_labels_and_zero_positive_false_success():
    with pytest.raises(ValueError, match="dev"):
        calibrate([row(split="test")], [0.9])
    result = calibrate([row(grade=0)], [0.1])
    assert result["goal_met"] is False
    assert result["threshold"] == 0.5
    assert result["metrics"]["recall"] is None


def test_strict_policy_counts_unknown_color_acceptance_as_error():
    gray = dict(row(grade=-1), reason="color_not_observable")
    result = policy_metrics([gray], [True])
    assert result["fp"] == result["unknown_accepted"] == result["grayscale_accepted"] == 1
    assert result["recall"] is None


def test_positive_support_uses_other_objects_not_self_similarity():
    assert positive_support_threshold([[1.0, 0.0], [0.0, 1.0]]) == pytest.approx(-0.05)
    with pytest.raises(ValueError, match="two distinct"):
        positive_support_threshold([[1.0, 0.0]])


def test_targeted_labels_preserve_sources_and_split_isolation():
    root = Path(__file__).resolve().parents[1]
    rows = [
        json.loads(s)
        for s in (root / "eval/boundary_targeted_v10/labels.jsonl").read_text().splitlines()
    ]
    validate_partitions(rows)
    assert len(rows) == 155
    assert all(not r["human_confirmed"] and not r["independent_human_gold"] for r in rows)
    for qid in {r["qid"] for r in rows}:
        r = next(r for r in rows if r["qid"] == qid)
        assert compile_known(r["query_zh"]).key == compile_known(r["query_en"]).key
    # Keyword-based candidate mining must never silently become the answer.
    by_key = {r["key"]: r for r in rows}
    assert by_key["red_bowl:met_48523"]["grade"] == 0  # red clay, green visible glaze
    assert by_key["gourd:cma_140244"]["grade"] == 0  # title says gourd, photo is single-bellied
    assert by_key["meiping:met_52931"]["grade"] == 3  # generic Vase title, visible meiping form
