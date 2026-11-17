import json

import numpy as np
import pytest

from curio.evaluate.boundary_diagnostics import (
    choose_threshold,
    crossfit_boundary,
    fit_logistic,
    threshold_curve,
)
from curio.retrieve.condition_boundary import (
    condition_features,
    plan_texts,
    prompt_plan,
    support_scores,
)


def test_ties_cannot_be_split_to_inflate_oracle():
    result = threshold_curve([0.8, 0.8, 0.2], [3, 0, 0])
    assert len(result["points"]) == 3
    assert result["oracle"]["f1"] == pytest.approx(2 / 3)
    assert result["average_precision"] == pytest.approx(0.5)


def test_unknowns_do_not_become_negatives_or_threshold_training_examples():
    a = threshold_curve([0.8, 0.4], [3, 0])
    b = threshold_curve([0.8, 0.4, 0.99], [3, 0, -1])
    assert a == b
    assert choose_threshold(np.array([[0.8], [0.4], [0.99]]), np.array([[3], [0], [-1]])) == 0.8


def test_empty_set_and_undefined_cases():
    result = threshold_curve([0.8, 0.2], [0, 0])
    assert result["status"] == "no_positives"
    assert result["oracle"]["threshold"] > 0.8
    assert result["average_precision"] is None
    assert threshold_curve([0.2], [-1])["oracle"] is None
    with pytest.raises(ValueError, match="No known"):
        choose_threshold(np.array([[0.2]]), np.array([[-1]]))
    with pytest.raises(ValueError):
        threshold_curve([float("nan")], [3])


def test_threshold_matches_bruteforce_and_macro_not_micro():
    scores = np.array([[0.7, 0.8], [0.6, 0.3], [0.1, 0.2]])
    grades = np.array([[3, 0], [0, 3], [-1, 3]])
    candidates = [0.1, 0.2, 0.3, 0.6, 0.7, 0.8, np.nextafter(0.8, np.inf)]

    def objective(t):
        rows = []
        for j in range(2):
            yes, pos, known = scores[:, j] >= t, grades[:, j] == 3, grades[:, j] >= 0
            tp = sum(yes & pos)
            den = 2 * tp + sum(yes & known & ~pos) + sum(~yes & pos)
            rows.append(2 * tp / den if den else 0)
        return (np.mean(rows), t)

    assert choose_threshold(scores, grades) == max(candidates, key=objective)


def test_replacements_change_one_condition_and_keep_the_other():
    plan = prompt_plan([["form_family", "bottle_vase"], ["tag", "green"]], "en")
    assert len(plan["groups"]) == 2
    assert all("green" in t for t in plan["groups"][0]["alternatives"])
    assert all("bottle or vase" in t for t in plan["groups"][1]["alternatives"])
    assert "made of" not in plan["positive"]  # no implicit material added


def test_historical_conditions_are_missing_not_visual_proof():
    plan = prompt_plan([["material", "porcelain"], ["era", "Qianlong"]], "zh")
    assert plan["missing"] == ["era/Qianlong"]
    assert plan["condition_count"] == 2
    assert len(plan["groups"]) == 1
    assert not any("乾隆" in t for t in plan_texts(plan))


def test_features_expose_missing_evidence_and_contrast_strength():
    plan = prompt_plan([["material", "porcelain"], ["era", "ming"]], "en")
    scores = {t: np.array([0.2, 0.2]) for t in plan_texts(plan)}
    scores[plan["positive"]] = np.array([0.7, 0.3])
    x = condition_features([0.7, 0.3], [0.1, 0.1], plan, scores)
    assert x.shape == (2, 9)
    np.testing.assert_allclose(x[:, 5], [0.5, 0.1])
    np.testing.assert_allclose(x[:, 7], [0.5, 0.5])
    np.testing.assert_allclose(x[:, 8], [2, 2])
    empty = prompt_plan([["era", "ming"]], "en")
    x = condition_features([0.2, 0.2], [0.1, 0.1], empty, {})
    np.testing.assert_array_equal(x[:, 3:8], np.zeros((2, 5)))
    assert np.isfinite(x).all()


def test_fit_excludes_unknown_even_with_extreme_features():
    x = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    g = np.array([0, 1, 3, 3])
    a = fit_logistic(x, g)
    b = fit_logistic(np.r_[x, [[9999.0]]], np.r_[g, [-1]])
    assert a == b
    portable = json.loads(json.dumps(a))
    assert support_scores(x, portable)[0] < support_scores(x, portable)[-1]


def test_group_crossfit_is_disjoint_and_deterministic():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(12, 3, 2))
    grades = np.tile([0, 1, 3], (12, 1))
    groups = np.repeat([f"g{i}" for i in range(6)], 2)
    result = crossfit_boundary(x, grades, groups)
    assert result == crossfit_boundary(x, grades, groups)
    validation = []
    for fold in result["folds"]:
        assert not set(fold["train_groups"]) & set(fold["validation_groups"])
        validation.extend(fold["validation_groups"])
    assert sorted(validation) == sorted(set(groups))


def test_single_class_training_fails_loudly():
    with pytest.raises(ValueError, match="positive and negative"):
        fit_logistic(np.ones((3, 2)), np.array([3, 3, -1]))
