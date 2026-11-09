"""The research bridge must not silently drop query constraints or leak labels."""

import numpy as np

from curio.retrieve.attribute_cv import (
    attribute_scores,
    cascade_order,
    parse_attributes,
    predict_heads,
    stable_order,
)


def test_unsupported_attribute_rejects_whole_query():
    p = np.array([[0.9], [0.1]])
    score, missing = attribute_scores(
        p, ["colors/green"], [("colors", "green", 1), ("shape", "handles", 1)]
    )
    assert score is None and missing == ["shape/handles"]


def test_conjunction_negation_and_order_independent_predictions():
    classes = ["colors/green", "shape/handles"]
    p = np.array([[0.9, 0.9], [0.8, 0.1], [0.1, 0.1]])
    atoms = [("colors", "green", 1), ("shape", "handles", 0)]
    score, _ = attribute_scores(p, classes, atoms)
    assert stable_order(score, ["a", "b", "c"])[0] == 1
    permutation = [2, 0, 1]
    changed, _ = attribute_scores(p[permutation], classes, atoms)
    np.testing.assert_allclose(changed, score[permutation])


def test_fixed_parser_preserves_supported_negation():
    assert ("shape", "handles", 0) in parse_attributes("窄口且没有可见把手的器物")
    assert ("shape", "narrow_mouth", 1) in parse_attributes(
        "a narrow mouth vessel without visible handles"
    )
    assert ("shape", "handles", 0) in parse_attributes(
        "a narrow mouth vessel without visible handles"
    )


def test_cascade_cannot_recover_items_absent_from_recall_set():
    order = cascade_order(np.array([0.9, 0.8, 0.1]), np.array([0.1, 0.8, 1]), ["a", "b", "c"], 2)
    assert order.tolist() == [1, 0, 2]


def test_saved_linear_heads_predict_using_pixels_only():
    x = np.array([[1, 0], [0, 1]], dtype=float)
    p = predict_heads(x, np.array([[2, -2]]), np.array([0]))
    assert p.shape == (2, 1) and p[0, 0] > 0.8 and p[1, 0] < 0.2
