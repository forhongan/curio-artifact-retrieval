import math

import numpy as np
import pytest
from sklearn.metrics import ndcg_score

from curio.evaluate.metrics import average_precision, evaluate_ranking, ndcg, precision, recall
from curio.evaluate.stats import bootstrap_ci, holm_adjust, paired_permutation


def test_hand_calculated():
    qrels = {"a": 3, "b": 2, "c": 0, "d": 1}
    assert ndcg(["a", "b", "d", "c"], qrels) == 1
    assert precision(["b", "c", "a"], qrels, 3) == 2 / 3
    assert recall(["a"], qrels) == 0.5
    assert average_precision(["c", "a", "b"], qrels) == pytest.approx((0.5 + 2 / 3) / 2)
    assert ndcg(["x"], {"a": 0}) == 0
    with pytest.raises(ValueError, match="duplicate"):
        evaluate_ranking(["a", "a"], qrels)


def test_random_ndcg_matches_sklearn():
    rng = np.random.default_rng(5523)
    for _ in range(200):
        grades = rng.integers(0, 4, 30)
        ranking = rng.permutation(30).tolist()
        score = np.empty(30)
        score[ranking] = np.arange(30, 0, -1)
        # sklearn uses supplied y_true as gains; we use conventional 2^rel - 1.
        expected = ndcg_score((2.0**grades - 1)[None, :], score[None, :], k=10)
        assert ndcg(ranking, dict(enumerate(grades))) == pytest.approx(expected, abs=1e-10)


def test_statistics():
    assert bootstrap_ci([1, 1, 1])["lower"] == 1
    assert bootstrap_ci([0.1, 0.3, 0.5]) == bootstrap_ci([0.1, 0.3, 0.5])
    assert paired_permutation([0, 0], [0, 0])["p_value"] == 1
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert math.isfinite(bootstrap_ci([1])["mean"])
