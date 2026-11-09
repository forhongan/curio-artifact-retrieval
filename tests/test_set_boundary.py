import numpy as np

from curio.evaluate.set_boundary import BoundaryConfig, combine_scores, set_metrics


def test_set_metrics_report_false_positives_and_negatives():
    values = set_metrics({"a", "b", "x"}, {"a", "b", "c"}, 10)
    assert values["selected_size"] == 3
    assert values["true_positives"] == 2
    assert values["false_positives"] == 1
    assert values["false_negatives"] == 1
    assert np.isclose(values["precision"], 2 / 3)
    assert np.isclose(values["recall"], 2 / 3)


def test_score_mix_and_attribute_requirement_define_the_yes_set():
    dense = np.array([0.9, 0.8, 0.7, 0.2])
    attrs = np.array([1.0, 0.0, 1.0, 1.0])
    config = BoundaryConfig(threshold=0.85, dense_weight=0.7, attribute_weight=0.3, require_attributes=True)
    assert combine_scores(dense, attrs, config).tolist() == [True, False, False, False]


def test_dense_percentile_is_a_recall_boundary():
    dense = np.array([0.9, 0.8, 0.7, 0.2])
    config = BoundaryConfig(threshold=0.0, min_dense_percentile=0.75)
    assert combine_scores(dense, None, config).tolist() == [True, False, False, False]
