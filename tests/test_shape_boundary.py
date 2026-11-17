import numpy as np
import pytest

from curio.retrieve.shape_boundary import (
    GEOMETRY_NAMES,
    catalogue_conflict,
    decide_shape,
    feature_vector,
    fit_shape,
    geometry_features,
    predict_shape,
)
from curio.retrieve.shape_counterevidence import apply_counterevidence


def confident_head():
    n = 3 + len(GEOMETRY_NAMES)
    return {
        "mean": [0.0] * n,
        "scale": [1.0] * n,
        "coefficient": [0.0] * n,
        "intercept": 4.0,
        "threshold": 0.5,
    }


def test_vertical_waist_distinguishes_stacked_from_adjacent_bodies():
    y, x = np.mgrid[:200, :200]
    stacked = ((x - 100) ** 2 + (y - 55) ** 2 < 40**2) | ((x - 100) ** 2 + (y - 135) ** 2 < 48**2)
    stacked[55:136, 90:111] = True
    parallel = ((x - 55) ** 2 + (y - 100) ** 2 < 40**2) | ((x - 135) ** 2 + (y - 100) ** 2 < 48**2)
    parallel[90:111, 55:136] = True
    i = GEOMETRY_NAMES.index("vertical_waist_depth")
    assert geometry_features(stacked)[i] > 0.15
    assert geometry_features(parallel)[i] < 0.05
    assert geometry_features(stacked) == geometry_features(np.pad(stacked, 30))


def test_no_mask_is_flagged_not_automatically_a_negative():
    geometry = geometry_features(np.zeros((50, 50), bool))
    assert geometry == [0.0] * len(GEOMETRY_NAMES)
    result = decide_shape(
        "meiping", confident_head(), [2.0, 2.0, 2.0] + geometry, {"yes_no_logit": 2.0}, {}
    )
    assert result["yes"]


def test_counter_evidence_cannot_reverse_direction_after_fitting():
    x = np.zeros((8, 3 + len(GEOMETRY_NAMES)))
    x[:, :3] = np.arange(8)[:, None]
    head = fit_shape(x, [1, 1, 1, 1, 0, 0, 0, 0])
    assert min(head["coefficient"][:3]) >= 0
    assert predict_shape(head, x[7]) >= predict_shape(head, x[0])


def test_catalogue_veto_has_field_evidence_and_ignores_decorative_mentions():
    conflict = catalogue_conflict("bowl", {"title": "Wine Cup with Red Glaze"})
    assert conflict["field"] == "title" and conflict["value"] == "Wine Cup with Red Glaze"
    assert catalogue_conflict("bowl", {"title": "Bowl with cups and flowers"}) is None
    assert (
        catalogue_conflict("bowl", {"title": "Bowl", "description": "used with wine cups"}) is None
    )
    assert catalogue_conflict("bowl", {"title": "Cupboard"}) is None
    assert catalogue_conflict("gourd", {"title": "Wine Cup"}) is None
    assert catalogue_conflict("bowl", {"title": "酒杯"}) is not None


def test_cup_category_can_veto_but_cannot_create_visual_support():
    head = confident_head()
    x = [0.0] * len(head["mean"])
    d = decide_shape("bowl", head, x, {"yes_no_logit": 1.0}, {"object_name": "Cup"})
    assert d["visual_yes"] and not d["yes"] and d["catalogue_evidence"]
    head["intercept"] = -4.0
    d = decide_shape("bowl", head, x, {"yes_no_logit": 1.0}, {"object_name": "Bowl"})
    assert not d["yes"]


def test_parallel_vessel_veto_cannot_be_overridden_by_classifier():
    head = confident_head()
    x = [0.0] * len(head["mean"])
    d = decide_shape("gourd", head, x, {"yes_no_logit": -4.0}, {})
    assert not d["yes"] and d["reason"] == "vertical_topology_rejected"
    assert decide_shape("gourd", head, x, {"yes_no_logit": 4.0}, {})["yes"]


def test_invalid_visual_evidence_raises_instead_of_silent_no():
    old = {"positive": {"yes_no_logit": 1.0}, "counter": {"yes_no_logit": -1.0}}
    with pytest.raises(ValueError):
        feature_vector(old, {"yes_no_logit": float("nan")}, [0.0] * len(GEOMETRY_NAMES))
    with pytest.raises(ValueError):
        fit_shape(np.zeros((4, 3 + len(GEOMETRY_NAMES))), [1, 1, 1, 1])


def test_counterevidence_only_removes_conflicting_positives():
    clear = {"parallel": {"yes_no_logit": -3.0}, "accessory": {"yes_no_logit": -4.0}}
    assert apply_counterevidence(True, clear)["yes"]
    assert not apply_counterevidence(False, clear)["yes"]
    conflict = {**clear, "parallel": {"yes_no_logit": 3.0}}
    result = apply_counterevidence(True, conflict)
    assert not result["yes"] and result["contradictions"] == ["parallel"]


@pytest.mark.parametrize("invalid", [None, float("nan"), float("inf")])
def test_counterevidence_rejects_invalid_scores(invalid):
    with pytest.raises(ValueError):
        apply_counterevidence(
            True, {"parallel": {"yes_no_logit": invalid}, "accessory": {"yes_no_logit": -2.0}}
        )


def test_incomplete_counterevidence_is_not_silently_converted_to_no():
    with pytest.raises(ValueError):
        apply_counterevidence(True, {"parallel": {"yes_no_logit": -2.0}})
