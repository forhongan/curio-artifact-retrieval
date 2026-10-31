import math

import pytest

from curio.query import parse_query
from curio.retrieve.evidence import retrieval_evidence


def summary(query="vase", *, scores=(.8, .7), coverage=None, **kwargs):
    options = {
        "parsed": parse_query(query), "dense_scores": scores,
        "dense_available": True, "results_available": True,
        "probe_available": True,
        "probe_coverage": coverage or {"positive": [{"slot": "object_type", "value": "vase", "status": "covered"}]},
    }
    options.update(kwargs)
    return retrieval_evidence(**options)


def test_gap_uses_cosines_and_never_becomes_correctness_probability():
    near = summary(scores=(.800001, .8))
    far = summary(scores=(.9, -.5))
    assert near["ranking_separation"]["top1_top2_margin"] == pytest.approx(.000001)
    assert far["ranking_separation"]["top1_top2_margin"] == pytest.approx(1.4)
    assert near["level"] == far["level"] == "evidence_available"
    assert far["calibration"] == {"status": "uncalibrated", "correctness_probability": None}
    assert "normalized_margin" not in far["ranking_separation"]


def test_ties_and_single_candidate_do_not_invent_separation():
    assert summary(scores=(.7, .7))["ranking_separation"]["top1_top2_margin"] == 0
    single = summary(scores=(.7,))["ranking_separation"]
    assert single["candidate_count"] == 1
    assert single["top2"] is None
    assert single["top1_top2_margin"] is None


@pytest.mark.parametrize("scores", [(math.nan, .1), (.7, math.inf), (1.1, .8), (.1, .2)])
def test_invalid_cosines_or_wrong_order_are_rejected(scores):
    with pytest.raises(ValueError):
        summary(scores=scores)


def test_dense_unavailable_and_empty_results_stay_unknown():
    empty = summary(scores=(), results_available=False)
    assert empty["level"] == "unknown"
    assert empty["ranking_separation"]["top1"] is None
    missing = summary(scores=None, dense_available=False)
    assert missing["level"] == "unknown"
    assert missing["signal_availability"]["dense"] is False
    assert summary(results_available=False)["level"] == "unknown"
    with pytest.raises(ValueError):
        summary(dense_available=False)


def test_unparsed_intent_cannot_be_reported_as_full_coverage():
    evidence = summary("a mysterious ineffable thing")
    assert evidence["level"] == "limited"
    assert evidence["visual_coverage"]["total"] == 0
    assert evidence["visual_coverage"]["ratio"] is None
    assert evidence["visual_coverage"]["status"] == "unknown"
    assert evidence["visual_coverage"]["full_query_verified"] is False
    assert evidence["parser"]["confidence_calibrated"] is False


def test_historical_and_unsupported_conditions_reduce_coverage():
    evidence = summary("Ming vase with two handles around the rim")
    entries = evidence["visual_coverage"]["constraints"]
    assert any(entry["slot"] == "era" and entry["status"] == "unknown" for entry in entries)
    assert {entry["value"] for entry in entries if entry["slot"] == "unsupported"} == {"count", "spatial"}
    # The recognized handles shape is also unknown: 1 covered / 5 constraints.
    assert evidence["visual_coverage"]["ratio"] == .2
    assert evidence["level"] == "limited"


def test_only_actual_query_constraints_count_and_missing_probes_stay_unknown():
    coverage = {"positive": [
        {"slot": "object_type", "value": "vase", "status": "covered"},
        {"slot": "material", "value": "bronze", "status": "covered"},
    ]}
    assert summary(coverage=coverage)["visual_coverage"]["total"] == 1
    unavailable = summary(coverage=coverage, probe_available=False)
    assert unavailable["visual_coverage"]["covered"] == 0
    assert unavailable["visual_coverage"]["status"] == "unknown"
    partial = summary("porcelain vase", coverage=coverage)
    assert partial["visual_coverage"]["covered"] == 1
    assert partial["visual_coverage"]["ratio"] == .5


def test_component_limit_reason_is_preserved():
    evidence = summary("vase without lid", coverage={
        "positive": [{"slot": "object_type", "value": "vase", "status": "covered"}],
        "negations": [{"slot": "object_type", "value": "lid", "status": "unknown", "reason": "component_presence_not_trained"}],
    })
    entry = next(entry for entry in evidence["visual_coverage"]["constraints"] if entry["negated"])
    assert entry["status"] == "unknown"
    assert entry["reason"] == "component_presence_not_trained"


def test_probe_availability_is_not_agreement_and_weight_zero_is_not_application():
    evidence = summary(probe_weight_positive=False)
    assert evidence["signal_availability"] == {"dense": True, "probes": True, "probes_applied": False}
    assert evidence["signal_agreement"]["status"] == "not_measured"
    assert evidence["signal_agreement"]["fraction"] is None
    assert summary(probe_weight_positive=True)["signal_availability"]["probes_applied"] is True


def test_measured_agreement_uses_each_constraint_not_average_probability():
    coverage = {
        "positive": [{"slot": "object_type", "value": "vase", "status": "covered"}],
        "negations": [{"slot": "motifs", "value": "dragon", "status": "covered"}],
    }
    diagnostics = [
        {"slot": "object_type", "value": "vase", "dense_top_support": .1, "max_support": .1},
        # Constraint support for a negation is 1-P, supplied by the caller.
        {"slot": "motifs", "value": "dragon", "negated": True, "dense_top_support": .85, "max_support": .95},
    ]
    evidence = summary("vase without dragon", coverage=coverage, probe_top_support=diagnostics)
    agreement = evidence["signal_agreement"]
    assert agreement["status"] == "mixed"
    assert agreement["compared"] == 2
    assert agreement["agreeing"] == 1
    assert agreement["fraction"] == .5
    assert agreement["independent_validation"] is False
    assert evidence["signal_availability"]["probes_applied"] is False


def test_tied_probe_maximum_counts_as_agreement_but_not_validation():
    diagnostics = [{"slot": "object_type", "value": "vase", "dense_top_support": .2, "max_support": .2}]
    evidence = summary(probe_top_support=diagnostics)
    assert evidence["signal_agreement"]["status"] == "agree"
    assert evidence["signal_agreement"]["independent_validation"] is False


def test_unrelated_duplicate_and_unavailable_comparisons_do_not_inflate_agreement():
    row = {"slot": "object_type", "value": "vase", "dense_top_support": .4, "max_support": .6}
    unrelated = {"slot": "material", "value": "bronze", "dense_top_support": 1, "max_support": 1}
    result = summary(probe_top_support=[row, row, unrelated])
    assert result["signal_agreement"]["compared"] == 1
    assert result["signal_agreement"]["status"] == "disagree"
    assert summary(probe_top_support=[row], probe_available=False)["signal_agreement"]["status"] == "not_measured"
    assert summary(probe_top_support=[row], results_available=False)["signal_agreement"]["status"] == "not_measured"


@pytest.mark.parametrize("support,maximum", [(math.nan, .5), (.8, .7), (.8, 1.2), (-.1, .7)])
def test_invalid_support_diagnostics_are_rejected(support, maximum):
    with pytest.raises(ValueError):
        summary(probe_top_support=[{"slot": "object_type", "value": "vase", "dense_top_support": support, "max_support": maximum}])


def test_mapping_and_dataclass_parsed_inputs_have_same_semantics():
    parsed = parse_query("vase")
    assert summary(parsed=parsed.to_dict()) == summary(parsed=parsed)
    assert summary(degraded=True)["level"] == "limited"
